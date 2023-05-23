// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{sync::{Arc, Mutex}, mem::size_of};

use ash::vk;
use dess_render_backend::{
    BackendError, Buffer, CommandBuffer, Device, Image, Instance, PhysicalDeviceList,
    PipelineVertex, SubImage, SubmitWaitDesc, Surface, Swapchain, RenderPassRecorder,
};
use sdl2::video::Window;
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier, ImageLayout};

use crate::{
    geometry::{GeometryCache, Index, StaticGeometry, CachedBuffer},
    RenderError, RenderResult, Staging,
};

const STAGING_SIZE: usize = 64 * 1024 * 1024;
const GEOMETRY_CACHE_SIZE: usize = 64 * 1024 * 1024;
const DROP_LIST_DEFAULT_SIZE: usize = 32;

#[repr(C, align(16))]
pub struct RenderOp {
    pso: u32,
    vertex_offset: u32,
    index_offset: u32,
    index_count: u32,
    descs: [u32; 4],
}

pub enum GpuType {
    DiscreteOnly,
    PreferDiscrete,
    PrefereIntegrated,
}

pub struct RenderSystem<'a> {
    window: &'a Window,
    device: Arc<Device>,
    swapchain: Mutex<Swapchain<'a>>,
    staging: Mutex<Staging>,
    geo_cache: Mutex<GeometryCache>,
    current_drop_list: Mutex<DropList>,
    drop_list: [Mutex<DropList>; 2],
}

pub struct RenderSystemDesc {
    pub debug: bool,
    pub gpu_type: GpuType,
}

impl RenderSystemDesc {
    pub fn new() -> Self {
        Self {
            debug: false,
            gpu_type: GpuType::PreferDiscrete,
        }
    }

    pub fn debug(mut self, value: bool) -> Self {
        self.debug = value;
        self
    }

    pub fn gpu_type(mut self, value: GpuType) -> Self {
        self.gpu_type = value;
        self
    }
}

pub struct UpdateContext<'a> {
    drop_list: &'a mut DropList,
    staging: &'a mut Staging,
}

pub struct RenderContext<'a> {
    pub cb: &'a CommandBuffer,
    pub image: &'a Image,
    geo_cache: &'a Buffer,
    device: &'a Device
}

impl<'a> RenderContext<'a> {
    pub fn render(&self, pass: &RenderPassRecorder, rops: &[RenderOp], name: Option<&str>) {
        if rops.is_empty() {
            return;
        }
        // TODO:: create some sort of Handle with not-valid state. For now we assume than 0 is no-state
        let mut current_descs = [0u32; 4];
        let mut current_pso = 0;
        pass.bind_vertex_buffer(self.geo_cache);
        pass.bind_index_buffer(self.geo_cache);
        rops.iter().for_each(|rop| {
            if current_pso != rop.pso {
                // TODO:: get and bind PSO
                current_pso = rop.pso;
            }
            rop.descs.iter().enumerate().for_each(|(index, desc)| {
                if current_descs[index] != *desc {
                    // TODO:: bind desc
                    current_descs[index] = *desc;
                }
            });
            assert_eq!(0, rop.index_offset % size_of::<Index>() as u32);
            let first_index = rop.index_offset / size_of::<Index>() as u32;
            pass.draw(rop.index_count, 0, first_index, rop.vertex_offset as _);
        });
    }
}

pub(crate) struct DropList {
    buffers: Vec<CachedBuffer>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            buffers: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
        }
    }
}

impl DropList {
    pub fn drop_static_buffer(&mut self, buffer: CachedBuffer) {
        self.buffers.push(buffer);
    }

    pub fn drop_static_geometry(&mut self, geometry: StaticGeometry) {
        self.drop_static_buffer(geometry.vertices);
        self.drop_static_buffer(geometry.indices);
    }

    pub fn free(&mut self, geo_cache: &mut GeometryCache) {
        self.buffers
            .drain(..)
            .for_each(|buffer| geo_cache.deallocate(buffer));
        self.buffers.shrink_to(DROP_LIST_DEFAULT_SIZE);
    }
}

impl<'a> RenderSystem<'a> {
    pub fn new(window: &'a Window, desc: RenderSystemDesc) -> RenderResult<Self> {
        let instance = Instance::builder().debug(desc.debug).build(window)?;
        let surface = Surface::create(&instance, window)?;
        let allowed_gpus = match desc.gpu_type {
            GpuType::PreferDiscrete => vec![
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ],
            GpuType::DiscreteOnly => vec![vk::PhysicalDeviceType::DISCRETE_GPU],
            GpuType::PrefereIntegrated => vec![
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::DISCRETE_GPU,
            ],
        };
        let pdevice = instance
            .enumerate_physical_devices()?
            .find_suitable_device(&surface, &allowed_gpus);
        if let Some(pdevice) = pdevice {
            let device = Device::create(instance, pdevice)?;
            let swapchain = Swapchain::new(&device, surface)?;
            let staging = Staging::new(&device, STAGING_SIZE)?;
            let geo_cache = GeometryCache::new(&device, GEOMETRY_CACHE_SIZE)?;

            Ok(Self {
                device,
                window,
                swapchain: Mutex::new(swapchain),
                staging: Mutex::new(staging),
                geo_cache: Mutex::new(geo_cache),
                current_drop_list: Mutex::new(DropList::default()),
                drop_list: [
                    Mutex::new(DropList::default()),
                    Mutex::new(DropList::default()),
                ],
            })
        } else {
            Err(crate::RenderError::DeviceNotFound)
        }
    }

    pub fn update_resources<F: FnOnce(UpdateContext)>(&self, update_cb: F) -> RenderResult<()> {
        puffin::profile_scope!("update resources");
        let mut staging = self.staging.lock().unwrap();
        let mut drop_list = self.current_drop_list.lock().unwrap();
        let context = UpdateContext {
            drop_list: &mut drop_list,
            staging: &mut staging
        };

        update_cb(context);

        Ok(())
    }

    pub fn render_frame<F: FnOnce(RenderContext)>(&self, frame_cb: F) -> RenderResult<()> {
        puffin::profile_scope!("render frame");
        let size = self.window.size();
        let mut swapchain = self.swapchain.lock().unwrap();
        let mut geo_cache = self.geo_cache.lock().unwrap();
        let render_area = swapchain.render_area();
        if size.0 != render_area.extent.width || size.1 != render_area.extent.height {
            self.device.wait();
            swapchain.recreate()?;

            return Err(RenderError::RecreateBuffers);
        }

        let frame = self.device.begin_frame()?;
        self.drop_list[0].lock().unwrap().free(&mut geo_cache);
        let image = match swapchain.acquire_next_image() {
            Err(BackendError::RecreateSwapchain) => {
                self.device.wait();
                swapchain.recreate()?;

                return Err(RenderError::RecreateBuffers);
            }
            Err(err) => return Err(RenderError::Backend(err)),
            Ok(image) => image,
        };

        let mut staging = self.staging.lock().unwrap();
        staging.upload()?;
        staging.wait()?;

        let context = RenderContext {
            cb: &frame.main_cb,
            image: &image.image,
            geo_cache: &geo_cache.buffer,
            device: &self.device
        };

        {
            puffin::profile_scope!("main cb");
            frame_cb(context);
            self.device.submit_render(
                &frame.main_cb,
                &[SubmitWaitDesc {
                    semaphore: image.acquire_semaphore,
                    stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                }],
                &[frame.render_finished],
            )?;
        }
        {
            puffin::profile_scope!("present cb");
            frame.presentation_cb.record(&self.device.raw, |recorder| {
            let barrier = ImageBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::Present],
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                src_queue_family_index: self.device.graphics_queue.family.index,
                dst_queue_family_index: self.device.graphics_queue.family.index,
                discard_contents: false,
                image: image.image.raw,
                range: image
                    .image
                    .subresource(SubImage::LayerAndMip(0, 0), vk::ImageAspectFlags::COLOR),
            };
            pipeline_barrier(&self.device.raw, *recorder.cb, None, &[], &[barrier]);
            })?;
            self.device.submit_render(
                &frame.presentation_cb,
                &[SubmitWaitDesc {
                    semaphore: frame.render_finished,
                    stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                }],
                &[image.presentation_finished],
            )?;
        }
        self.device.end_frame(frame)?;
        *self.drop_list[0].lock().unwrap() =
            std::mem::take::<DropList>(&mut self.current_drop_list.lock().unwrap());
        std::mem::swap(
            &mut self.drop_list[0].lock().unwrap(),
            &mut self.drop_list[1].lock().unwrap(),
        );
        {
            puffin::profile_scope!("present");
            swapchain.present_image(image);
        }

        Ok(())
    }

    pub fn create_static_geometry<T: PipelineVertex>(
        &self,
        vertices: &[T],
        indices: &[Index],
        name: Option<&str>,
    ) -> RenderResult<StaticGeometry> {
        let mut geo_cache = self.geo_cache.lock().unwrap();
        let geometry = geo_cache.create::<T>(vertices.len(), indices.len(), name)?;
        let mut staging = self.staging.lock().unwrap();
        staging.upload_buffer(&geometry.vertices, vertices)?;
        staging.upload_buffer(&geometry.indices, indices)?;

        Ok(geometry)
    }
}
