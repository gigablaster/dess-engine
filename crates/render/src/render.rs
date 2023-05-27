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

use std::{
    collections::HashMap,
    mem::size_of,
    slice,
    sync::{Arc, Mutex},
};

use arrayvec::ArrayVec;
use ash::vk;
use dess_render_backend::{
    BackendError, Buffer, BufferDesc, CommandBuffer, DescriptorSetInfo, Device, Image, Instance,
    PhysicalDeviceList, Pipeline, RenderPassRecorder, SubImage, SubmitWaitDesc, Surface, Swapchain, PipelineVertex,
};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier, ImageLayout};

use crate::{
    descriptors::{DescriptorCache, DescriptorHandle},
    geometry::{CachedBuffer, GeometryCache, Index, StaticGeometry},
    uniforms::UniformBuffer,
    DescriptorAllocator, DescriptorSet, RenderError, RenderResult, Staging,
};

const STAGING_SIZE: usize = 64 * 1024 * 1024;
const GEOMETRY_CACHE_SIZE: usize = 64 * 1024 * 1024;
const DROP_LIST_DEFAULT_SIZE: usize = 128;
const DESCRIPTOR_SET_GROW: u32 = 128;

#[repr(C, align(16))]
pub struct RenderOp {
    pso: u32,
    vertex_offset: u32,
    index_offset: u32,
    index_count: u32,
    descs: [DescriptorHandle; 4],
}

pub enum GpuType {
    DiscreteOnly,
    PreferDiscrete,
    PrefereIntegrated,
}

pub struct RenderSystem {
    device: Arc<Device>,
    swapchain: Mutex<Swapchain>,
    staging: Mutex<Staging>,
    geo_cache: Mutex<GeometryCache>,
    current_drop_list: Mutex<DropList>,
    descriptor_cache: Mutex<DescriptorCache>,
    desciptor_allocator: Mutex<DescriptorAllocator>,
    drop_list: [Mutex<DropList>; 2],
}

pub struct RenderSystemDesc {
    pub debug: bool,
    pub gpu_type: GpuType,
    pub resolution: [u32; 2],
}

impl RenderSystemDesc {
    pub fn new(resolution: [u32; 2]) -> Self {
        Self {
            debug: false,
            gpu_type: GpuType::PreferDiscrete,
            resolution,
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
    geo_cache: &'a mut GeometryCache,
    descriptor_cache: &'a mut DescriptorCache,
}

impl<'a> UpdateContext<'a> {
    pub fn create_uniform(&mut self, set: &DescriptorSetInfo) -> RenderResult<DescriptorHandle> {
        self.descriptor_cache.create(set)
    }

    pub fn destroy_uniform(&mut self, handle: DescriptorHandle) {
        self.descriptor_cache.remove(handle);
    }

    pub fn set_uniform_buffer(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        buffer: UniformBuffer,
    ) {
        self.descriptor_cache
            .set_buffer(&mut self.drop_list, handle, binding, buffer);
    }

    pub fn set_uniform_image(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        image: &Image,
        layout: vk::ImageLayout,
        aspect: vk::ImageAspectFlags,
    ) {
        self.descriptor_cache
            .set_image(handle, binding, image, aspect, layout)
            .unwrap();
    }

    pub fn create_buffer<T: Sized>(&mut self, data: &[T]) -> RenderResult<CachedBuffer> {
        let buffer = self.geo_cache.allocate(data.len() * size_of::<T>())?;
        self.staging.upload_buffer(&buffer, data)?;

        Ok(buffer)
    }

    pub fn destroy_buffer(&mut self, buffer: CachedBuffer) {
        self.drop_list.drop_static_buffer(buffer);
    }
}

pub struct RenderContext<'a> {
    pub cb: &'a CommandBuffer,
    pub backbuffer: &'a Image,
    pub device: &'a ash::Device,
    geo_cache: &'a Buffer,
    descriptor_cache: &'a DescriptorCache,
}

impl<'a> RenderContext<'a> {
    pub fn render(
        &self,
        pipelines: &HashMap<u32, Pipeline>,
        pass: &RenderPassRecorder,
        rops: &[RenderOp],
        _name: Option<&str>,
    ) {
        if rops.is_empty() {
            return;
        }
        let mut current_descs = [DescriptorHandle::invalid(); 4];
        let mut current_pso_index = 0;
        let mut current_pso = None;
        pass.bind_vertex_buffer(self.geo_cache);
        pass.bind_index_buffer(self.geo_cache);
        rops.iter().for_each(|rop| {
            if current_pso_index != rop.pso {
                if let Some(pipeline) = pipelines.get(&rop.pso) {
                    current_pso_index = rop.pso;
                    current_pso = Some(pipeline);
                    unsafe {
                        self.device.cmd_bind_pipeline(
                            self.cb.raw,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline,
                        )
                    }
                } else {
                    return;
                }
            }
            let pipeline = current_pso.unwrap();
            rop.descs.iter().enumerate().for_each(|(index, desc)| {
                if current_descs[index] != *desc {
                    if let Some(value) = self.descriptor_cache.get(*desc) {
                        if let Some(descriptor) = &value.descriptor {
                            current_descs[index] = *desc;
                            let offsets = value
                                .buffers
                                .iter()
                                .map(|x| x.data.unwrap().offset)
                                .collect::<ArrayVec<_, 64>>();

                            unsafe {
                                self.device.cmd_bind_descriptor_sets(
                                    self.cb.raw,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline.pipeline_layout,
                                    index as _,
                                    slice::from_ref(descriptor.raw()),
                                    &offsets,
                                )
                            }
                        } else {
                            return;
                        }
                    } else {
                        return;
                    }
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
    descriptors: Vec<DescriptorSet>,
    uniforms: Vec<UniformBuffer>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            buffers: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
            descriptors: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
            uniforms: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
        }
    }
}

impl DropList {
    pub fn drop_static_buffer(&mut self, buffer: CachedBuffer) {
        self.buffers.push(buffer);
    }

    pub fn drop_descriptor_set(&mut self, descriptor: DescriptorSet) {
        self.descriptors.push(descriptor);
    }

    pub fn drop_uniform_buffer(&mut self, buffer: UniformBuffer) {
        self.uniforms.push(buffer);
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

impl RenderSystem {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        desc: RenderSystemDesc,
    ) -> RenderResult<Self> {
        let instance = Instance::builder()
            .debug(desc.debug)
            .build(display_handle)?;
        let surface = Surface::create(&instance, display_handle, window_handle)?;
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
            let swapchain = Swapchain::new(&device, surface, desc.resolution)?;
            let staging = Staging::new(&device, STAGING_SIZE)?;
            let geo_cache = GeometryCache::new(&device, GEOMETRY_CACHE_SIZE)?;
            let buffer = Arc::new(Buffer::graphics(
                &device,
                BufferDesc::upload(4 * 1024 * 1023, vk::BufferUsageFlags::UNIFORM_BUFFER),
                Some("uniforms"),
            )?);
            let descriptor_cache = DescriptorCache::new(&device, &buffer);
            let descriptor_allocator = DescriptorAllocator::new(DESCRIPTOR_SET_GROW);

            Ok(Self {
                device,
                swapchain: Mutex::new(swapchain),
                staging: Mutex::new(staging),
                geo_cache: Mutex::new(geo_cache),
                descriptor_cache: Mutex::new(descriptor_cache),
                current_drop_list: Mutex::new(DropList::default()),
                desciptor_allocator: Mutex::new(descriptor_allocator),
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
        let mut descriptors = self.desciptor_allocator.lock().unwrap();
        let mut descriptor_cache = self.descriptor_cache.lock().unwrap();
        let mut geo_cache = self.geo_cache.lock().unwrap();
        let context = UpdateContext {
            drop_list: &mut drop_list,
            staging: &mut staging,
            descriptor_cache: &mut descriptor_cache,
            geo_cache: &mut geo_cache
        };

        update_cb(context);

        descriptor_cache.update_descriptors(&mut descriptors, &mut drop_list)?;

        Ok(())
    }

    pub fn render_frame<F: FnOnce(RenderContext)>(
        &self,
        current_resolution: [u32; 2],
        frame_cb: F,
    ) -> RenderResult<()> {
        puffin::profile_scope!("render frame");
        let mut swapchain = self.swapchain.lock().unwrap();
        let mut geo_cache = self.geo_cache.lock().unwrap();
        let render_area = swapchain.render_area();
        if current_resolution[0] != render_area.extent.width
            || current_resolution[1] != render_area.extent.height
        {
            self.device.wait();
            swapchain.recreate(current_resolution)?;

            return Err(RenderError::RecreateBuffers);
        }

        let frame = self.device.begin_frame()?;
        self.drop_list[0].lock().unwrap().free(&mut geo_cache);
        let image = match swapchain.acquire_next_image() {
            Err(BackendError::RecreateSwapchain) => {
                self.device.wait();
                swapchain.recreate(current_resolution)?;

                return Err(RenderError::RecreateBuffers);
            }
            Err(err) => return Err(RenderError::Backend(err)),
            Ok(image) => image,
        };

        let mut staging = self.staging.lock().unwrap();
        staging.upload()?;
        staging.wait()?;
        {
            puffin::profile_scope!("main cb");

            let context = RenderContext {
                cb: &frame.main_cb,
                backbuffer: &image.image,
                geo_cache: &geo_cache.buffer,
                device: &self.device.raw,
                descriptor_cache: &self.descriptor_cache.lock().unwrap(),
            };

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
}
