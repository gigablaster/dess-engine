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
use buffer_allocator::{
    BufferCacheDesc, GeometryBufferCache, GeometryBufferHandle, UniformBufferCache,
    UniformBufferHandle,
};
use dess_render_backend::{
    create_pipeline_cache, BackendError, Buffer, BufferView, CommandBuffer, CommandBufferRecorder,
    DescriptorSetInfo, Device, FreeGpuResource, GpuAllocator, Image, Instance, PhysicalDeviceList,
    Pipeline, PipelineDesc, PipelineVertex, RenderPass, RenderPassLayout, RenderPassRecorder,
    Shader, ShaderDesc, SubImage, SubmitWaitDesc, Surface, Swapchain,
};

use gpu_descriptor_ash::AshDescriptorDevice;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use vk_sync::{AccessType, ImageBarrier, ImageLayout};

use crate::{
    descriptors::{DescriptorCache, DescriptorHandle},
    DescriptorAllocator, DescriptorSet, RenderError, RenderResult, Staging,
};

const STAGING_SIZE: usize = 64 * 1024 * 1024;
const DROP_LIST_DEFAULT_SIZE: usize = 128;
const DESCRIPTOR_SET_GROW: u32 = 128;

#[repr(C, align(16))]
pub struct RenderOp {
    pub pso: u32,
    pub vertex_buffer: GeometryBufferHandle,
    pub index_buffer: GeometryBufferHandle,
    pub index_count: u32,
    pub descs: [DescriptorHandle; 4],
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
    geometry_cache: Mutex<GeometryBufferCache>,
    uniform_cache: Mutex<UniformBufferCache>,
    current_drop_list: Mutex<DropList>,
    descriptor_cache: Mutex<DescriptorCache>,
    desciptor_allocator: Mutex<DescriptorAllocator>,
    drop_list: [Mutex<DropList>; 2],
    pipeline_cache: vk::PipelineCache,
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
    device: &'a ash::Device,
    allocator: &'a mut GpuAllocator,
    drop_list: &'a mut DropList,
    staging: &'a mut Staging,
    uniforms: &'a mut UniformBufferCache,
    geometry: &'a mut GeometryBufferCache,
    descriptor_cache: &'a mut DescriptorCache,
}

impl<'a> UpdateContext<'a> {
    pub fn create_uniform(&mut self, set: &DescriptorSetInfo) -> RenderResult<DescriptorHandle> {
        self.descriptor_cache.create(set)
    }

    pub fn destroy_uniform(&mut self, handle: DescriptorHandle) {
        self.descriptor_cache.remove(handle);
    }

    pub fn set_uniform_buffer<T: Sized>(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        data: &T,
    ) -> RenderResult<()> {
        let buffer = self
            .uniforms
            .allocate(self.device, self.allocator, size_of::<T>() as _)
            .unwrap();
        self.staging
            .upload_cached_buffer(buffer, &self.uniforms, slice::from_ref(data))?;
        self.descriptor_cache
            .set_buffer(self.drop_list, handle, binding, buffer, size_of::<T>());

        Ok(())
    }

    pub fn set_uniform_image(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        image: &Image,
        layout: vk::ImageLayout,
        aspect: vk::ImageAspectFlags,
    ) -> RenderResult<()> {
        self.descriptor_cache
            .set_image(handle, binding, image, aspect, layout)?;

        Ok(())
    }

    pub fn create_buffer<T: Sized>(&mut self, data: &[T]) -> RenderResult<GeometryBufferHandle> {
        let buffer = self
            .geometry
            .allocate(
                self.device,
                self.allocator,
                (data.len() * size_of::<T>()) as _,
            )
            .unwrap();
        self.staging
            .upload_cached_buffer(buffer, self.geometry, data)?;

        Ok(buffer)
    }

    pub fn destroy_buffer(&mut self, buffer: GeometryBufferHandle) {
        self.drop_list.drop_geometry_buffer(buffer);
    }
}

pub struct RenderContext<'a> {
    pub backbuffer: &'a Image,
    pub resolution: [u32; 2],
    pub graphics_queue: u32,
    pub transfer_queue: u32,
    cb: &'a CommandBuffer,
    device: &'a Device,
    descriptor_cache: &'a DescriptorCache,
    uniform_cache: &'a UniformBufferCache,
    geometry_cache: &'a GeometryBufferCache,
}

impl<'a> RenderContext<'a> {
    pub fn render(
        &self,
        pipelines: &HashMap<u32, Pipeline>,
        pass: &RenderPassRecorder,
        rops: &[RenderOp],
        name: Option<&str>,
    ) {
        if rops.is_empty() {
            return;
        }
        let mut current_descs = [DescriptorHandle::invalid(); 4];
        let mut current_pso_index = 0;
        let mut current_pso = None;
        let mut current_vertex_buffer = None;
        let mut current_index_buffer = None;

        let _label = name.map(|name| self.device.scoped_label(self.cb.raw, name));

        rops.iter().for_each(|rop| {
            if current_pso_index != rop.pso {
                if let Some(pipeline) = pipelines.get(&rop.pso) {
                    current_pso_index = rop.pso;
                    current_pso = Some(pipeline);
                    unsafe {
                        self.device.raw.cmd_bind_pipeline(
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
                                .map(|x| self.uniform_cache.resolve(x.data.unwrap().handle).offset)
                                .collect::<ArrayVec<_, 64>>();

                            unsafe {
                                self.device.raw.cmd_bind_descriptor_sets(
                                    self.cb.raw,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    pipeline.pipeline_layout,
                                    index as _,
                                    slice::from_ref(descriptor.raw()),
                                    &offsets,
                                )
                            }
                        }
                    }
                }
            });
            let vertex_buffer = self.geometry_cache.resolve(rop.vertex_buffer);
            let index_buffer = self.geometry_cache.resolve(rop.index_buffer);
            if Some(vertex_buffer.buffer) != current_vertex_buffer {
                pass.bind_vertex_buffer(vertex_buffer.buffer, 0);
                current_vertex_buffer = Some(vertex_buffer.buffer);
            }
            if Some(index_buffer.buffer) != current_index_buffer {
                pass.bind_index_buffer(index_buffer.buffer, 0);
                current_index_buffer = Some(index_buffer.buffer);
            }

            pass.draw(
                rop.index_count,
                1,
                index_buffer.offset / size_of::<u16>() as u32,
                vertex_buffer.offset as i32,
            );
        });
    }

    pub fn record<F: FnOnce(CommandBufferRecorder)>(&self, cb: F) -> RenderResult<()> {
        self.cb.record(&self.device.raw, cb)?;

        Ok(())
    }
}

pub(crate) struct DropList {
    geometry_buffers: Vec<GeometryBufferHandle>,
    uniform_buffers: Vec<UniformBufferHandle>,
    descriptors: Vec<DescriptorSet>,
}

impl Default for DropList {
    fn default() -> Self {
        Self {
            geometry_buffers: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
            uniform_buffers: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
            descriptors: Vec::with_capacity(DROP_LIST_DEFAULT_SIZE),
        }
    }
}

impl DropList {
    pub fn drop_geometry_buffer(&mut self, buffer: GeometryBufferHandle) {
        self.geometry_buffers.push(buffer);
    }

    pub fn drop_uniform_buffer(&mut self, buffer: UniformBufferHandle) {
        self.uniform_buffers.push(buffer);
    }

    pub fn drop_descriptor_set(&mut self, descriptor: DescriptorSet) {
        self.descriptors.push(descriptor);
    }

    pub fn free(
        &mut self,
        device: &ash::Device,
        descriptor_allocator: &mut DescriptorAllocator,
        geometry_cache: &mut GeometryBufferCache,
        uniform_cache: &mut UniformBufferCache,
    ) {
        self.geometry_buffers
            .drain(..)
            .for_each(|buffer| geometry_cache.deallocate(buffer));
        self.uniform_buffers
            .drain(..)
            .for_each(|buffer| uniform_cache.deallocate(buffer));
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(device),
                self.descriptors.drain(..),
            )
        };
        self.geometry_buffers.shrink_to(DROP_LIST_DEFAULT_SIZE);
        self.uniform_buffers.shrink_to(DROP_LIST_DEFAULT_SIZE);
        self.descriptors.shrink_to(DROP_LIST_DEFAULT_SIZE);
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
            let descriptor_cache = DescriptorCache::new(&device);
            let descriptor_allocator = DescriptorAllocator::new(DESCRIPTOR_SET_GROW);
            let uniform_cache = UniformBufferCache::new(
                BufferCacheDesc::default()
                    .family_index(device.graphics_queue.family.index)
                    .align(
                        device
                            .pdevice
                            .properties
                            .limits
                            .min_uniform_buffer_offset_alignment as _,
                    ),
            );
            let geometry_cache = GeometryBufferCache::new(
                BufferCacheDesc::default().family_index(device.graphics_queue.family.index),
            );

            Ok(Self {
                pipeline_cache: create_pipeline_cache(&device.raw)?,
                device,
                swapchain: Mutex::new(swapchain),
                staging: Mutex::new(staging),
                descriptor_cache: Mutex::new(descriptor_cache),
                current_drop_list: Mutex::new(DropList::default()),
                desciptor_allocator: Mutex::new(descriptor_allocator),
                drop_list: [
                    Mutex::new(DropList::default()),
                    Mutex::new(DropList::default()),
                ],
                uniform_cache: Mutex::new(uniform_cache),
                geometry_cache: Mutex::new(geometry_cache),
            })
        } else {
            Err(crate::RenderError::DeviceNotFound)
        }
    }

    pub fn update_resources<U, F: FnOnce(UpdateContext) -> U>(
        &self,
        update_cb: F,
    ) -> RenderResult<U> {
        puffin::profile_scope!("update resources");
        let mut staging = self.staging.lock().unwrap();
        let mut drop_list = self.current_drop_list.lock().unwrap();
        let mut descriptor_cache = self.descriptor_cache.lock().unwrap();
        let mut geometry_cache = self.geometry_cache.lock().unwrap();
        let mut uniform_cache = self.uniform_cache.lock().unwrap();
        let mut allocator = self.device.allocator();
        let context = UpdateContext {
            device: &self.device.raw,
            drop_list: &mut drop_list,
            staging: &mut staging,
            descriptor_cache: &mut descriptor_cache,
            geometry: &mut geometry_cache,
            uniforms: &mut uniform_cache,
            allocator: &mut allocator,
        };

        Ok(update_cb(context))
    }

    pub fn render_frame<F: FnOnce(RenderContext)>(
        &self,
        current_resolution: [u32; 2],
        frame_cb: F,
    ) -> RenderResult<()> {
        puffin::profile_scope!("render frame");
        let mut swapchain = self.swapchain.lock().unwrap();
        let mut geometry_cache = self.geometry_cache.lock().unwrap();
        let mut uniform_cache = self.uniform_cache.lock().unwrap();
        let mut descriptor_allocator = self.desciptor_allocator.lock().unwrap();

        let render_area = swapchain.render_area();
        if current_resolution[0] != render_area.extent.width
            || current_resolution[1] != render_area.extent.height
        {
            self.device.wait();
            swapchain.recreate(current_resolution)?;

            return Err(RenderError::RecreateBuffers);
        }

        let frame = self.device.begin_frame()?;
        self.drop_list[0].lock().unwrap().free(
            &self.device.raw,
            &mut descriptor_allocator,
            &mut geometry_cache,
            &mut uniform_cache,
        );
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
        let mut descriptor_cache = self.descriptor_cache.lock().unwrap();
        let mut drop_list = self.current_drop_list.lock().unwrap();

        descriptor_cache.update_descriptors(
            &mut descriptor_allocator,
            &mut drop_list,
            &uniform_cache,
        )?;

        staging.upload()?;
        staging.wait()?;
        {
            puffin::profile_scope!("main cb");

            let context = RenderContext {
                cb: &frame.main_cb,
                backbuffer: &image.image,
                device: &self.device,
                descriptor_cache: &descriptor_cache,
                resolution: current_resolution,
                graphics_queue: self.device.graphics_queue.family.index,
                transfer_queue: self.device.transfer_queue.family.index,
                geometry_cache: &geometry_cache,
                uniform_cache: &uniform_cache,
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
                let _label = self
                    .device
                    .scoped_label(frame.presentation_cb.raw, "Present");
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::ColorAttachmentWrite],
                    next_accesses: &[AccessType::Present],
                    previous_layout: ImageLayout::Optimal,
                    next_layout: ImageLayout::Optimal,
                    src_queue_family_index: self.device.graphics_queue.family.index,
                    dst_queue_family_index: self.device.graphics_queue.family.index,
                    discard_contents: false,
                    image: image.image.raw,
                    range: image.image.subresource_range(
                        SubImage::LayerAndMip(0, 0),
                        vk::ImageAspectFlags::COLOR,
                    ),
                };
                recorder.barrier(None, &[], &[barrier]);
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
        *self.drop_list[0].lock().unwrap() = std::mem::take::<DropList>(&mut drop_list);
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

    pub fn create_shader(&self, desc: ShaderDesc, name: Option<&str>) -> RenderResult<Shader> {
        Ok(Shader::new(&self.device, desc, name)?)
    }

    pub fn destriy_shader(&self, shader: Shader) {
        shader.free(&self.device.raw);
    }

    pub fn create_pipeline<T: PipelineVertex>(&self, desc: PipelineDesc) -> RenderResult<Pipeline> {
        Ok(Pipeline::new::<T>(
            &self.device,
            &self.pipeline_cache,
            desc,
        )?)
    }

    pub fn destroy_pipeline(&self, pipeline: Pipeline) {
        pipeline.free(&self.device.raw);
    }

    pub fn create_render_pass(&self, layout: RenderPassLayout) -> RenderResult<RenderPass> {
        Ok(RenderPass::new(&self.device.raw, layout)?)
    }

    pub fn destroy_render_pass(&self, render_pass: RenderPass) {
        render_pass.free(&self.device.raw);
    }

    pub fn back_buffer_format(&self) -> vk::Format {
        self.swapchain.lock().unwrap().backbuffer_format()
    }

    pub fn clear_fbos(&self, render_pass: &RenderPass) {
        render_pass.clear_fbos(&self.device.raw);
    }
}

impl Drop for RenderSystem {
    fn drop(&mut self) {
        self.device.wait();
        let mut current_drop_list = self.current_drop_list.lock().unwrap();
        let mut descriptors = self.desciptor_allocator.lock().unwrap();
        let mut geometry_cache = self.geometry_cache.lock().unwrap();
        let mut uniform_cache = self.uniform_cache.lock().unwrap();
        current_drop_list.free(
            &self.device.raw,
            &mut descriptors,
            &mut geometry_cache,
            &mut uniform_cache,
        );
        for drop_list in &self.drop_list {
            let mut drop_list = drop_list.lock().unwrap();
            drop_list.free(
                &self.device.raw,
                &mut descriptors,
                &mut geometry_cache,
                &mut uniform_cache,
            );
        }
        unsafe { descriptors.cleanup(AshDescriptorDevice::wrap(&self.device.raw)) };
    }
}
