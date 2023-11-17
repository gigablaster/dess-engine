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

mod descriptor_storage;
mod error;
mod pipeline_cache;
mod staging;
mod uniforms;
pub mod vulkan;

use std::{marker::PhantomData, sync::Arc};

use ash::vk;
pub use descriptor_storage::*;
use dess_common::{Handle, HandleContainer};
pub use error::*;
use parking_lot::RwLock;
pub use pipeline_cache::*;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub use staging::*;
use vulkan::{
    Buffer, BufferDesc, Device, Image, ImageDesc, Instance, InstanceBuilder, PhysicalDeviceList,
    PipelineBuilder, Program, RenderPass, RenderPassLayout, ShaderDesc, Surface, Swapchain,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index<T> {
    value: u32,
    _phantom: PhantomData<T>,
}

impl<T> Index<T> {
    pub fn new(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn value(&self) -> u32 {
        self.value
    }
}

pub(crate) trait GpuResource {
    fn free(&self, device: &ash::Device);
}

pub type ImageHandle = Handle<vk::Image, Image>;
pub type BufferHandle = Handle<vk::Buffer, Buffer>;
pub type RenderPassHandle = Handle<vk::RenderPass, RenderPass>;

type ImageStorage = HandleContainer<vk::Image, Image>;
type BufferStorage = HandleContainer<vk::Buffer, Buffer>;
type RenderPassStorage = HandleContainer<vk::RenderPass, RenderPass>;

pub struct RenderBackend {
    device: Arc<Device>,
    swapchain: Swapchain,
    images: RwLock<ImageStorage>,
    buffers: RwLock<BufferStorage>,
    render_passes: RwLock<RenderPassStorage>,
    descriptors: RwLock<DescriptorStorage>,
    pipeline_cache: PipelineCache,
    staging: Staging,
}

pub struct DescriptorUpdateContext<'a> {
    storage: &'a mut DescriptorStorage,
    images: &'a ImageStorage,
    buffers: &'a BufferStorage,
}

impl<'a> DescriptorUpdateContext<'a> {
    pub fn create(
        &mut self,
        program: &Program,
        index: usize,
    ) -> Result<DescriptorHandle, BackendError> {
        self.storage.create(program.descriptor_set(index))
    }

    pub fn bind_image(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Result<(), BackendError> {
        self.storage.set_image(
            handle,
            binding,
            self.images
                .get_cold(image)
                .ok_or(BackendError::InvalidHandle)?,
            layout,
        )
    }

    pub fn set_uniform<T>(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        data: &T,
    ) -> Result<(), BackendError> {
        self.storage.set_uniform(handle, binding, data)
    }

    pub fn bind_dynamic_uniform_buffer(
        &mut self,
        handle: DescriptorHandle,
        binding: usize,
        buffer: BufferHandle,
    ) -> Result<(), BackendError> {
        self.storage.set_dynamic_uniform(
            handle,
            binding,
            self.buffers
                .get_cold(buffer)
                .ok_or(BackendError::InvalidHandle)?,
        )
    }

    pub fn invalidate_image(&mut self, handle: DescriptorHandle, binding: usize) {
        self.storage.invalidate_image(handle, binding);
    }

    pub fn invalidate_uniform(&mut self, handle: DescriptorHandle, binding: usize) {
        self.storage.invalidate_uniform(handle, binding)
    }

    pub fn invalidate_dynamic_uniform_buffer(&mut self, handle: DescriptorHandle, binding: usize) {
        self.storage.invalidate_dynamic_uniform(handle, binding)
    }
}

impl RenderBackend {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        dims: [u32; 2],
    ) -> Result<Self, BackendError> {
        let builder = InstanceBuilder::default();
        #[cfg(debug_assertions)]
        let builder = builder.debug(true);
        let instance = Instance::new(builder, display_handle)?;
        let surface = Surface::new(&instance, display_handle, window_handle)?;
        let pdevice = instance
            .enumerate_physical_devices()?
            .find_suitable_device(
                &surface,
                &[
                    vk::PhysicalDeviceType::DISCRETE_GPU,
                    vk::PhysicalDeviceType::INTEGRATED_GPU,
                ],
            )
            .ok_or(BackendError::NoSuitableDevice)?;
        let device = Device::new(instance, pdevice)?;
        let swapchain = Swapchain::new(&device, surface, dims)?;
        let images = Default::default();
        let buffers = Default::default();
        let render_passes = Default::default();
        let descriptors = RwLock::new(DescriptorStorage::new(&device)?);
        let pipeline_cache = PipelineCache::new(&device)?;
        let staging = Staging::new(&device, 32 * 1024 * 1024)?;

        Ok(Self {
            device,
            swapchain,
            images,
            buffers,
            render_passes,
            descriptors,
            pipeline_cache,
            staging,
        })
    }

    pub fn create_image(
        &self,
        desc: ImageDesc,
        initial_data: Option<&[ImageSubresourceData]>,
    ) -> Result<ImageHandle, BackendError> {
        let image = Image::texture(&self.device, desc)?;
        if let Some(initial_data) = initial_data {
            self.staging.upload_image(&image, initial_data)?;
        }
        Ok(self.images.write().push(image.raw(), image))
    }

    pub fn update_image(
        &self,
        handle: ImageHandle,
        data: &[ImageSubresourceData],
    ) -> Result<(), BackendError> {
        self.staging.upload_image(
            self.images
                .read()
                .get_cold(handle)
                .ok_or(BackendError::InvalidHandle)?,
            data,
        )
    }

    pub fn destroy_image(&self, handle: ImageHandle) {
        self.images.write().remove(handle);
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> Result<BufferHandle, BackendError> {
        let buffer = Buffer::new(&self.device, desc)?;
        Ok(self.buffers.write().push(buffer.raw(), buffer))
    }

    pub fn update_buffer<T>(
        &self,
        handle: BufferHandle,
        offset: u32,
        data: &[T],
    ) -> Result<(), BackendError> {
        self.staging.upload_buffer(
            self.buffers
                .read()
                .get_cold(handle)
                .ok_or(BackendError::InvalidHandle)?,
            offset,
            data,
        )
    }

    pub fn destroy_buffer(&self, handle: BufferHandle) {
        self.buffers.write().remove(handle);
    }

    pub fn create_render_pass(
        &self,
        layout: RenderPassLayout,
        builder: PipelineBuilder,
    ) -> Result<RenderPassHandle, BackendError> {
        let render_pass = RenderPass::new(&self.device, layout, builder, &self.pipeline_cache)?;
        Ok(self
            .render_passes
            .write()
            .push(render_pass.raw(), render_pass))
    }

    pub fn create_program(&self, shaders: &[ShaderDesc]) -> Result<Arc<Program>, BackendError> {
        Ok(Arc::new(Program::new(&self.device, shaders)?))
    }

    pub fn update_descriptors<F: FnOnce(DescriptorUpdateContext) -> Result<(), BackendError>>(
        &self,
        callback: F,
    ) -> Result<(), BackendError> {
        let context = DescriptorUpdateContext {
            storage: &mut self.descriptors.write(),
            images: &self.images.read(),
            buffers: &self.buffers.read(),
        };
        callback(context)
    }
}
