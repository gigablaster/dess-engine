// Copyright (C) 2024 gigablaster

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

use std::{mem, slice, sync::Arc};

use ash::vk;
use dess_backend::{
    AsVulkan, Buffer, BufferCreateDesc, Device, Image, ImageCreateDesc, ImageSubresourceData,
    ImageViewDesc, Program,
};
use dess_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy};
use parking_lot::{Mutex, RwLock};

use crate::{
    staging::{Staging, StagingDesc},
    temp::TempBuffer,
    BufferSlice, Error, GpuBuferWriter, GpuBufferWriterImpl,
};

pub type ImageHandle = Handle<Arc<Image>>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type ProgramHandle = Pool<Arc<Program>>;

type ImagePool = Pool<Arc<Image>>;
type BufferPool = HotColdPool<vk::Buffer, Arc<Buffer>, SentinelPoolStrategy<vk::Buffer>>;
type ProgramPool = Pool<Arc<Program>>;

pub struct ResourceManager {
    device: Arc<Device>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    bindless_pool: vk::DescriptorPool,
    bindless_layout: vk::DescriptorSetLayout,
    bindless_set: vk::DescriptorSet,
    sampled_image_updates: Mutex<Vec<(u32, vk::DescriptorImageInfo)>>,
    storage_image_updates: Mutex<Vec<(u32, vk::DescriptorImageInfo)>>,
    storage_buffer_updates: Mutex<Vec<(u32, vk::DescriptorBufferInfo)>>,
    staging: Mutex<Staging>,
    temp: TempBuffer,
    temp_buffer_handle: BufferHandle,
}

// For every possible item in Pool.
const MAX_RESOURCES: u32 = 262143;
const SAMPLED_IMAGE_BINDING: u32 = 0;
const STORAGE_IMAGE_BINDING: u32 = 1;
const STORAGE_BUFFER_BINDING: u32 = 2;
const TEMP_BUFFER_PAGE_SIZE: usize = 32 * 1024 * 1024;

impl ResourceManager {
    pub fn new(device: &Arc<Device>) -> dess_backend::Result<Self> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: MAX_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: MAX_RESOURCES,
            },
        ];
        let bindless_pool = unsafe {
            device.get().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .build(),
                None,
            )
        }?;
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLED_IMAGE_BINDING)
                .descriptor_count(MAX_RESOURCES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(STORAGE_IMAGE_BINDING)
                .descriptor_count(MAX_RESOURCES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(STORAGE_BUFFER_BINDING)
                .descriptor_count(MAX_RESOURCES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
        ];
        let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND; 3];
        let mut layout_binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags);
        let bindless_layout = unsafe {
            device.get().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut layout_binding_flags)
                    .build(),
                None,
            )
        }?;
        let mut alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(bindless_pool)
            .set_layouts(slice::from_ref(&bindless_layout));
        alloc_info.descriptor_set_count = 1;
        let bindless_set = unsafe { device.get().allocate_descriptor_sets(&alloc_info) }?[0];

        let temp = TempBuffer::new(device, TEMP_BUFFER_PAGE_SIZE)?;
        let mut buffers = BufferPool::default();
        let temp_buffer = temp.get();
        let temp_buffer_raw = temp_buffer.as_vk();
        let temp_buffer_handle = buffers.push(temp_buffer_raw, temp_buffer.clone());
        let storage_buffer_updates = vec![(
            temp_buffer_handle.index(),
            vk::DescriptorBufferInfo::builder()
                .buffer(temp_buffer_raw)
                .offset(0)
                .range(temp_buffer.desc().size as _)
                .build(),
        )];
        Ok(Self {
            device: device.clone(),
            images: RwLock::default(),
            buffers: RwLock::new(buffers),
            bindless_pool,
            bindless_layout,
            bindless_set,
            sampled_image_updates: Mutex::default(),
            storage_image_updates: Mutex::default(),
            storage_buffer_updates: Mutex::new(storage_buffer_updates),
            staging: Mutex::new(Staging::new(device, StagingDesc::default())?),
            temp,
            temp_buffer_handle,
        })
    }

    pub fn create_image(
        &self,
        desc: ImageCreateDesc,
        view: ImageViewDesc,
        layout: vk::ImageLayout,
        data: Option<&[ImageSubresourceData]>,
    ) -> dess_backend::Result<ImageHandle> {
        let image = Arc::new(Image::new(&self.device, desc)?);
        if let Some(data) = data {
            self.staging.lock().upload_image(&image, data)?;
        }
        self.add_image(&image, view, layout)
    }

    pub fn add_image(
        &self,
        image: &Arc<Image>,
        view: ImageViewDesc,
        layout: vk::ImageLayout,
    ) -> dess_backend::Result<ImageHandle> {
        let mut images = self.images.write();
        let view = image.view(view)?;
        let handle = images.push(image.clone());
        if image.desc().usage.contains(vk::ImageUsageFlags::SAMPLED) {
            self.sampled_image_updates.lock().push((
                handle.index(),
                vk::DescriptorImageInfo::builder()
                    .image_view(view)
                    .image_layout(layout)
                    .build(),
            ))
        }
        if image.desc().usage.contains(vk::ImageUsageFlags::STORAGE) {
            self.storage_image_updates.lock().push((
                handle.index(),
                vk::DescriptorImageInfo::builder()
                    .image_view(view)
                    .image_layout(layout)
                    .build(),
            ))
        }

        Ok(handle)
    }

    pub fn remove_image(&self, handle: ImageHandle) {
        self.images.write().remove(handle);
    }

    pub fn update_image(
        &self,
        handle: ImageHandle,
        image: &Arc<Image>,
        view: ImageViewDesc,
        layout: vk::ImageLayout,
    ) -> dess_backend::Result<()> {
        let mut images = self.images.write();
        let view = image.view(view)?;
        images.replace(handle, image.clone());
        if image.desc().usage.contains(vk::ImageUsageFlags::SAMPLED) {
            self.sampled_image_updates.lock().push((
                handle.index() as _,
                vk::DescriptorImageInfo::builder()
                    .image_view(view)
                    .image_layout(layout)
                    .build(),
            ))
        }
        if image.desc().usage.contains(vk::ImageUsageFlags::STORAGE) {
            self.storage_image_updates.lock().push((
                handle.index() as _,
                vk::DescriptorImageInfo::builder()
                    .image_view(view)
                    .image_layout(layout)
                    .build(),
            ))
        }

        Ok(())
    }

    pub fn upload_image(
        &self,
        handle: ImageHandle,
        data: &[ImageSubresourceData],
    ) -> dess_backend::Result<()> {
        if let Some(image) = self.images.read().get(handle).cloned() {
            self.staging.lock().upload_image(&image, data)?;
        }

        Ok(())
    }

    pub fn image(&self, handle: ImageHandle) -> Option<Arc<Image>> {
        self.images.read().get(handle).cloned()
    }

    pub fn create_buffer(&self, desc: BufferCreateDesc) -> dess_backend::Result<BufferHandle> {
        let buffer = Arc::new(Buffer::new(&self.device, desc)?);
        Ok(self.add_buffer(buffer))
    }

    pub fn create_buffer_from<T: Sized + Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: Option<&str>,
    ) -> dess_backend::Result<BufferHandle> {
        let mut desc = BufferCreateDesc::gpu(mem::size_of_val(data))
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
        if let Some(name) = name {
            desc = desc.name(name);
        }
        let handle = self.create_buffer(desc)?;
        self.upload_buffer(handle, 0, data)?;

        Ok(handle)
    }

    pub fn add_buffer(&self, buffer: Arc<Buffer>) -> BufferHandle {
        let size = buffer.desc().size;
        let usage = buffer.desc().usage;
        let raw = buffer.as_vk();
        let handle = self.buffers.write().push(buffer.as_vk(), buffer);
        if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            self.storage_buffer_updates.lock().push((
                handle.index(),
                vk::DescriptorBufferInfo::builder()
                    .buffer(raw)
                    .offset(0)
                    .range(size as _)
                    .build(),
            ));
        }
        handle
    }

    pub fn update_buffer(&self, handle: BufferHandle, buffer: Arc<Buffer>) {
        let mut buffers = self.buffers.write();
        buffers.replace(handle, buffer.as_vk());
        if buffer
            .desc()
            .usage
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER)
        {
            self.storage_buffer_updates.lock().push((
                handle.index(),
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.as_vk())
                    .offset(0)
                    .range(buffer.desc().size as _)
                    .build(),
            ));
        }
        buffers.replace_cold(handle, buffer);
    }

    pub fn update_buffer_from<T: Sized + Copy>(
        &self,
        handle: BufferHandle,
        data: &[T],
    ) -> dess_backend::Result<()> {
        let size = mem::size_of_val(data);
        let mut buffers = self.buffers.write();
        if let Some(buffer) = buffers.get_cold(handle) {
            let usage = buffer.desc().usage | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = Arc::new(Buffer::new(
                &self.device,
                BufferCreateDesc::gpu(size).usage(usage),
            )?);
            self.staging.lock().upload_buffer(&buffer, 0, data)?;
            buffers.replace(handle, buffer.as_vk());
            if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                self.storage_buffer_updates.lock().push((
                    handle.index(),
                    vk::DescriptorBufferInfo::builder()
                        .buffer(buffer.as_vk())
                        .offset(0)
                        .range(size as _)
                        .build(),
                ));
            }
            buffers.replace_cold(handle, buffer);
        }
        Ok(())
    }

    pub fn remove_buffer(&self, handle: BufferHandle) {
        self.buffers.write().remove(handle);
    }

    pub fn upload_buffer<T: Sized + Copy>(
        &self,
        handle: BufferHandle,
        offset: usize,
        data: &[T],
    ) -> dess_backend::Result<()> {
        let size = mem::size_of_val(data);
        if let Some(buffer) = self.buffers.read().get_cold(handle).cloned() {
            assert!(size + offset <= buffer.desc().size);
            self.staging.lock().upload_buffer(&buffer, offset, data)?;
        }
        Ok(())
    }

    pub fn buffer(&self, handle: BufferHandle) -> Option<Arc<Buffer>> {
        self.buffers.read().get_cold(handle).cloned()
    }

    pub fn bindless_set(&self) -> vk::DescriptorSet {
        self.bindless_set
    }

    pub fn tick(&self) {
        self.update_bindless_descriptors();
        self.temp.next_frame();
    }

    fn update_bindless_descriptors(&self) {
        puffin::profile_function!();
        let mut sampled_image_updates = self.sampled_image_updates.lock();
        let mut storage_image_updates = self.storage_image_updates.lock();
        let mut storage_buffer_updates = self.storage_buffer_updates.lock();
        let descritptor_updates = sampled_image_updates
            .iter()
            .map(|(slot, update)| {
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_array_element(*slot)
                    .dst_binding(SAMPLED_IMAGE_BINDING)
                    .image_info(slice::from_ref(update))
                    .build()
            })
            .chain(storage_image_updates.iter().map(|(slot, update)| {
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(*slot)
                    .dst_binding(STORAGE_IMAGE_BINDING)
                    .image_info(slice::from_ref(update))
                    .build()
            }))
            .chain(storage_buffer_updates.iter().map(|(slot, update)| {
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(*slot)
                    .dst_binding(STORAGE_BUFFER_BINDING)
                    .buffer_info(slice::from_ref(update))
                    .build()
            }))
            .collect::<Vec<_>>();
        unsafe {
            self.device
                .get()
                .update_descriptor_sets(&descritptor_updates, &[])
        };
        sampled_image_updates.clear();
        storage_image_updates.clear();
        storage_buffer_updates.clear();
    }

    pub fn push_uniform<T: Sized + Copy>(&self, data: &T) -> Result<usize, Error> {
        self.temp.push_uniform(data)
    }

    pub fn push_buffer<T: Sized + Copy>(&self, data: &[T]) -> Result<BufferSlice, Error> {
        let offset = self.temp.push_bufer(data)?;
        Ok(BufferSlice(self.temp_buffer_handle, offset as _))
    }

    pub fn write_buffer<T: Sized + Copy>(&self, count: usize) -> Result<GpuBuferWriter<T>, Error> {
        let writer = self.temp.write_buffer(count)?;
        Ok(GpuBuferWriter {
            handle: self.temp_buffer_handle,
            writer,
        })
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .get()
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device
                .get()
                .destroy_descriptor_set_layout(self.bindless_layout, None);
        }
    }
}
