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

use core::slice;
use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::Arc,
};

use ash::vk;
use bevy_tasks::AsyncComputeTaskPool;
use bytes::Bytes;
use dess_backend::{
    compile_raster_pipeline, AsVulkan, Buffer, CommandBufferRecorder, DescriptorSetLayout, Device,
    Image, ImageCreateDesc, ImageSubresourceData, ImageViewDesc, Program, RasterPipelineCreateDesc,
    RenderPass, ShaderDesc,
};
use dess_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy, TempList};
use gpu_descriptor_ash::AshDescriptorDevice;
use parking_lot::{Mutex, RwLock};

use crate::{
    staging::{Staging, StagingDesc},
    temp::TempBuffer,
    uniforms::UniformPool,
    BufferSlice, Error, GpuBuferWriter, GpuDescriptorAllocator, GpuDescriptorSet,
};

pub type ImageHandle = Handle<Arc<Image>>;
pub type BufferHandle = Handle<vk::Buffer>;
pub type DescriptorHandle = Handle<vk::DescriptorSet>;

#[derive(Debug, Clone, Hash, Copy, PartialEq, Eq)]
pub struct ProgramHandle(u32);

#[derive(Debug, Clone, Hash, Copy, PartialEq, Eq)]
pub struct RenderPassHandle(u32);

#[derive(Debug, Clone, Hash, Copy, PartialEq, Eq)]
pub struct RasterPipelineHandle(u32);

type ImagePool = Pool<Arc<Image>>;
type BufferPool = HotColdPool<vk::Buffer, Arc<Buffer>, SentinelPoolStrategy<vk::Buffer>>;
type ProgramPool = Vec<Arc<Program>>;
type RenderPassPool = Vec<Arc<RenderPass>>;
type DescriptorSetPool = HotColdPool<vk::DescriptorSet, DescriptorSetData>;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RasterPipelineDesc {
    program: ProgramHandle,
    render_pass: RenderPassHandle,
    subpass: usize,
    desc: RasterPipelineCreateDesc,
}

#[derive(Debug, Clone, Copy)]
struct BindingPoint<T> {
    pub binding: u32,
    pub data: Option<T>,
}

impl<T> BindingPoint<T> {
    pub fn new(binding: u32) -> Self {
        Self {
            binding,
            data: None,
        }
    }
}

#[derive(Debug)]
struct DescriptorSetData {
    descriptor: GpuDescriptorSet,
    uniform_buffers: Vec<BindingPoint<(vk::Buffer, u32, u32)>>,
    dynamic_uniform_bufffers: Vec<BindingPoint<(vk::Buffer, usize)>>,
    storage_buffers: Vec<BindingPoint<(vk::Buffer, u32, u32)>>,
    dynamic_storage_buffers: Vec<BindingPoint<(vk::Buffer, usize)>>,
    storage_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    sampled_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    layout: Arc<DescriptorSetLayout>,
}

impl DescriptorSetData {
    fn new(
        device: &Device,
        layout: &Arc<DescriptorSetLayout>,
        allocator: &mut GpuDescriptorAllocator,
    ) -> Result<Self, Error> {
        let mut uniform_buffers = Vec::with_capacity(layout.count.uniform_buffer as _);
        let mut dynamic_uniform_bufffers =
            Vec::with_capacity(layout.count.uniform_buffer_dynamic as _);
        let mut storage_buffers = Vec::with_capacity(layout.count.storage_buffer as _);
        let mut dynamic_storage_buffers =
            Vec::with_capacity(layout.count.storage_buffer_dynamic as _);
        let mut storage_images = Vec::with_capacity(layout.count.storage_image as _);
        let mut sampled_images = Vec::with_capacity(layout.count.sampled_image as _);
        layout.types.iter().for_each(|(binding, ty)| match *ty {
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                sampled_images.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::SAMPLED_IMAGE => {
                sampled_images.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::STORAGE_BUFFER => {
                storage_buffers.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                dynamic_storage_buffers.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::STORAGE_IMAGE => {
                storage_images.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::UNIFORM_BUFFER => {
                uniform_buffers.push(BindingPoint::new(*binding as u32));
            }
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                dynamic_uniform_bufffers.push(BindingPoint::new(*binding as u32));
            }
            _ => panic!("Descriptor type isn't supported: {:?}", ty),
        });
        let descriptor = unsafe {
            allocator.allocate(
                AshDescriptorDevice::wrap(device.get()),
                &layout.as_vk(),
                gpu_descriptor::DescriptorSetLayoutCreateFlags::empty(),
                &layout.count,
                1,
            )
        }?
        .remove(0);
        Ok(Self {
            descriptor,
            uniform_buffers,
            dynamic_uniform_bufffers,
            storage_buffers,
            dynamic_storage_buffers,
            storage_images,
            sampled_images,
            layout: layout.clone(),
        })
    }

    pub fn is_valid(&self) -> bool {
        self.uniform_buffers
            .iter()
            .all(|buffer| buffer.data.is_some())
            && self.sampled_images.iter().all(|image| image.data.is_some())
            && self
                .storage_buffers
                .iter()
                .all(|buffer| buffer.data.is_some())
            && self.storage_images.iter().all(|image| image.data.is_some())
            && self
                .dynamic_uniform_bufffers
                .iter()
                .all(|buffer| buffer.data.is_some())
    }

    fn bind_to_slot<T>(
        bindings: &mut [BindingPoint<T>],
        slot: usize,
        data: T,
    ) -> Result<Option<T>, Error> {
        let slot = bindings
            .iter()
            .enumerate()
            .find_map(|(index, point)| (point.binding == slot as u32).then_some(index))
            .ok_or(Error::BindingNotFound(slot))?;
        Ok(bindings[slot].data.replace(data))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DescriptorSetBinding {
    TempUniform(usize),
    TempStorage(usize),
    Uniform(Bytes),
    Storage(BufferSlice, usize),
    SampledImage(ImageHandle, vk::ImageLayout),
    StorageImage(ImageHandle, vk::ImageLayout),
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorBindings {
    bindings: Vec<(usize, DescriptorSetBinding)>,
}

impl DescriptorBindings {
    pub fn temp_uniform(mut self, slot: usize, size: usize) -> Self {
        self.bindings
            .push((slot, DescriptorSetBinding::TempUniform(size)));
        self
    }

    pub fn temp_storage(mut self, slot: usize, size: usize) -> Self {
        self.bindings
            .push((slot, DescriptorSetBinding::TempStorage(size)));
        self
    }

    pub fn uniform<T: Sized + Copy>(mut self, slot: usize, data: &T) -> Self {
        let data = Bytes::copy_from_slice(unsafe {
            slice::from_raw_parts(
                slice::from_ref(data).as_ptr() as *const u8,
                mem::size_of::<T>(),
            )
        });
        self.bindings
            .push((slot, DescriptorSetBinding::Uniform(data)));
        self
    }

    pub fn storage(mut self, slot: usize, buffer: BufferSlice, size: usize) -> Self {
        self.bindings
            .push((slot, DescriptorSetBinding::Storage(buffer, size)));
        self
    }

    pub fn sampled_image(
        mut self,
        slot: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Self {
        self.bindings
            .push((slot, DescriptorSetBinding::SampledImage(image, layout)));
        self
    }

    pub fn storage_image(
        mut self,
        slot: usize,
        image: ImageHandle,
        layout: vk::ImageLayout,
    ) -> Self {
        self.bindings
            .push((slot, DescriptorSetBinding::StorageImage(image, layout)));
        self
    }
}

pub struct ResourceManager {
    device: Arc<Device>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    programs: RwLock<ProgramPool>,
    pipelines: RwLock<Vec<(vk::Pipeline, vk::PipelineLayout)>>,
    descriptors: RwLock<DescriptorSetPool>,
    pipeline_descriptons: Mutex<HashMap<RasterPipelineDesc, RasterPipelineHandle>>,
    render_passes: RwLock<RenderPassPool>,
    temp: TempBuffer,
    temp_buffer_handle: BufferHandle,
    staging: Mutex<Staging>,
    uniforms: Mutex<UniformPool>,
    descriptors_to_recreate: Mutex<HashSet<DescriptorHandle>>,
    descriptors_to_update: Mutex<HashSet<DescriptorHandle>>,
    descriptor_allocator: Mutex<GpuDescriptorAllocator>,
    retired_uniforms: Mutex<Vec<u32>>,
    retired_descriptors: Mutex<Vec<GpuDescriptorSet>>,
    uniforms_to_free: Mutex<Vec<u32>>,
    descriptors_to_free: Mutex<Vec<GpuDescriptorSet>>,
}

const TEMP_BUFFER_PAGE_SIZE: usize = 32 * 1024 * 1024;

unsafe impl Send for ResourceManager {}
unsafe impl Sync for ResourceManager {}

impl ResourceManager {
    pub fn new(device: &Arc<Device>) -> Result<Self, Error> {
        let temp = TempBuffer::new(device, TEMP_BUFFER_PAGE_SIZE)?;
        let mut buffers = BufferPool::default();
        let temp_buffer = temp.get();
        let temp_buffer_raw = temp_buffer.as_vk();
        let temp_buffer_handle = buffers.push(temp_buffer_raw, temp_buffer.clone());
        Ok(Self {
            device: device.clone(),
            images: RwLock::default(),
            buffers: RwLock::new(buffers),
            programs: RwLock::default(),
            render_passes: RwLock::default(),
            descriptors: RwLock::default(),
            pipelines: RwLock::default(),
            pipeline_descriptons: Mutex::default(),
            temp,
            temp_buffer_handle,
            staging: Mutex::new(Staging::new(device, StagingDesc::default())?),
            retired_uniforms: Mutex::default(),
            uniforms: Mutex::new(UniformPool::new(device)?),
            descriptors_to_recreate: Mutex::default(),
            descriptors_to_update: Mutex::default(),
            descriptor_allocator: Mutex::new(GpuDescriptorAllocator::new(0)),
            retired_descriptors: Mutex::default(),
            uniforms_to_free: Mutex::default(),
            descriptors_to_free: Mutex::default(),
        })
    }

    pub fn add_image(&self, image: Arc<Image>) -> Result<ImageHandle, Error> {
        let mut images = self.images.write();
        let handle = images.push(image.clone());

        Ok(handle)
    }

    pub fn create_image(&self, desc: ImageCreateDesc) -> Result<ImageHandle, Error> {
        let image = Image::new(&self.device, desc)?.into();
        self.add_image(image)
    }

    pub fn remove_image(&self, handle: ImageHandle) {
        self.images.write().remove(handle);
    }

    pub fn update_image(&self, handle: ImageHandle, image: Arc<Image>) -> dess_backend::Result<()> {
        let mut images = self.images.write();
        images.replace(handle, image.clone());

        Ok(())
    }

    pub fn upload_image(
        &self,
        handle: ImageHandle,
        data: &[ImageSubresourceData],
    ) -> Result<(), Error> {
        if let Some(image) = self.images.read().get(handle) {
            self.staging.lock().upload_image(image, data)?;
        }
        Ok(())
    }
    pub fn image(&self, handle: ImageHandle) -> Option<Arc<Image>> {
        self.images.read().get(handle).cloned()
    }

    pub fn add_buffer(&self, buffer: Arc<Buffer>) -> BufferHandle {
        let handle = self.buffers.write().push(buffer.as_vk(), buffer);

        handle
    }

    pub fn update_buffer(&self, handle: BufferHandle, buffer: Arc<Buffer>) {
        let mut buffers = self.buffers.write();
        buffers.replace(handle, buffer.as_vk());
        buffers.replace_cold(handle, buffer);
    }

    pub fn remove_buffer(&self, handle: BufferHandle) {
        self.buffers.write().remove(handle);
    }

    pub fn upload_buffer<T: Sized + Copy>(
        &self,
        buffer: BufferSlice,
        data: &[T],
    ) -> Result<(), Error> {
        if let Some(target) = self.buffers.read().get_cold(buffer.handle()) {
            self.staging
                .lock()
                .upload_buffer(target, buffer.offset() as _, data)?;
        }
        Ok(())
    }

    pub fn buffer(&self, handle: BufferHandle) -> Option<Arc<Buffer>> {
        self.buffers.read().get_cold(handle).cloned()
    }

    /// Update bindings, submits staging buffer uploads. Returns sempahore to wait
    /// before starting actual render.
    pub fn before_frame_record(&self) -> Result<(vk::Semaphore, vk::PipelineStageFlags), Error> {
        self.temp.next_frame();
        let mut uniforms_to_free = self.uniforms_to_free.lock();
        let mut descriptors_to_free = self.descriptors_to_free.lock();
        let mut uniforms = self.uniforms.lock();
        let mut descriptor_allocator = self.descriptor_allocator.lock();
        uniforms_to_free
            .drain(..)
            .for_each(|x| uniforms.dealloc(x as _));
        unsafe {
            descriptor_allocator.free(
                AshDescriptorDevice::wrap(self.device.get()),
                descriptors_to_free.drain(..),
            )
        };
        descriptors_to_free.append(&mut self.retired_descriptors.lock());
        uniforms_to_free.append(&mut self.retired_uniforms.lock());
        self.update_descriptor_sets()?;

        self.staging.lock().upload()
    }

    /// Finalizes resource uploading.
    pub fn before_start_renering(&self, recorder: &CommandBufferRecorder) {
        self.staging.lock().execute_pending_barriers(recorder);
    }

    pub fn push_temp_uniform<T: Sized + Copy>(&self, data: &T) -> Result<usize, Error> {
        self.temp.push_uniform(data)
    }

    pub fn push_temp_buffer<T: Sized + Copy>(&self, data: &[T]) -> Result<BufferSlice, Error> {
        let offset = self.temp.push_bufer(data)?;
        Ok(BufferSlice(self.temp_buffer_handle, offset as _))
    }

    pub fn write_temp_buffer<T: Sized + Copy>(
        &self,
        count: usize,
    ) -> Result<GpuBuferWriter<T>, Error> {
        let writer = self.temp.write_buffer(count)?;
        Ok(GpuBuferWriter {
            handle: self.temp_buffer_handle,
            writer,
        })
    }

    pub fn add_program(&self, program: Arc<Program>) -> ProgramHandle {
        let mut programs = self.programs.write();
        let index = programs.len() as u32;
        programs.push(program);
        ProgramHandle(index)
    }

    pub fn create_program(&self, shaders: &[ShaderDesc]) -> Result<ProgramHandle, Error> {
        let program = Program::new(&self.device, shaders)?.into();
        Ok(self.add_program(program))
    }

    pub fn program(&self, handle: ProgramHandle) -> Option<Arc<Program>> {
        self.programs.read().get(handle.0 as usize).cloned()
    }

    pub fn update_program(&self, handle: ProgramHandle, program: Arc<Program>) {
        self.programs.write()[handle.0 as usize] = program;
        let descs = self.pipeline_descriptons.lock();
        let mut pipelines = self.pipelines.write();
        descs
            .iter()
            .filter_map(|(desc, index)| (desc.program == handle).then_some(*index))
            .for_each(|handle| {
                pipelines[handle.0 as usize] = (vk::Pipeline::null(), vk::PipelineLayout::null())
            });
    }

    pub fn add_render_pass(&self, render_pass: Arc<RenderPass>) -> RenderPassHandle {
        let mut render_passes = self.render_passes.write();
        let index = render_passes.len() as u32;
        render_passes.push(render_pass);
        RenderPassHandle(index)
    }

    pub fn create_pipeline(
        &self,
        program: ProgramHandle,
        render_pass: RenderPassHandle,
        subpass: usize,
        desc: RasterPipelineCreateDesc,
    ) -> RasterPipelineHandle {
        let desc = RasterPipelineDesc {
            program,
            render_pass,
            subpass,
            desc,
        };
        let mut descs = self.pipeline_descriptons.lock();
        if let Some(handle) = descs.get(&desc) {
            *handle
        } else {
            let mut pipelines = self.pipelines.write();
            let handle = RasterPipelineHandle(pipelines.len() as u32);
            pipelines.push((vk::Pipeline::null(), vk::PipelineLayout::null()));
            descs.insert(desc, handle);
            handle
        }
    }

    pub async fn compile_pipelines(&self) -> Result<(), Error> {
        let descs = self.pipeline_descriptons.lock();
        let compiled_pipelines = AsyncComputeTaskPool::get().scope(|s| {
            let pipelines = self.pipelines.read();
            descs
                .iter()
                .filter(|(_, handle)| pipelines[handle.0 as usize].0 == vk::Pipeline::null())
                .for_each(|(desc, handle)| s.spawn(self.compile_single_pipeline(*handle, desc)));
        });
        let mut pipelines = self.pipelines.write();
        for it in compiled_pipelines {
            let (handle, data) = it?;
            pipelines[handle.0 as usize] = data;
        }
        Ok(())
    }

    async fn compile_single_pipeline(
        &self,
        handle: RasterPipelineHandle,
        desc: &RasterPipelineDesc,
    ) -> Result<(RasterPipelineHandle, (vk::Pipeline, vk::PipelineLayout)), Error> {
        let program = self
            .programs
            .read()
            .get(handle.0 as usize)
            .ok_or(Error::InvalidHandle)?
            .clone();
        let render_pass = self
            .render_passes
            .read()
            .get(desc.render_pass.0 as usize)
            .ok_or(Error::InvalidHandle)?
            .clone();
        Ok((
            handle,
            compile_raster_pipeline(
                &self.device,
                &program,
                &render_pass,
                desc.subpass,
                &desc.desc,
            )?,
        ))
    }

    pub fn create_descriptor_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        bindings: DescriptorBindings,
    ) -> Result<DescriptorHandle, Error> {
        let mut descriptor =
            DescriptorSetData::new(&self.device, layout, &mut self.descriptor_allocator.lock())?;
        self.update_bindings_impl(
            &mut descriptor,
            bindings,
            &self.buffers.read(),
            &self.images.read(),
            &mut self.uniforms.lock(),
            &mut self.retired_uniforms.lock(),
        )?;
        let handle = self
            .descriptors
            .write()
            .push(*descriptor.descriptor.raw(), descriptor);
        self.descriptors_to_update.lock().insert(handle);
        Ok(handle)
    }

    pub fn update_descriptor_set(
        &self,
        handle: DescriptorHandle,
        bindings: DescriptorBindings,
    ) -> Result<(), Error> {
        let mut descrptors = self.descriptors.write();
        if let Some(descriptor) = descrptors.get_cold_mut(handle) {
            self.update_bindings_impl(
                descriptor,
                bindings,
                &self.buffers.read(),
                &self.images.read(),
                &mut self.uniforms.lock(),
                &mut self.retired_uniforms.lock(),
            )?;
            self.descriptors_to_recreate.lock().insert(handle);
        }
        todo!()
    }

    pub fn remove_descriptor_set(&self, handle: DescriptorHandle) {
        if let Some((_, mut descriptor)) = self.descriptors.write().remove(handle) {
            let mut to_retire = self.retired_uniforms.lock();
            descriptor
                .uniform_buffers
                .drain(..)
                .for_each(|x| x.data.iter().for_each(|x| to_retire.push(x.1)));
        }
    }

    fn update_bindings_impl(
        &self,
        descriptor: &mut DescriptorSetData,
        bindings: DescriptorBindings,
        buffers: &BufferPool,
        images: &ImagePool,
        uniforms: &mut UniformPool,
        to_retire: &mut Vec<u32>,
    ) -> Result<(), Error> {
        for (slot, binding) in bindings.bindings {
            match binding {
                DescriptorSetBinding::TempUniform(size) => {
                    DescriptorSetData::bind_to_slot(
                        &mut descriptor.dynamic_uniform_bufffers,
                        slot,
                        (self.temp.get().as_vk(), size),
                    )?;
                }
                DescriptorSetBinding::TempStorage(size) => {
                    DescriptorSetData::bind_to_slot(
                        &mut descriptor.dynamic_storage_buffers,
                        slot,
                        (self.temp.get().as_vk(), size),
                    )?;
                }
                DescriptorSetBinding::Uniform(data) => {
                    let offset = uniforms.push_bytes(&data)?;
                    let old_uniform = DescriptorSetData::bind_to_slot(
                        &mut descriptor.uniform_buffers,
                        slot,
                        (uniforms.get().as_vk(), offset as u32, data.len() as u32),
                    )?;
                    if let Some((_, offset, _)) = old_uniform {
                        to_retire.push(offset);
                    }
                }
                DescriptorSetBinding::Storage(buffer, size) => {
                    let offset = buffer.offset();
                    let buffer = buffers
                        .get(buffer.handle())
                        .copied()
                        .ok_or(Error::InvalidHandle)?;
                    DescriptorSetData::bind_to_slot(
                        &mut descriptor.storage_buffers,
                        slot,
                        (buffer, offset, size as u32),
                    )?;
                }
                DescriptorSetBinding::SampledImage(handle, layout) => {
                    let image = images.get(handle).ok_or(Error::InvalidHandle)?;
                    let view = image.view(ImageViewDesc::color())?;
                    DescriptorSetData::bind_to_slot(
                        &mut descriptor.sampled_images,
                        slot,
                        (handle, view, layout),
                    )?;
                }
                DescriptorSetBinding::StorageImage(handle, layout) => {
                    let image = images.get(handle).ok_or(Error::InvalidHandle)?;
                    let view = image.view(ImageViewDesc::color())?;
                    DescriptorSetData::bind_to_slot(
                        &mut descriptor.storage_images,
                        slot,
                        (handle, view, layout),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn update_descriptor_sets(&self) -> Result<(), Error> {
        puffin::profile_function!();
        let mut allocator = self.descriptor_allocator.lock();
        let mut to_update = self.descriptors_to_update.lock();
        let mut to_recreate = self.descriptors_to_recreate.lock();
        let mut update_list = HashSet::with_capacity(to_update.len() + to_recreate.len());
        let mut uniforms = self.uniforms.lock();
        let mut retired_descriptors = self.retired_descriptors.lock();
        to_update.drain().for_each(|x| {
            update_list.insert(x);
        });
        let mut descriptors = self.descriptors.write();
        for handle in to_recreate.drain() {
            if let Some(desc) = descriptors.get_cold_mut(handle) {
                if !desc.is_valid() {
                    return Err(Error::InvalidDescriptorSet);
                }
                let new_descriptor = unsafe {
                    allocator.allocate(
                        AshDescriptorDevice::wrap(self.device.get()),
                        &desc.layout.as_vk(),
                        gpu_descriptor::DescriptorSetLayoutCreateFlags::empty(),
                        &desc.layout.count,
                        1,
                    )
                }?
                .remove(0);
                retired_descriptors.push(mem::replace(&mut desc.descriptor, new_descriptor));
                update_list.insert(handle);
            }
        }
        let mut writes = Vec::with_capacity(update_list.len());
        let mut images = TempList::new();
        let mut buffers = TempList::new();
        for handle in update_list.drain() {
            Self::prepare_descriptor(
                handle,
                &mut uniforms,
                &mut descriptors,
                &mut writes,
                &mut images,
                &mut buffers,
            );
            let raw = descriptors
                .get_cold(handle)
                .map(|x| x.descriptor.raw())
                .copied();
            if let Some(raw) = raw {
                descriptors.replace(handle, raw);
            }
        }
        if !writes.is_empty() {
            unsafe { self.device.get().update_descriptor_sets(&writes, &[]) };
        }

        Ok(())
    }

    fn prepare_descriptor(
        handle: DescriptorHandle,
        uniforms: &mut UniformPool,
        storage: &mut DescriptorSetPool,
        writes: &mut Vec<vk::WriteDescriptorSet>,
        images: &mut TempList<vk::DescriptorImageInfo>,
        buffers: &mut TempList<vk::DescriptorBufferInfo>,
    ) {
        if let Some(desc) = storage.get_cold(handle) {
            desc.sampled_images
                .iter()
                .map(|binding| {
                    let image = binding.data.as_ref().unwrap();
                    let image = vk::DescriptorImageInfo::builder()
                        .image_view(image.1)
                        .image_layout(image.2)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .image_info(slice::from_ref(images.add(image)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .build()
                })
                .for_each(|x| writes.push(x));
            desc.storage_images
                .iter()
                .map(|binding| {
                    let image = binding.data.as_ref().unwrap();
                    let image = vk::DescriptorImageInfo::builder()
                        .image_view(image.1)
                        .image_layout(image.2)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .image_info(slice::from_ref(images.add(image)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .build()
                })
                .for_each(|x| writes.push(x));
            desc.uniform_buffers
                .iter()
                .map(|binding| {
                    let data = binding.data.unwrap();
                    let buffer = vk::DescriptorBufferInfo::builder()
                        .buffer(uniforms.get().as_vk())
                        .offset(data.1 as _)
                        .range(data.2 as _)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .buffer_info(slice::from_ref(buffers.add(buffer)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .build()
                })
                .for_each(|x| writes.push(x));
            desc.dynamic_uniform_bufffers
                .iter()
                .map(|binding| {
                    let data = &binding.data.unwrap();
                    let buffer = vk::DescriptorBufferInfo::builder()
                        .buffer(data.0)
                        .offset(0)
                        .range(data.1 as _)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .buffer_info(slice::from_ref(buffers.add(buffer)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .build()
                })
                .for_each(|x| writes.push(x));
            desc.storage_buffers
                .iter()
                .map(|binding| {
                    let data = &binding.data.unwrap();
                    let buffer = vk::DescriptorBufferInfo::builder()
                        .buffer(data.0.as_vk())
                        .offset(data.1 as u64)
                        .range(data.2 as u64)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .buffer_info(slice::from_ref(buffers.add(buffer)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .build()
                })
                .for_each(|x| writes.push(x));
            desc.dynamic_storage_buffers
                .iter()
                .map(|binding| {
                    let data = &binding.data.unwrap();
                    let buffer = vk::DescriptorBufferInfo::builder()
                        .buffer(data.0)
                        .offset(0)
                        .range(data.1 as _)
                        .build();

                    vk::WriteDescriptorSet::builder()
                        .buffer_info(slice::from_ref(buffers.add(buffer)))
                        .dst_binding(binding.binding)
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                        .build()
                })
                .for_each(|x| writes.push(x));
        }
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        let mut allocator = self.descriptor_allocator.lock();
        unsafe {
            self.device.get().device_wait_idle().unwrap();
            allocator.free(
                AshDescriptorDevice::wrap(self.device.get()),
                self.descriptors_to_free.lock().drain(..),
            );

            allocator.cleanup(AshDescriptorDevice::wrap(self.device.get()));
        };
    }
}
