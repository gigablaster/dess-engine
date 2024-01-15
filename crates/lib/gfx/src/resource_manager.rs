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
use dess_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy};
use gpu_descriptor::DescriptorTotalCount;
use parking_lot::{Mutex, RwLock};
use smol_str::SmolStr;

use crate::{
    staging::{Staging, StagingDesc},
    temp::TempBuffer,
    uniforms::UniformPool,
    BufferSlice, Error, GpuBuferWriter, GpuDescriptorSet,
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
type DescriptorSetPool = HotColdPool<vk::DescriptorSet, Box<DescriptorSetData>>;

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

#[derive(Debug)]
struct DescriptorSetData {
    descriptor: GpuDescriptorSet,
    uniform_buffers: Vec<BindingPoint<(vk::Buffer, u32, u32)>>,
    dynamic_uniform_bufffers: Vec<BindingPoint<(vk::Buffer, usize)>>,
    storage_buffers: Vec<BindingPoint<(vk::Buffer, u32, u32)>>,
    dynamic_storage_buffers: Vec<BindingPoint<(vk::Buffer, usize)>>,
    storage_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    sampled_images: Vec<BindingPoint<(ImageHandle, vk::ImageView, vk::ImageLayout)>>,
    layout: vk::DescriptorSetLayout,
    names: HashMap<SmolStr, usize>,
}

impl DescriptorSetData {
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
        bindings: &mut Vec<BindingPoint<T>>,
        slot: usize,
        data: T,
    ) -> Result<Option<T>, Error> {
        let slot = bindings
            .iter()
            .enumerate()
            .find_map(|(index, point)| (point.binding == slot as u32).then_some(index))
            .ok_or(Error::BindingNotFoun(slot as usize))?;
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
    pub bindings: Vec<(usize, DescriptorSetBinding)>,
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
    descritptors_to_delete: Mutex<Vec<GpuDescriptorSet>>,
    uniforms_to_free: Mutex<Vec<u32>>,
    uniforms: Mutex<UniformPool>,
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
            descritptors_to_delete: Mutex::default(),
            uniforms_to_free: Mutex::default(),
            uniforms: Mutex::new(UniformPool::new(device)?),
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

    pub fn create_descritpro_set(
        &self,
        set: &DescriptorSetLayout,
        bindings: DescriptorBindings,
    ) -> Result<DescriptorHandle, Error> {
        todo!()
    }

    fn update_bindings_impl(
        &self,
        descriptor: &mut DescriptorSetData,
        bindings: DescriptorBindings,
    ) -> Result<(), Error> {
        let buffers = self.buffers.read();
        let images = self.images.read();
        let mut uniforms = self.uniforms.lock();
        let mut to_retire = self.uniforms_to_free.lock();
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
}
