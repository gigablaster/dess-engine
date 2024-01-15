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

use std::{collections::HashMap, sync::Arc};

use ash::vk;
use bevy_tasks::AsyncComputeTaskPool;
use dess_backend::{
    compile_raster_pipeline, AsVulkan, Buffer, CommandBufferRecorder, Device, Image,
    ImageCreateDesc, ImageSubresourceData, Program, RasterPipelineCreateDesc, RenderPass,
    ShaderDesc,
};
use dess_common::{Handle, HotColdPool, Pool, SentinelPoolStrategy};
use parking_lot::{Mutex, RwLock};

use crate::{
    staging::{Staging, StagingDesc},
    temp::TempBuffer,
    BufferSlice, Error, GpuBuferWriter,
};

pub type ImageHandle = Handle<Arc<Image>>;
pub type BufferHandle = Handle<vk::Buffer>;

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

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct RasterPipelineDesc {
    program: ProgramHandle,
    render_pass: RenderPassHandle,
    subpass: usize,
    desc: RasterPipelineCreateDesc,
}

pub struct ResourceManager {
    device: Arc<Device>,
    images: RwLock<ImagePool>,
    buffers: RwLock<BufferPool>,
    programs: RwLock<ProgramPool>,
    pipelines: RwLock<Vec<(vk::Pipeline, vk::PipelineLayout)>>,
    pipeline_descriptons: Mutex<HashMap<RasterPipelineDesc, RasterPipelineHandle>>,
    render_passes: RwLock<RenderPassPool>,
    temp: TempBuffer,
    temp_buffer_handle: BufferHandle,
    staging: Mutex<Staging>,
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
            pipelines: RwLock::default(),
            pipeline_descriptons: Mutex::default(),
            temp,
            temp_buffer_handle,
            staging: Mutex::new(Staging::new(device, StagingDesc::default())?),
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
}
