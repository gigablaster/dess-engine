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

use std::{marker::PhantomData, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk::{self};
use parking_lot::Mutex;

use crate::{
    BackendError, BackendResult, ClearRenderTarget, DrawStream, FboKey, ImageView, RenderArea,
    RenderPass, RenderPassHandle, RenderPassStorage, UpdateBindGroupsContext, MAX_ATTACHMENTS,
    MAX_COLOR_ATTACHMENTS,
};

use super::{
    frame::Frame, BindGroupStorage, BufferHandle, BufferSlice, BufferStorage, Device, ImageStorage,
    RasterPipelineStorage,
};

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub target: ImageView,
    pub clear: Option<ClearRenderTarget>,
}

impl RenderTarget {
    pub fn new(target: ImageView) -> Self {
        Self {
            target,
            clear: None,
        }
    }

    pub fn clear(mut self, color: ClearRenderTarget) -> Self {
        self.clear = Some(color);

        self
    }
}

pub(crate) struct RasterizerPass<'a> {
    render_pass: RenderPassHandle,
    color_targets: ArrayVec<RenderTarget, MAX_COLOR_ATTACHMENTS>,
    depth_target: Option<RenderTarget>,
    streams: Vec<DrawStream>,
    dims: [u32; 2],
    _marker: PhantomData<&'a ()>,
}

impl<'a> RasterizerPass<'a> {
    fn submit_one_stream(
        &self,
        context: &'a ExecutionContext<'a>,
        render_pass: vk::RenderPass,
        fbo: vk::Framebuffer,
        stream: &DrawStream,
    ) -> BackendResult<vk::CommandBuffer> {
        let cb = context.frame.get_or_create_secondary_buffer(
            &context.device.raw,
            context.device.queue_familt_index,
        )?;
        unsafe {
            context.device.raw.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(
                        vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                            | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                    )
                    .inheritance_info(
                        &vk::CommandBufferInheritanceInfo::builder()
                            .render_pass(render_pass)
                            .framebuffer(fbo)
                            .subpass(stream.get_subpass() as _)
                            .build(),
                    )
                    .build(),
            )
        }
        .unwrap();
        stream.execute(context, cb)?;
        unsafe { context.device.raw.end_command_buffer(cb).unwrap() };
        Ok(cb)
    }

    fn execute(&self, context: &ExecutionContext) -> BackendResult<()> {
        puffin::profile_function!();
        let attachments = self
            .color_targets
            .iter()
            .chain(self.depth_target.iter())
            .map(|x| x.target)
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        let clear = self
            .color_targets
            .iter()
            .chain(self.depth_target.iter())
            .map(|x| {
                x.clear
                    .unwrap_or(ClearRenderTarget::Color([0.0, 0.0, 0.0, 0.0]))
                    .into()
            })
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let render_pass = context
            .render_passes
            .get(self.render_pass.index())
            .ok_or(BackendError::InvalidHandle)?;
        let fbo = render_pass.get_or_create_fbo(
            &context.device.raw,
            FboKey {
                dims: self.dims,
                attachments,
            },
        )?;

        let begin_info = vk::RenderPassBeginInfo::builder()
            .clear_values(&clear)
            .framebuffer(fbo)
            .render_area(RenderArea::new(0, 0, self.dims[0], self.dims[1]).into())
            .render_pass(render_pass.raw)
            .build();

        unsafe {
            context.device.raw.cmd_begin_render_pass(
                context.frame.main_cb.raw,
                &begin_info,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            )
        };
        let cbs = Arc::new(Mutex::new(vec![
            vk::CommandBuffer::null();
            self.streams.len()
        ]));

        {
            puffin::profile_scope!("Generate comand buffers");
            rayon::scope(|s| {
                for (index, stream) in self.streams.iter().enumerate() {
                    let cbs = cbs.clone();
                    s.spawn(move |_| {
                        let cb = self.submit_one_stream(context, render_pass.raw, fbo, stream);
                        cbs.lock()[index] = cb.unwrap();
                    })
                }
            });
        };
        {
            puffin::profile_scope!("Execute command buffers");
            unsafe {
                context
                    .device
                    .raw
                    .cmd_execute_commands(context.frame.main_cb.raw, &cbs.lock())
            };
        }

        unsafe {
            context
                .device
                .raw
                .cmd_end_render_pass(context.frame.main_cb.raw)
        };

        Ok(())
    }
}

pub struct FrameContext<'device, 'frame> {
    pub(crate) device: &'device Device,
    pub(crate) dims: [u32; 2],
    pub(crate) target_view: ImageView,
    pub(crate) frame: &'frame Frame,
    pub(crate) temp_buffer_handle: BufferHandle,
    pub(crate) passes: Mutex<Vec<RasterizerPass<'frame>>>,
}

impl<'device, 'frame> FrameContext<'device, 'frame> {
    /// Allocate temporary memory on GPU and copy data there.
    ///
    /// Buffer slive is only valid during current frame, no need to free it in any way
    pub fn get_temp_buffer<T: Sized>(&self, data: &[T]) -> BackendResult<BufferSlice> {
        let offset = self.frame.temp_allocate(data)?;
        Ok(BufferSlice::new(self.temp_buffer_handle, offset))
    }

    pub fn get_temp_buffer_offset<T: Sized>(&self, data: &[T]) -> BackendResult<u32> {
        Ok(self.get_temp_buffer(data)?.offset)
    }

    pub fn with_bind_groups<F>(&self, cb: F) -> BackendResult<()>
    where
        F: FnOnce(&mut UpdateBindGroupsContext) -> BackendResult<()>,
    {
        self.device.with_bind_groups(cb)
    }

    /// Record render pass
    ///
    /// Render pass is a set of draw streams and barriers that executed before
    /// drawing start.
    ///
    /// Actual recording is done later, might be multithreaded.
    pub fn execute(
        &self,
        pass: RenderPassHandle,
        color_targets: &[RenderTarget],
        depth_target: Option<RenderTarget>,
        dims: [u32; 2],
        streams: impl Iterator<Item = DrawStream>,
    ) {
        let pass = RasterizerPass {
            render_pass: pass,
            dims,
            color_targets: color_targets
                .iter()
                .copied()
                .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>(),
            depth_target,
            streams: streams.collect(),
            _marker: PhantomData,
        };
        self.passes.lock().push(pass);
    }

    pub fn get_backbuffer(&self) -> RenderTarget {
        RenderTarget {
            target: self.target_view,
            clear: None,
        }
    }

    pub fn get_backbuffer_size(&self) -> [u32; 2] {
        self.dims
    }

    pub fn get_backbuffer_aspect_ratio(&self) -> f32 {
        (self.dims[0] as f32) / (self.dims[1] as f32)
    }
}

pub(crate) struct ExecutionContext<'a> {
    pub device: &'a Device,
    pub frame: &'a Frame,
    pub buffers: &'a BufferStorage,
    pub pipelines: &'a RasterPipelineStorage,
    pub descriptors: &'a BindGroupStorage,
    pub render_passes: &'a RenderPassStorage,
}

impl<'a> ExecutionContext<'a> {
    pub fn execute_rasterizing_passes<I>(&self, passes: I) -> BackendResult<()>
    where
        I: Iterator<Item = &'a RasterizerPass<'a>>,
    {
        puffin::profile_function!();
        for pass in passes {
            pass.execute(self)?;
        }

        Ok(())
    }
}
