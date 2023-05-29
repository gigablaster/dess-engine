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

use ash::vk;
use dess_render::{RenderError, RenderSystem, RenderSystemDesc};
use dess_render_backend::{
    PipelineDesc, PipelineVertex, RenderPassAttachmentDesc, RenderPassLayout, ShaderDesc,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
struct Vertex {
    pub pos: glam::Vec3,
    pub color: glam::Vec4,
}

static DESC: [vk::VertexInputAttributeDescription; 2] = [
    vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        location: 1,
        binding: 0,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: 12,
    },
];

impl PipelineVertex for Vertex {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription] {
        &DESC
    }
}

fn main() {
    SimpleLogger::new().init().unwrap();
    dess_vfs::scan(".").unwrap();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(false)
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();
    window.set_title("Vulkan Triangle");
    let size = window.inner_size();
    let render = RenderSystem::new(
        window.raw_display_handle(),
        window.raw_window_handle(),
        RenderSystemDesc::new([size.width, size.height]).debug(true),
    )
    .unwrap();
    let vertex_shader = render
        .create_shader(
            ShaderDesc::vertex(dess_vfs::get("shaders/simple.vert.spv").unwrap().data()),
            Some("vertex"),
        )
        .unwrap();
    let fragment_shader = render
        .create_shader(
            ShaderDesc::fragment(dess_vfs::get("shaders/simple.frag.spv").unwrap().data()),
            Some("fragment"),
        )
        .unwrap();
    let backbuffer = RenderPassAttachmentDesc::new(render.back_buffer_format());
    let render_pass_layout = RenderPassLayout {
        color_attachments: &[backbuffer],
        depth_attachment: None,
    };
    let render_pass = render.create_render_pass(render_pass_layout).unwrap();
    let pipeline_desc = PipelineDesc::new(&render_pass)
        .add_shader(&vertex_shader)
        .add_shader(&fragment_shader)
        .face_cull(false)
        .subpass(0);
    let pipeline = render.create_pipeline::<Vertex>(pipeline_desc).unwrap();
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            }
            Event::RedrawRequested(_) => {
                let size = window.inner_size();
                match render.render_frame([size.width, size.height], |context| {}) {
                    Err(RenderError::RecreateBuffers) => {
                        render.clear_fbos(&render_pass);
                        Ok(())
                    }
                    Err(other) => Err(other),
                    Ok(_) => Ok(()),
                }
                .unwrap();
            }
            _ => {}
        }
    });
}
