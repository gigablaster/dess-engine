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

use std::collections::HashMap;

use ash::vk;
use dess_render::{
    DescriptorHandle, GpuType, RenderError, RenderOp, RenderSystem, RenderSystemDesc,
};
use dess_render_backend::{
    PipelineDesc, PipelineVertex, RenderPassAttachment, RenderPassAttachmentDesc, RenderPassLayout,
    ShaderDesc, SubImage,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use vk_sync::{AccessType, ImageBarrier};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C, packed)]
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
        RenderSystemDesc::new([size.width, size.height])
            .debug(true)
            .gpu_type(GpuType::PreferIntegrated),
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
    let backbuffer = RenderPassAttachmentDesc::new(render.back_buffer_format()).clear_input();
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
    let mut pipeline_cache = HashMap::new();
    pipeline_cache.insert(1u32, pipeline);
    let (vertex_buffer, index_buffer) = render
        .update_resources(|context| {
            let mut context = context;
            let vertex_buffer = context
                .create_buffer(&[
                    Vertex {
                        pos: glam::Vec3::new(0.0, -0.5, 0.0),
                        color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                    },
                    Vertex {
                        pos: glam::Vec3::new(0.5, 0.5, 0.0),
                        color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
                    },
                    Vertex {
                        pos: glam::Vec3::new(-0.5, 0.5, 0.0),
                        color: glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
                    },
                ])
                .unwrap();
            let index_buffer = context.create_buffer(&[0u16, 1u16, 2u16]).unwrap();

            (vertex_buffer, index_buffer)
        })
        .unwrap();
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            }
            Event::RedrawRequested(_) => {
                let inner_size = window.inner_size();
                match render.render_frame([inner_size.width, inner_size.height], |context| {
                    let rop = RenderOp {
                        pso: 1,
                        vertex_buffer,
                        index_buffer,
                        index_count: 3,
                        descs: [
                            DescriptorHandle::default(),
                            DescriptorHandle::default(),
                            DescriptorHandle::default(),
                            DescriptorHandle::default(),
                        ],
                    };
                    context
                        .record(|recorder| {
                            let color_attachment = RenderPassAttachment::new(
                                context.backbuffer,
                                vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: [0.2, 0.2, 0.25, 1.0],
                                    },
                                },
                            );
                            let image_barrier = ImageBarrier {
                                previous_accesses: &[AccessType::Nothing],
                                next_accesses: &[AccessType::ColorAttachmentWrite],
                                previous_layout: vk_sync::ImageLayout::Optimal,
                                next_layout: vk_sync::ImageLayout::Optimal,
                                discard_contents: true,
                                src_queue_family_index: context.graphics_queue,
                                dst_queue_family_index: context.graphics_queue,
                                image: context.backbuffer.raw,
                                range: context.backbuffer.subresource_range(
                                    SubImage::LayerAndMip(0, 0),
                                    vk::ImageAspectFlags::COLOR,
                                ),
                            };
                            recorder.barrier(None, &[], &[image_barrier]);
                            recorder.render_pass(&render_pass, &[color_attachment], None, |pass| {
                                pass.set_viewport(
                                    vk::Viewport::builder()
                                        .width(context.resolution[0] as _)
                                        .height(context.resolution[1] as _)
                                        .min_depth(0.0)
                                        .max_depth(1.0)
                                        .x(0.0)
                                        .y(0.0)
                                        .build(),
                                );
                                pass.set_scissor(vk::Rect2D {
                                    offset: vk::Offset2D { x: 0, y: 0 },
                                    extent: vk::Extent2D {
                                        width: context.resolution[0],
                                        height: context.resolution[1],
                                    },
                                });
                                context.render(&pipeline_cache, &pass, &[rop], Some("Triangle"))
                            })
                        })
                        .unwrap();
                }) {
                    Err(RenderError::RecreateBuffers) => {
                        render.clear_fbos(&render_pass);
                        Ok(())
                    }
                    Err(other) => Err(other),
                    Ok(_) => Ok(()),
                }
                .unwrap();
            }
            Event::RedrawEventsCleared => window.request_redraw(),
            _ => {}
        }
    });
}
