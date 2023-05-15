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

use core::slice;
use std::{thread::sleep, time::Duration};

use ash::vk;

use glam::Vec3;
use render::vulkan::{
    create_pipeline_cache, BackendError, Device, FreeGpuResource, Instance, PhysicalDeviceList,
    Pipeline, PipelineDesc, PipelineVertex, RenderPass, RenderPassAttachment,
    RenderPassAttachmentDesc, RenderPassLayout, Shader, SubImage, SubmitWaitDesc, Surface,
    Swapchain,
};
use sdl2::event::{Event, WindowEvent};
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier};

#[repr(C, packed)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
}

impl PipelineVertex for Vertex {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription] {
        static desc: [vk::VertexInputAttributeDescription; 2] = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
        ];

        &desc
    }
}

fn main() -> Result<(), String> {
    simple_logger::init().unwrap();
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let _timer = sdl.timer()?;

    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    eprintln!("Serving demo profile data on {server_addr}");

    vfs::scan(".").unwrap();

    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();

    puffin::set_scopes_on(true);
    let window = video
        .window("vk-triangle", 1280, 720)
        .position_centered()
        .resizable()
        .vulkan()
        .build()
        .map_err(|x| x.to_string())?;

    let mut event_pump = sdl.event_pump()?;

    let instance = Instance::builder()
        .debug(cfg!(debug_assertions))
        .build(&window)
        .unwrap();

    let surface = Surface::create(&instance, &window).unwrap();

    let pdevice = instance
        .enumerate_physical_devices()
        .unwrap()
        .find_suitable_device(
            &surface,
            &[
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ],
        )
        .unwrap();

    let device = Device::create(instance, pdevice).unwrap();

    let vertex_shader = Shader::vertex(
        &device.raw,
        vfs::get("shaders/simple.vert.spv").unwrap().data(),
    )
    .unwrap();

    let fragment_shader = Shader::fragment(
        &device.raw,
        vfs::get("shaders/simple.frag.spv").unwrap().data(),
    )
    .unwrap();

    let mut swapchain = Swapchain::new(&device, surface).unwrap();

    let color_attachment_desc =
        RenderPassAttachmentDesc::new(swapchain.backbuffer_format()).clear_input();
    let render_pass_desc = RenderPassLayout {
        color_attachments: &[color_attachment_desc],
        depth_attachment: None,
    };
    let render_pass = RenderPass::new(&device.raw, render_pass_desc).unwrap();
    let pipeline_cache = create_pipeline_cache(&device.raw).unwrap();
    let pipeline_desc = PipelineDesc::new(&render_pass)
        .add_shader(&fragment_shader)
        .add_shader(&vertex_shader)
        .face_cull(false);
    let pipeline = Pipeline::new::<Vertex>(&device.raw, &pipeline_cache, pipeline_desc).unwrap();

    let vertices = [
        Vertex {
            pos: Vec3::new(0.0, -0.5, 0.0),
            color: Vec3::new(1.0, 0.0, 0.0),
        },
        Vertex {
            pos: Vec3::new(-0.5, 0.5, 0.0),
            color: Vec3::new(0.0, 1.0, 0.0),
        },
        Vertex {
            pos: Vec3::new(0.5, 0.5, 0.0),
            color: Vec3::new(0.0, 0.0, 1.0),
        },
    ];
    let indices = [0u16, 1u16, 2u16];
    let _vertex_buffer = device.create_geometry_buffer_from(&vertices).unwrap();
    let _index_buffer = device.create_geometry_buffer_from(&indices).unwrap();

    /*let mut vertex_staging = Buffer::new(
        &device,
        BufferDesc::staging(3 * size_of::<Vertex>()),
        Some("Vertex staging"),
    )
    .unwrap();
    let mut index_staging = Buffer::new(
        &device,
        BufferDesc::staging(3 * size_of::<u16>()),
        Some("Index staging"),
    )
    .unwrap();
    let vertex_buffer = Buffer::new(
        &device,
        BufferDesc::vertex::<Vertex>(3),
        Some("Vertex buffer"),
    )
    .unwrap();
    let index_buffer = Buffer::new(&device, BufferDesc::index(3), Some("Index buffer")).unwrap();


    let mut map = vertex_staging.map().unwrap();
    map.push(&vertices);
    vertex_staging.unmap().unwrap();

    let mut map = index_staging.map().unwrap();
    map.push(&indices).unwrap();
    index_staging.unmap();

    device
        .with_setup_cb(|recorder| {
            let barriers = [
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferRead],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    buffer: vertex_staging.raw,
                    offset: 0,
                    size: vertex_staging.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferRead],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    buffer: index_staging.raw,
                    offset: 0,
                    size: index_staging.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::TransferWrite],
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    buffer: vertex_buffer.raw,
                    offset: 0,
                    size: vertex_buffer.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::TransferWrite],
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    buffer: index_buffer.raw,
                    offset: 0,
                    size: index_buffer.desc.size,
                },
            ];
            pipeline_barrier(&device.raw, *recorder.cb, None, &barriers, &[]);
            recorder.copy_buffers(&index_staging, &index_buffer);
            recorder.copy_buffers(&vertex_staging, &vertex_buffer);
            let barriers = [
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferRead],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    buffer: vertex_staging.raw,
                    offset: 0,
                    size: vertex_staging.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferRead],
                    next_accesses: &[AccessType::TransferRead],
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    buffer: index_staging.raw,
                    offset: 0,
                    size: index_staging.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::VertexBuffer],
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    buffer: vertex_buffer.raw,
                    offset: 0,
                    size: vertex_buffer.desc.size,
                },
                BufferBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::VertexBuffer],
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    buffer: index_buffer.raw,
                    offset: 0,
                    size: index_buffer.desc.size,
                },
            ];
            pipeline_barrier(&device.raw, *recorder.cb, None, &barriers, &[]);
        })
        .unwrap();
    */
    let mut skip_render = false;
    'running: loop {
        let mut recreate_swapchain = false;
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::Window {
                    win_event: WindowEvent::Resized(_w, _h),
                    ..
                } => recreate_swapchain = true,
                Event::Window {
                    win_event: WindowEvent::Minimized,
                    ..
                }
                | Event::Window {
                    win_event: WindowEvent::Hidden,
                    ..
                } => skip_render = true,
                Event::Window {
                    win_event: WindowEvent::Restored,
                    ..
                }
                | Event::Window {
                    win_event: WindowEvent::Shown,
                    ..
                } => skip_render = false,
                _ => {}
            }
        }
        if skip_render {
            sleep(Duration::from_millis(16));
            continue;
        }
        puffin::GlobalProfiler::lock().new_frame();
        let image = match swapchain.acquire_next_image() {
            Ok(image) => Ok(image),
            Err(BackendError::RecreateSwapchain) => {
                swapchain.recreate().unwrap();
                render_pass.clear_fbos(&device.raw);
                continue;
            }
            Err(err) => Err(err),
        }
        .unwrap();
        if recreate_swapchain {
            swapchain.recreate().unwrap();
            render_pass.clear_fbos(&device.raw);
            continue;
        }
        let frame = device.begin_frame().unwrap();
        {
            {
                puffin::profile_scope!("main cb");
                let recorder = frame.main_cb.record(&device).unwrap();
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::Nothing],
                    next_accesses: &[AccessType::ColorAttachmentWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    image: image.image.raw,
                    range: image
                        .image
                        .subresource(SubImage::LayerAndMip(0, 0), vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
                let attachments = [RenderPassAttachment::new(
                    &image.image,
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.1, 0.15, 1.0],
                        },
                    },
                )];
                {
                    let _pass = recorder.render_pass(&device.raw, &render_pass, &attachments, None);
                    let _render_area = swapchain.render_area();
                    /*pass.set_scissor(render_area);
                    pass.set_viewport(vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: render_area.extent.width as f32,
                        height: render_area.extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    });
                    pass.bind_pipeline(&pipeline);
                    pass.bind_index_buffer(&index_buffer);
                    pass.bind_vertex_buffer(&vertex_buffer);
                    pass.draw(3, 1, 0, 0); */
                }
            }
            device
                .submit_render(
                    &frame.main_cb,
                    &[SubmitWaitDesc {
                        semaphore: image.acquire_semaphore,
                        stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    }],
                    &[frame.render_finished],
                )
                .unwrap();
        }
        {
            puffin::profile_scope!("present cb");
            {
                let recorder = frame.presentation_cb.record(&device).unwrap();
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::ColorAttachmentWrite],
                    next_accesses: &[AccessType::Present],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    image: image.image.raw,
                    range: image
                        .image
                        .subresource(SubImage::LayerAndMip(0, 0), vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
            }
            device
                .submit_render(
                    &frame.presentation_cb,
                    &[SubmitWaitDesc {
                        stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        semaphore: frame.render_finished,
                    }],
                    &[image.presentation_finished],
                )
                .unwrap();
        }
        device.end_frame(frame).unwrap();
        swapchain.present_image(image);
    }
    device.wait();

    unsafe { device.raw.destroy_pipeline_cache(pipeline_cache, None) };
    pipeline.free(&device.raw);
    vertex_shader.free(&device.raw);
    fragment_shader.free(&device.raw);
    render_pass.free(&device.raw);

    Ok(())
}
