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
use std::{sync::Arc, thread::sleep, time::Duration};

use ash::vk;

use render::{
    vulkan::{
        Device, Image, ImageDesc, ImageType, Instance, PhysicalDeviceList, RenderPass,
        RenderPassAttachment, RenderPassAttachmentDesc, RenderPassLayout, SubmitWaitDesc, Surface,
        Swapchain,
    },
    BackendError,
};
use sdl2::event::{Event, WindowEvent};
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier};

fn create_rt(device: &Arc<Device>, format: vk::Format, width: u32, height: u32) -> Image {
    let rt_desc = ImageDesc::new(format, ImageType::Tex2D, [width, height])
        .flags(vk::ImageCreateFlags::empty())
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);

    Image::new(&device, rt_desc, None).unwrap()
}

fn main() -> Result<(), String> {
    simple_logger::init().unwrap();
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let _timer = sdl.timer()?;

    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    eprintln!("Serving demo profile data on {server_addr}");

    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();

    puffin::set_scopes_on(true);
    let window = video
        .window("vk-clear", 1280, 720)
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

    let device = Device::create(&instance, &pdevice).unwrap();

    let mut swapchain = Swapchain::new(&device, surface).unwrap();

    let color_attachment_desc =
        RenderPassAttachmentDesc::new(swapchain.backbuffer_format()).clear_input();
    let render_pass_desc = RenderPassLayout {
        color_attachments: &[color_attachment_desc],
        depth_attachment: None,
    };
    let render_pass = RenderPass::new(&device, render_pass_desc).unwrap();

    let mut rt = create_rt(
        &device,
        swapchain.backbuffer_format(),
        window.size().0,
        window.size().1,
    );

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
                rt = create_rt(
                    &device,
                    swapchain.backbuffer_format(),
                    window.size().0,
                    window.size().1,
                );
                swapchain.recreate().unwrap();
                render_pass.clear_fbos();
                continue;
            }
            Err(err) => Err(err),
        }
        .unwrap();
        if recreate_swapchain {
            rt = create_rt(
                &device,
                swapchain.backbuffer_format(),
                window.size().0,
                window.size().1,
            );
            swapchain.recreate().unwrap();
            render_pass.clear_fbos();
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
                    image: rt.raw,
                    range: rt.subresource(0, 0, vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
                let attachments = [RenderPassAttachment::new(
                    &rt,
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [1.0, 0.5, 0.25, 1.0],
                        },
                    },
                )];
                {
                    let _pass = recorder.render_pass(&render_pass, &attachments, None);
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
                    next_accesses: &[AccessType::TransferRead],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.graphics_queue.family.index,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    image: rt.raw,
                    range: rt.subresource(0, 0, vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::Nothing],
                    next_accesses: &[AccessType::TransferWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: 0,
                    dst_queue_family_index: device.transfer_queue.family.index,
                    image: image.image.raw,
                    range: image.image.subresource(0, 0, vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
                let region = [vk::ImageCopy::builder()
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .extent(vk::Extent3D {
                        width: window.size().0,
                        height: window.size().1,
                        depth: 1,
                    })
                    .build()];
                unsafe {
                    recorder.device.cmd_copy_image(
                        frame.presentation_cb.raw,
                        rt.raw,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image.image.raw,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &region,
                    )
                };
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[AccessType::Present],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    image: image.image.raw,
                    range: image.image.subresource(0, 0, vk::ImageAspectFlags::COLOR),
                };
                pipeline_barrier(
                    recorder.device,
                    *recorder.cb,
                    None,
                    &[],
                    slice::from_ref(&barrier),
                );
                let barrier = ImageBarrier {
                    previous_accesses: &[AccessType::TransferRead],
                    next_accesses: &[AccessType::ColorAttachmentWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: true,
                    src_queue_family_index: device.transfer_queue.family.index,
                    dst_queue_family_index: device.graphics_queue.family.index,
                    image: rt.raw,
                    range: rt.subresource(0, 0, vk::ImageAspectFlags::COLOR),
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
                .submit_transfer(
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

    Ok(())
}
