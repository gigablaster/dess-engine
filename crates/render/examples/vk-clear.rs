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

use render::{
    vulkan::{
        Device, Image, ImageDesc, ImageType, Instance, PhysicalDeviceList, RenderPass,
        RenderPassAttachment, RenderPassAttachmentDesc, RenderPassLayout, Surface, Swapchain,
    },
    BackendError,
};
use sdl2::event::{Event, WindowEvent};
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier};

fn main() -> Result<(), String> {
    simple_logger::init().unwrap();
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let _timer = sdl.timer()?;

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
        let _image = Image::new(
            &device,
            ImageDesc::new(vk::Format::R8G8B8A8_SRGB, ImageType::Tex2D, [256, 256])
                .usage(vk::ImageUsageFlags::SAMPLED)
                .flags(vk::ImageCreateFlags::empty()),
            None,
        )
        .unwrap();
        let frame = device.begin_frame().unwrap();
        let image = match swapchain.acquire_next_image() {
            Ok(image) => Ok(image),
            Err(BackendError::RecreateSwapchain) => {
                swapchain.recreate().unwrap();
                render_pass.clear_fbos();
                continue;
            }
            Err(err) => Err(err),
        }
        .unwrap();
        if recreate_swapchain {
            swapchain.recreate().unwrap();
            render_pass.clear_fbos();
            continue;
        }
        {
            let recorder = frame.command_buffer.record(&device).unwrap();
            let barrier = ImageBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::ColorAttachmentWrite],
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                src_queue_family_index: 0,
                dst_queue_family_index: 0,
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
            let attachments = [RenderPassAttachment::new(
                &image.image,
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.5, 0.25, 1.0],
                    },
                },
            )];
            {
                let _pass = recorder.render_pass(&render_pass, &attachments, None);
            }

            let barrier = ImageBarrier {
                previous_accesses: &[AccessType::ColorAttachmentWrite],
                next_accesses: &[AccessType::Present],
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: true,
                src_queue_family_index: 0,
                dst_queue_family_index: 0,
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
        }
        device.submit_render(&frame.command_buffer, &image).unwrap();
        device.end_frame(frame).unwrap();
        swapchain.present_image(image);
    }
    device.wait();

    Ok(())
}
