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

use render_backend::vulkan::{
    Device, Image, ImageDesc, ImageType, Instance, PhysicalDeviceList, RenderPass,
    RenderPassAttachmentDesc, RenderPassDesc, Surface, Swapchain, SwapchainDesc,
};
use sdl2::event::Event;

fn main() -> Result<(), String> {
    simple_logger::init().unwrap();
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let _timer = sdl.timer()?;

    let window = video
        .window("vk-clear", 1280, 720)
        .position_centered()
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
    let desc = SwapchainDesc {
        format: Swapchain::select_surface_format(
            &Swapchain::enumerate_surface_formats(&device, &surface).unwrap(),
        )
        .unwrap(),
        dims: vk::Extent2D {
            width: window.size().0,
            height: window.size().1,
        },
        vsync: true,
    };

    let color_attachment_desc = RenderPassAttachmentDesc::new(desc.format.format)
        .garbage_input()
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let render_pass_desc = RenderPassDesc {
        color_attachments: &[color_attachment_desc],
        depth_attachment: None,
    };
    let render_pass = RenderPass::new(&device, &render_pass_desc).unwrap();

    let mut swapchain = Swapchain::new(&device, &surface, &desc).unwrap();

    let _image = Image::new(
        &device,
        ImageDesc::new(vk::Format::R8G8B8A8_SRGB, ImageType::Tex2D, [256, 256])
            .usage(vk::ImageUsageFlags::SAMPLED)
            .flags(vk::ImageCreateFlags::empty()),
        None,
    )
    .unwrap();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
            {
                let frame = device.begin_frame().unwrap();
                let image = swapchain.acquire_next_image().unwrap();
                {
                    let cb = frame.command_buffer.begin().unwrap();
                    cb.begin_pass(
                        [window.size().0, window.size().1],
                        &render_pass,
                        &[&image.image],
                        None,
                    );
                    cb.clear(
                        [window.size().0, window.size().1],
                        &[vk::ClearColorValue {
                            float32: [1.0, 0.0, 0.0, 1.0],
                        }],
                        None,
                    );
                    cb.end_pass();
                }
                device.submit_render(&frame.command_buffer, &image).unwrap();
                device.end_frame(frame).unwrap();
                swapchain.present_image(image);
            }
        }
    }
    device.wait();

    Ok(())
}
