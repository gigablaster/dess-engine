use std::time::Instant;

use ash::vk;
use dess_backend::{
    vulkan::{
        Device, FrameResult, Instance, InstanceBuilder, PhysicalDeviceList, Surface, Swapchain,
    },
    ResourcePool,
};
use dess_common::TimeFilter;
use log::info;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, WindowBuilder, WindowButtons},
};

use crate::{Client, ClientState, RenderContext};

pub struct Runner<T: Client> {
    client: T,
    title: String,
    dims: [u32; 2],
}

impl<T: Client> Runner<T> {
    pub fn new(client: T, title: &str) -> Self {
        Self {
            client,
            dims: [1280, 720],
            title: title.to_owned(),
        }
    }

    pub fn run(&mut self) {
        simple_logger::init().unwrap();
        info!("Init systems");

        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title(&self.title)
            .with_inner_size(PhysicalSize::new(self.dims[0], self.dims[1]))
            .with_resizable(false)
            .with_enabled_buttons(WindowButtons::MINIMIZE | WindowButtons::CLOSE)
            .build(&event_loop)
            .unwrap();

        let instance = Instance::new(
            InstanceBuilder::default().debug(true),
            window.raw_display_handle(),
        )
        .unwrap();
        let surface = Surface::new(
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
        )
        .unwrap();
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
        let device = Device::new(instance, pdevice).unwrap();
        let pool = ResourcePool::new(&device).unwrap();
        let mut swapchain = None;
        let mut skip_draw = false;
        let mut paused = false;
        let mut time_filter = TimeFilter::default();
        info!("Main loop enter");
        let mut last_timestamp = Instant::now();
        let mut alt_pressed = false;
        event_loop
            .run(|event, elwt| {
                elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
                match event {
                    Event::Suspended => paused = true,
                    Event::Resumed => {
                        paused = false;
                        last_timestamp = Instant::now();
                    }
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::RedrawRequested => {
                            if skip_draw {
                                return;
                            }
                            if swapchain.is_none() {
                                pool.purge();
                                let resolution =
                                    [window.inner_size().width, window.inner_size().height];
                                swapchain =
                                    Some(Swapchain::new(&device, &surface, resolution).unwrap());
                            }
                            if let Some(current_swapchain) = &swapchain {
                                if let FrameResult::NeedRecreate = device
                                    .frame(current_swapchain, |context| {
                                        let context = RenderContext {
                                            frame: context,
                                            pool: &pool,
                                        };
                                        self.client.render(context)
                                    })
                                    .unwrap()
                                {}
                            }
                        }
                        WindowEvent::Resized(new_size) => {
                            skip_draw = new_size.width == 0 || new_size.height == 0;
                            swapchain = None;
                        }
                        WindowEvent::ModifiersChanged(mods) => {
                            alt_pressed = mods.state().alt_key();
                        }
                        WindowEvent::KeyboardInput { event, .. } => {
                            // dbg!(event.physical_key);
                            if event.physical_key == PhysicalKey::Code(KeyCode::Enter)
                                && event.state.is_pressed()
                                && alt_pressed
                            {
                                if window.fullscreen().is_none() {
                                    window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                } else {
                                    window.set_fullscreen(None);
                                }
                                swapchain = None;
                            }
                        }
                        _ => {}
                    },
                    Event::AboutToWait => {
                        if paused {
                            last_timestamp = Instant::now();
                            return;
                        }
                        let current_timestamp = Instant::now();
                        let dt =
                            time_filter.sample((current_timestamp - last_timestamp).as_secs_f64());
                        last_timestamp = current_timestamp;
                        if self.client.tick(dt) == ClientState::Exit {
                            elwt.exit();
                        }

                        window.request_redraw()
                    }
                    _ => {}
                }
            })
            .unwrap();
        drop(swapchain);
        drop(surface);
        drop(pool);
        drop(device);
        info!("Main loop exit");
    }
}
