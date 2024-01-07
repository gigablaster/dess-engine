use std::time::Instant;

use bevy_tasks::{AsyncComputeTaskPool, ComputeTaskPool, IoTaskPool, TaskPool};
use dess_backend::{
    Device, FindSuitableDevice, FrameResult, Instance, InstanceBuilder, PhysicalDeviceType,
    Surface, Swapchain,
};
use dess_common::TimeFilter;
use dess_engine::{BufferPool, PipelineCache, ResourceManager};
use log::info;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, WindowBuilder, WindowButtons},
};

use crate::{Client, ClientState, InitContext};

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
            InstanceBuilder::default().debug(cfg!(debug_assertions)),
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
                &[PhysicalDeviceType::Discrete, PhysicalDeviceType::Integrated],
            )
            .unwrap();
        let device = Device::new(instance, pdevice).unwrap();
        let buffer_pool = BufferPool::new(&device);
        let resource_manager = ResourceManager::new(&device, &buffer_pool);
        let mut pipeline_cache = PipelineCache::new(&device);
        let mut swapchain = None;
        let mut skip_draw = false;
        let mut paused = false;
        let mut time_filter = TimeFilter::default();
        IoTaskPool::get_or_init(TaskPool::default);
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
        ComputeTaskPool::get_or_init(TaskPool::default);
        info!("Init game");
        {
            self.client.init(InitContext {
                resource_manager: resource_manager.clone(),
                pipeline_cache: &pipeline_cache,
                render_device: &device,
                buffer_pool: &buffer_pool,
            });
        }
        info!("Main loop enter");
        let mut last_timestamp = Instant::now();
        let mut alt_pressed = false;
        event_loop
            .run(|event, elwt| {
                elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);
                puffin::GlobalProfiler::lock().new_frame();
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
                                let resolution =
                                    [window.inner_size().width, window.inner_size().height];
                                swapchain =
                                    Some(Swapchain::new(&device, &surface, resolution).unwrap());
                            }
                            if let Some(current_swapchain) = &swapchain {
                                pipeline_cache.sync();
                                if let FrameResult::NeedRecreate = device
                                    .frame(current_swapchain, |context| self.client.render(context))
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
                            if event.physical_key == PhysicalKey::Code(KeyCode::F11)
                                && event.state.is_pressed()
                            {
                                resource_manager.reload_programs();
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
                        resource_manager.maintain();
                        if self.client.tick(
                            crate::UpdateContext {
                                resource_manager: resource_manager.clone(),
                            },
                            dt,
                        ) == ClientState::Exit
                        {
                            elwt.exit();
                        }

                        window.request_redraw()
                    }
                    _ => {}
                }
            })
            .unwrap();
        info!("Main loop exit");
        drop(swapchain);
        drop(surface);
        drop(buffer_pool);
        drop(device);
        info!("Done.");
    }
}
