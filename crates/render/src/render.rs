use std::sync::{Arc, Mutex};

use ash::vk;
use dess_render_backend::{
    BackendError, CommandBuffer, Device, FrameContext, Image, Instance, PhysicalDeviceList,
    SubImage, SubmitWaitDesc, Surface, Swapchain, SwapchainImage,
};
use sdl2::video::Window;
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier, ImageLayout};

use crate::{RenderError, RenderResult, Staging};

const STAGING_SIZE: usize = 64 * 1024 * 1024;

pub enum GpuType {
    DiscreteOnly,
    PreferDiscrete,
    PrefereIntegrated,
}

pub struct RenderSystem<'a> {
    window: &'a Window,
    device: Arc<Device>,
    swapchain: Swapchain<'a>,
    staging: Mutex<Staging>,
}

pub struct RenderSystemDesc {
    pub debug: bool,
    pub gpu_type: GpuType,
}

impl RenderSystemDesc {
    pub fn new() -> Self {
        Self {
            debug: false,
            gpu_type: GpuType::PreferDiscrete,
        }
    }

    pub fn debug(mut self, value: bool) -> Self {
        self.debug = value;
        self
    }

    pub fn gpu_type(mut self, value: GpuType) -> Self {
        self.gpu_type = value;
        self
    }
}

struct RenderContext<'a> {
    pub cb: &'a CommandBuffer,
    pub image: &'a Image,
}

impl<'a> RenderSystem<'a> {
    pub fn new(window: &'a Window, desc: RenderSystemDesc) -> RenderResult<Self> {
        let instance = Instance::builder().debug(desc.debug).build(window)?;
        let surface = Surface::create(&instance, window)?;
        let allowed_gpus = match desc.gpu_type {
            GpuType::PreferDiscrete => vec![
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ],
            GpuType::DiscreteOnly => vec![vk::PhysicalDeviceType::DISCRETE_GPU],
            GpuType::PrefereIntegrated => vec![
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                vk::PhysicalDeviceType::DISCRETE_GPU,
            ],
        };
        let pdevice = instance
            .enumerate_physical_devices()?
            .find_suitable_device(&surface, &allowed_gpus);
        if let Some(pdevice) = pdevice {
            let device = Device::create(instance, pdevice)?;
            let swapchain = Swapchain::new(&device, surface)?;
            let staging = Staging::new(&device, STAGING_SIZE)?;

            Ok(Self {
                device,
                window,
                swapchain,
                staging: Mutex::new(staging),
            })
        } else {
            Err(crate::RenderError::DeviceNotFound)
        }
    }

    pub fn render<F: FnOnce(RenderContext)>(&mut self, cb: F) -> RenderResult<()> {
        let size = self.window.size();
        let render_area = self.swapchain.render_area();
        if size.0 != render_area.extent.width || size.1 != render_area.extent.height {
            self.device.wait();
            self.swapchain.recreate()?;

            return Err(RenderError::RecreateBuffers);
        }

        let frame = self.device.begin_frame()?;
        let image = match self.swapchain.acquire_next_image() {
            Err(BackendError::RecreateSwapchain) => {
                self.device.wait();
                self.swapchain.recreate()?;

                return Err(RenderError::RecreateBuffers);
            }
            Err(err) => return Err(RenderError::Backend(err)),
            Ok(image) => image,
        };

        let mut staging = self.staging.lock().unwrap();
        staging.upload()?;
        staging.wait()?;

        let context = RenderContext {
            cb: &frame.main_cb,
            image: &image.image,
        };

        cb(context);

        self.device.submit_render(
            &frame.main_cb,
            &[SubmitWaitDesc {
                semaphore: image.acquire_semaphore,
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            }],
            &[frame.render_finished],
        )?;
        frame.presentation_cb.record(&self.device.raw, |recorder| {
            let barrier = ImageBarrier {
                previous_accesses: &[AccessType::Nothing],
                next_accesses: &[AccessType::Present],
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                src_queue_family_index: self.device.graphics_queue.family.index,
                dst_queue_family_index: self.device.graphics_queue.family.index,
                discard_contents: false,
                image: image.image.raw,
                range: image
                    .image
                    .subresource(SubImage::LayerAndMip(0, 0), vk::ImageAspectFlags::COLOR),
            };
            pipeline_barrier(&self.device.raw, *recorder.cb, None, &[], &[barrier]);
        })?;
        self.device.submit_render(
            &frame.presentation_cb,
            &[SubmitWaitDesc {
                semaphore: frame.render_finished,
                stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            }],
            &[image.presentation_finished],
        )?;
        self.device.end_frame(frame)?;
        self.swapchain.present_image(image);

        Ok(())
    }
}
