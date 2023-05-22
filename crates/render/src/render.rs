use std::sync::{Arc, Mutex};

use ash::vk;
use dess_render_backend::{
    BackendError, Buffer, CommandBuffer, Device, Image, Instance, PhysicalDeviceList, SubImage,
    SubmitWaitDesc, Surface, Swapchain,
};
use sdl2::video::Window;
use vk_sync::{cmd::pipeline_barrier, AccessType, ImageBarrier, ImageLayout};

use crate::{geometry::GeometryCache, RenderError, RenderResult, Staging};

const STAGING_SIZE: usize = 64 * 1024 * 1024;
const GEOMETRY_CACHE_SIZE: usize = 64 * 1024 * 1024;

const RING_BUFFER_MASK: u32 = 1 << 31;

pub enum BufferOffset {
    Main(u32),
    Ring(u32),
}

pub struct Offset {
    raw: u32,
}

impl Offset {
    pub fn main(offset: u32) -> Self {
        assert!(offset & RING_BUFFER_MASK == 0);
        Self { raw: offset }
    }

    pub fn ring(offset: u32) -> Self {
        assert!(offset & RING_BUFFER_MASK == 0);
        Self {
            raw: offset | RING_BUFFER_MASK,
        }
    }

    pub fn offset(&self) -> BufferOffset {
        let raw_offset = self.raw & !RING_BUFFER_MASK;
        if self.raw & RING_BUFFER_MASK == 0 {
            BufferOffset::Main(raw_offset)
        } else {
            BufferOffset::Ring(raw_offset)
        }
    }

    pub fn same_buffer(&self, other: &Offset) -> bool {
        self.raw & RING_BUFFER_MASK == other.raw & RING_BUFFER_MASK
    }
}

#[repr(C, align(16))]
pub struct RenderOp {
    pso: u32,
    vertex_offset: Offset,
    index_offset: Offset,
    primitve_count: u32,
    descs: [u32; 4],
}

pub enum GpuType {
    DiscreteOnly,
    PreferDiscrete,
    PrefereIntegrated,
}

pub struct RenderSystem<'a> {
    window: &'a Window,
    device: Arc<Device>,
    swapchain: Mutex<Swapchain<'a>>,
    staging: Mutex<Staging>,
    geo_cache: Mutex<GeometryCache>,
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
    geo_cache: &'a Buffer,
}

impl<'a> RenderContext<'a> {
    pub fn render(&self, _rops: &[RenderOp]) {}
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
            let geo_cache = GeometryCache::new(&device, GEOMETRY_CACHE_SIZE)?;

            Ok(Self {
                device,
                window,
                swapchain: Mutex::new(swapchain),
                staging: Mutex::new(staging),
                geo_cache: Mutex::new(geo_cache),
            })
        } else {
            Err(crate::RenderError::DeviceNotFound)
        }
    }

    pub fn render_frame<F: FnOnce(RenderContext)>(&self, frame_cb: F) -> RenderResult<()> {
        let size = self.window.size();
        let mut swapchain = self.swapchain.lock().unwrap();
        let render_area = swapchain.render_area();
        if size.0 != render_area.extent.width || size.1 != render_area.extent.height {
            self.device.wait();
            swapchain.recreate()?;

            return Err(RenderError::RecreateBuffers);
        }

        let frame = self.device.begin_frame()?;
        let image = match swapchain.acquire_next_image() {
            Err(BackendError::RecreateSwapchain) => {
                self.device.wait();
                swapchain.recreate()?;

                return Err(RenderError::RecreateBuffers);
            }
            Err(err) => return Err(RenderError::Backend(err)),
            Ok(image) => image,
        };

        let mut staging = self.staging.lock().unwrap();
        staging.upload()?;
        staging.wait()?;

        let geo_cache = self.geo_cache.lock().unwrap();

        let context = RenderContext {
            cb: &frame.main_cb,
            image: &image.image,
            geo_cache: &geo_cache.buffer,
        };

        frame_cb(context);

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
        swapchain.present_image(image);

        Ok(())
    }
}
