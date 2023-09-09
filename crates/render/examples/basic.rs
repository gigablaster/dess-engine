use ash::vk;
use dess_render::{
    vulkan::{Buffer, BufferDesc, Device, InstanceBuilder, PhysicalDeviceList, Surface, Swapchain},
    Staging,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

fn main() {
    SimpleLogger::new().init().unwrap();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_resizable(false)
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();
    window.set_title("Basic");
    let instance = InstanceBuilder::default()
        .debug(true)
        .build(window.raw_display_handle())
        .unwrap();
    let surface = Surface::create(
        &instance,
        window.raw_display_handle(),
        window.raw_window_handle(),
    )
    .unwrap();
    let physical_device = instance
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
    let device = Device::create(&instance, physical_device).unwrap();
    let _swapchain = Swapchain::new(&device, surface, [1280, 720]).unwrap();
    let buffer = Buffer::new(
        &device,
        BufferDesc::gpu_only(4000, vk::BufferUsageFlags::TRANSFER_SRC),
        None,
    )
    .unwrap();
    let staging = Staging::new(&device, 32 * 1024 * 1024).unwrap();
}
