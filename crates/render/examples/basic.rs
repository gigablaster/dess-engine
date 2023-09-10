use ash::vk;
use dess_render::{
    vulkan::{
        Buffer, BufferDesc, Device, InstanceBuilder, PhysicalDeviceList, Program, ShaderDesc,
        Surface, Swapchain,
    },
    DescriptorCache, Staging,
};
use glam::Mat4;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

struct Camera {
    view: Mat4,
    projection: Mat4,
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
    let _buffer = Buffer::new(
        &device,
        BufferDesc::gpu_only(4000, vk::BufferUsageFlags::TRANSFER_SRC),
        None,
    )
    .unwrap();
    let vertex = dess_vfs::get("shaders/unlit.vert.spv").unwrap();
    let fragment = dess_vfs::get("shaders/unlit.frag.spv").unwrap();
    let shaders = [
        ShaderDesc::vertex(vertex.data()),
        ShaderDesc::fragment(fragment.data()),
    ];
    let program = Program::new(&device, &shaders).unwrap();
    let _staging = Staging::new(&device, 32 * 1024 * 1024).unwrap();
    let mut desciptors = DescriptorCache::new(&device).unwrap();
    let handle = desciptors.create(program.descriptor_set(0)).unwrap();
    desciptors
        .set_uniform(
            handle,
            0,
            &Camera {
                view: Mat4::IDENTITY,
                projection: Mat4::IDENTITY,
            },
        )
        .unwrap();
    desciptors.update_descriptors().unwrap();
    desciptors.remove(handle);
}
