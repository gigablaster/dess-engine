use ash::vk::{self};
use dess_render::{
    vulkan::{
        AcquireError, Buffer, BufferDesc, Device, Image, ImageDesc, ImageType, InstanceBuilder,
        PhysicalDeviceList, Program, RenderPassAttachment, ShaderDesc, SubImage, SubmitWait,
        Surface, Swapchain,
    },
    DescriptorCache, Staging,
};
use dess_vfs::{AssetPath, Vfs};
use glam::Mat4;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use vk_sync::{AccessType, ImageBarrier, ImageLayout};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[allow(dead_code)]
struct Camera {
    view: Mat4,
    projection: Mat4,
}

fn main() {
    SimpleLogger::new().init().unwrap();
    let vfs = Vfs::new("gigablaster", "engine basic example");
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
    let mut swapchain = Swapchain::new(&device, surface, [1280, 720]).unwrap();
    let mut vertex = vfs
        .load(AssetPath::Content("shaders/unlit.vert.spv"))
        .unwrap();
    let mut fragment = vfs
        .load(AssetPath::Content("shaders/unlit.frag.spv"))
        .unwrap();
    let vertex = vertex.load().unwrap();
    let fragment = fragment.load().unwrap();
    let shaders = [ShaderDesc::vertex(&vertex), ShaderDesc::fragment(&fragment)];
    let program = Program::new(&device, &shaders).unwrap();
    let _staging = Staging::new(&device, 32 * 1024 * 1024).unwrap();
    let mut desciptors = DescriptorCache::new(&device).unwrap();
    let handle1 = desciptors.create(program.descriptor_set(0)).unwrap();
    let handle2 = desciptors.create(program.descriptor_set(1)).unwrap();
    let camera = Camera {
        view: Mat4::IDENTITY,
        projection: Mat4::IDENTITY,
    };
    let image = Image::texture(
        &device,
        ImageDesc::new(
            vk::Format::A8B8G8R8_UNORM_PACK32,
            ImageType::Tex2D,
            [512, 512],
        )
        .usage(vk::ImageUsageFlags::SAMPLED),
    )
    .unwrap();
    let _buffer = Buffer::new(
        &device,
        BufferDesc::gpu_only(
            1024,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ),
    )
    .unwrap();
    desciptors.set_uniform(handle1, 0, &camera).unwrap();
    desciptors.set_uniform(handle2, 0, &Mat4::IDENTITY).unwrap();
    desciptors
        .set_image(
            handle2,
            1,
            &image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )
        .unwrap();
    desciptors.update_descriptors().unwrap();
    desciptors.remove(handle1);
    desciptors.remove(handle2);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let mut need_recreate = false;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => need_recreate = true,
                _ => {}
            },
            Event::RedrawRequested(_) => match swapchain.acquire_next_image() {
                Err(AcquireError::Suboptimal) | Err(AcquireError::OutOfDate) => {
                    need_recreate = true;
                }
                Ok(backbuffer) => {
                    let frame = device.begin_frame().unwrap();
                    frame
                        .main_cb()
                        .record(device.raw(), |recorder| {
                            {
                                let backbuffer = RenderPassAttachment::color_target(
                                    &backbuffer.image,
                                    Some(glam::Vec4::new(0.01, 0.01, 0.222, 1.0)),
                                );

                                recorder.render_pass(&[backbuffer], None, |_| {}).unwrap();
                            }

                            recorder.barrier(
                                None,
                                &[],
                                &[ImageBarrier {
                                    previous_accesses: &[AccessType::Nothing],
                                    next_accesses: &[AccessType::Present],
                                    previous_layout: ImageLayout::Optimal,
                                    next_layout: ImageLayout::Optimal,
                                    discard_contents: false,
                                    src_queue_family_index: device.queue_index(),
                                    dst_queue_family_index: device.queue_index(),
                                    image: backbuffer.image.raw(),
                                    range: backbuffer.image.subresource_range(
                                        SubImage::LayerAndMip(0, 0),
                                        vk::ImageAspectFlags::COLOR,
                                    ),
                                }],
                            )
                        })
                        .unwrap();

                    device
                        .submit(
                            frame.main_cb(),
                            &[SubmitWait::ColorAttachmentOutput(
                                &backbuffer.acquire_semaphore,
                            )],
                            &[backbuffer.rendering_finished],
                        )
                        .unwrap();

                    device.end_frame(frame);
                    swapchain.present_image(backbuffer);
                }
                Err(err) => panic!("Error: {:?}", err),
            },
            _ => {}
        }
        if need_recreate {
            let inner_size = window.inner_size();
            if inner_size.width > 0 && inner_size.height > 0 {
                swapchain
                    .recreate(
                        &device,
                        [window.inner_size().width, window.inner_size().height],
                    )
                    .unwrap();
            }
        }
    });
}
