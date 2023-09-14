use ash::vk::{self};
use dess_render::{
    vulkan::{
        AcquireError, Buffer, BufferDesc, Device, Image, ImageDesc, ImageType, InstanceBuilder,
        PhysicalDeviceList, Program, RenderPass, RenderPassAttachment, RenderPassAttachmentDesc,
        RenderPassLayout, ShaderDesc, SubmitWait, Surface, Swapchain,
    },
    DescriptorCache, Staging,
};
use dess_vfs::{AssetPath, Vfs};
use glam::Mat4;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, WindowBuilder},
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
    let _handle1 = desciptors.create(program.descriptor_set(0)).unwrap();
    let _handle2 = desciptors.create(program.descriptor_set(1)).unwrap();
    let _image = Image::texture(
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

    let mut depth = Image::texture(
        &device,
        ImageDesc::new(
            vk::Format::D24_UNORM_S8_UINT,
            ImageType::Tex2D,
            [window.inner_size().width, window.inner_size().height],
        )
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
    )
    .unwrap();
    depth.name("Depth");
    let render_pass = RenderPass::new(
        &device,
        RenderPassLayout::new(
            &[
                RenderPassAttachmentDesc::color(swapchain.backbuffer_format())
                    .clear_input()
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            ],
            Some(
                RenderPassAttachmentDesc::depth(depth.desc().format)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
            ),
        ),
    )
    .unwrap();

    let mut alt_pressed = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let mut need_recreate = false;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => need_recreate = true,
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(vk) = input.virtual_keycode {
                        if vk == VirtualKeyCode::Return
                            && input.state == ElementState::Pressed
                            && alt_pressed
                        {
                            match window.fullscreen() {
                                None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
                                Some(_) => window.set_fullscreen(None),
                            }
                        }
                    }
                }
                WindowEvent::ModifiersChanged(modifier) => alt_pressed = modifier.alt(),
                _ => {}
            },
            Event::RedrawRequested(_) => match swapchain.acquire_next_image() {
                Err(AcquireError::Suboptimal) | Err(AcquireError::OutOfDate) => {
                    need_recreate = true;
                }
                Ok(backbuffer) => {
                    desciptors.update_descriptors().unwrap();
                    let frame = device.begin_frame().unwrap();
                    frame
                        .main_cb()
                        .record(device.raw(), |recorder| {
                            let backbuffer = RenderPassAttachment::color(
                                &backbuffer.image,
                                [0.01, 0.01, 0.222, 1.0],
                            );

                            let depth = RenderPassAttachment::depth(&depth, 1.0);

                            recorder
                                .render_pass(&render_pass, &[backbuffer], Some(depth), |_| {})
                                .unwrap();
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
            let current_size = swapchain.render_area();
            if inner_size.width > 0
                && inner_size.height > 0
                && current_size.extent.width != inner_size.width
                && current_size.extent.height != inner_size.height
            {
                swapchain
                    .recreate(
                        &device,
                        [window.inner_size().width, window.inner_size().height],
                    )
                    .unwrap();
                depth = Image::texture(
                    &device,
                    ImageDesc::new(
                        vk::Format::D24_UNORM_S8_UINT,
                        ImageType::Tex2D,
                        [window.inner_size().width, window.inner_size().height],
                    )
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                )
                .unwrap();
                depth.name("Depth");
            }
        }
    });
}
