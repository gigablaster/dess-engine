use std::{io::Cursor, mem::size_of_val};

use arrayvec::ArrayVec;
use ash::vk::{self};
use dess_render::{
    vulkan::{
        create_pipeline_cache, AcquiredSurface, Buffer, BufferDesc, Device, Image, ImageDesc,
        ImageType, InstanceBuilder, PhysicalDeviceList, PipelineState, PipelineStateDesc,
        PipelineVertex, Program, RenderPass, RenderPassAttachment, RenderPassAttachmentDesc,
        RenderPassLayout, ShaderDesc, SubmitWait, Surface, Swapchain,
    },
    DescriptorCache, ImageSubresourceData, Staging,
};
use dess_vfs::{AssetPath, Vfs};
use glam::Mat4;
use image::io::Reader;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, WindowBuilder},
};

#[allow(dead_code)]
#[repr(C, packed)]
#[derive(Debug)]
struct Camera {
    view: Mat4,
    projection: Mat4,
}

#[repr(C, packed)]
struct Vertex {
    pub pos: glam::Vec3,
    pub uv: glam::Vec2,
}

const VERTEX_FORMAT: [vk::VertexInputAttributeDescription; 2] = [
    vk::VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        location: 1,
        binding: 0,
        format: vk::Format::R32G32_SFLOAT,
        offset: 12,
    },
];

impl PipelineVertex for Vertex {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription] {
        &VERTEX_FORMAT
    }
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
    let mut staging = Staging::new(&device, 32 * 1024 * 1024).unwrap();
    let mut desciptors = DescriptorCache::new(&device).unwrap();
    let handle1 = desciptors.create(program.descriptor_set(0)).unwrap();
    let handle2 = desciptors.create(program.descriptor_set(1)).unwrap();

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
                    .clear_input()
                    .initial_layout(vk::ImageLayout::UNDEFINED),
            ),
        ),
    )
    .unwrap();

    let cache = create_pipeline_cache(device.raw()).unwrap();
    let pipeline = PipelineState::new::<Vertex>(
        &program,
        PipelineStateDesc::new(&render_pass)
            .face_cull(false)
            .depth_write(false)
            .subpass(0),
        &cache,
    )
    .unwrap();

    let vertices = [
        Vertex {
            pos: glam::Vec3::new(-0.5, 0.5, 0.0),
            uv: glam::Vec2::new(0.0, 1.0),
        },
        Vertex {
            pos: glam::Vec3::new(0.5, 0.5, 0.0),
            uv: glam::Vec2::new(1.0, 1.0),
        },
        Vertex {
            pos: glam::Vec3::new(0.5, -0.5, 0.0),
            uv: glam::Vec2::new(1.0, 0.0),
        },
        Vertex {
            pos: glam::Vec3::new(-0.5, -0.5, 0.0),
            uv: glam::Vec2::new(0.0, 0.0),
        },
    ];

    let indices = [0u16, 1u16, 2u16, 0u16, 3u16, 2u16];

    let vbo = Buffer::new(
        &device,
        BufferDesc::gpu_only(
            size_of_val(&vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ),
    )
    .unwrap();

    let ibo = Buffer::new(
        &device,
        BufferDesc::gpu_only(
            size_of_val(&indices),
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ),
    )
    .unwrap();

    staging.upload_buffer(&vbo, 0, &vertices).unwrap();
    staging.upload_buffer(&ibo, 0, &indices).unwrap();

    desciptors
        .set_uniform(
            handle1,
            0,
            &Camera {
                projection: Mat4::IDENTITY,
                view: Mat4::IDENTITY,
            },
        )
        .unwrap();

    desciptors.set_uniform(handle2, 0, &Mat4::IDENTITY).unwrap();

    let image_data = vfs
        .load(AssetPath::Content("images/test.png"))
        .unwrap()
        .load()
        .unwrap();
    let loaded_image = Reader::new(Cursor::new(image_data))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    let image_desc = ImageDesc::new(
        vk::Format::R8G8B8A8_UNORM,
        ImageType::Tex2D,
        [loaded_image.width(), loaded_image.height()],
    )
    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
    let image = Image::texture(&device, image_desc).unwrap();

    desciptors
        .set_image(
            handle2,
            1,
            &image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )
        .unwrap();
    let pixels = loaded_image.as_flat_samples();
    staging
        .upload_image(
            &image,
            &[&ImageSubresourceData {
                data: pixels.samples,
                row_pitch: loaded_image.width() as usize,
            }],
        )
        .unwrap();

    let mut alt_pressed = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let mut need_recreate = false;

        let mut wait = ArrayVec::<SubmitWait, 3>::new();

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
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => match swapchain.acquire_next_image() {
                Ok(AcquiredSurface::Image(backbuffer)) => {
                    desciptors
                        .update_descriptors()
                        .unwrap()
                        .into_iter()
                        .for_each(|x| wait.push(x));
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
                                .render_pass(&render_pass, &[backbuffer], Some(depth), |cb| {
                                    cb.bind_pipeline(pipeline.pipeline());
                                    cb.bind_descriptor_set(
                                        0,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        program.pipeline_layout(),
                                        desciptors.get(handle1).unwrap().raw,
                                    );
                                    cb.bind_descriptor_set(
                                        1,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        program.pipeline_layout(),
                                        desciptors.get(handle2).unwrap().raw,
                                    );
                                    cb.bind_vertex_buffer(vbo.raw(), 0);
                                    cb.bind_index_buffer(ibo.raw(), 0);
                                    let render_area = swapchain.render_area();
                                    cb.set_scissor(render_area);
                                    cb.set_viewport(vk::Viewport {
                                        x: 0.0,
                                        y: 0.0,
                                        width: render_area.extent.width as _,
                                        height: render_area.extent.height as _,
                                        min_depth: 0.0,
                                        max_depth: 1.0,
                                    });
                                    cb.draw(6, 1, 0, 0);
                                })
                                .unwrap();
                        })
                        .unwrap();
                    staging
                        .upload()
                        .unwrap()
                        .into_iter()
                        .for_each(|x| wait.push(x));

                    wait.push(SubmitWait::ColorAttachmentOutput(
                        backbuffer.acquire_semaphore,
                    ));
                    device
                        .submit(frame.main_cb(), &wait, &[backbuffer.rendering_finished])
                        .unwrap();

                    device.end_frame(frame);
                    swapchain.present_image(backbuffer);
                }
                Ok(AcquiredSurface::NeedRecreate) => need_recreate = true,
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
