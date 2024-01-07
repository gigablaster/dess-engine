use std::{f32::consts::PI, marker::PhantomData, mem, sync::Arc};

use dess_assets::{ContentSource, GltfSource, ShaderSource};
use dess_backend::{
    BindGroupLayoutDesc, BindType, BindingDesc, ClearRenderTarget, DepthCompareOp, DrawStream,
    Format, FrameContext, ImageLayout, ImageUsage, InputVertexAttributeDesc, InputVertexStreamDesc,
    RasterPipelineCreateDesc, RasterPipelineHandle, RenderPassHandle, RenderPassLayout,
    RenderTargetDesc, ShaderStage, SubpassLayout, TemporaryImageDims, EMPTY_BIND_LAYOUT,
    {BindGroupHandle, RenderTarget},
};
use dess_common::GameTime;
use dess_engine::{
    render::{BasicVertex, BASIC_MESH_LAYOUT, BASIC_VERTEX_LAYOUT},
    ModelCollection, ResourceLoader, MESH_PBR_MATERIAL_LAYOUT, PACKED_MESH_OBJECT_LAYOUT,
};
use dess_runner::{Client, InitContext, Runner, UpdateContext};
use glam::{vec2, vec3, vec3a};

const MAX_MATRICES_PER_DRAW: usize = 256;

#[derive(Default)]
struct RenderDemo<'a> {
    model: Arc<ModelCollection>,
    pipeline: RasterPipelineHandle,
    tonemapping: RasterPipelineHandle,
    scene_bind_group: BindGroupHandle,
    draw_bind_group: BindGroupHandle,
    tonemapping_bind_group: BindGroupHandle,
    skybox_pipeline: RasterPipelineHandle,
    skybox_bind_group: BindGroupHandle,
    main_pass: RenderPassHandle,
    tonemapping_pass: RenderPassHandle,
    rotation: f32,
    _phantom: PhantomData<&'a ()>,
}

const MAIN_PASS_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: ShaderStage::Graphics,
    set: &[
        BindingDesc {
            slot: 0,
            name: "pass",
            ty: BindType::UniformBuffer,
            count: 1,
        },
        BindingDesc {
            slot: 1,
            name: "light",
            ty: BindType::UniformBuffer,
            count: 1,
        },
        BindingDesc {
            slot: 32,
            name: "base_sampler",
            ty: BindType::Sampler,
            count: 1,
        },
    ],
};

const SKYBOX_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: ShaderStage::Graphics,
    set: &[
        BindingDesc {
            slot: 0,
            name: "pass",
            ty: BindType::UniformBuffer,
            count: 1,
        },
        BindingDesc {
            slot: 1,
            name: "light",
            ty: BindType::UniformBuffer,
            count: 1,
        },
    ],
};

const TONEMAPPING_PASS_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: ShaderStage::Graphics,
    set: &[
        BindingDesc {
            slot: 0,
            name: "hdr",
            ty: BindType::SampledImage,
            count: 1,
        },
        BindingDesc {
            slot: 1,
            name: "params",
            ty: BindType::UniformBuffer,
            count: 1,
        },
        BindingDesc {
            slot: 32,
            name: "sampler",
            ty: BindType::Sampler,
            count: 1,
        },
    ],
};

const DRAW_CALL_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: ShaderStage::Graphics,
    set: &[BindingDesc {
        slot: 0,
        name: "draw",
        ty: BindType::DynamicStorageBuffer,
        count: 1,
    }],
};

const DRAW_PIPELINE_LAYOUT: [BindGroupLayoutDesc; 4] = [
    MAIN_PASS_BIND_LAYOUT,
    MESH_PBR_MATERIAL_LAYOUT,
    PACKED_MESH_OBJECT_LAYOUT,
    DRAW_CALL_BIND_LAYOUT,
];

const POSTPORCESSING_PIPELINE_LAYOUT: [BindGroupLayoutDesc; 4] = [
    TONEMAPPING_PASS_BIND_LAYOUT,
    EMPTY_BIND_LAYOUT,
    EMPTY_BIND_LAYOUT,
    EMPTY_BIND_LAYOUT,
];

const SKYBOX_PIPELINE_LAYOUT: [BindGroupLayoutDesc; 4] = [
    SKYBOX_BIND_LAYOUT,
    EMPTY_BIND_LAYOUT,
    EMPTY_BIND_LAYOUT,
    DRAW_CALL_BIND_LAYOUT,
];

#[repr(C, align(16))]
struct SceneUniform {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub eye_position: glam::Vec3,
}

#[repr(C, align(16))]
struct DirectionalLight {
    pub direction: glam::Vec3A,
    pub color: glam::Vec3A,
}

#[repr(C, align(16))]
struct AmbientLight {
    pub top: glam::Vec3A,
    pub middle: glam::Vec3A,
    pub bottom: glam::Vec3A,
}

#[repr(C, align(16))]
struct LightUniform {
    pub main: DirectionalLight,
    pub fill: DirectionalLight,
    pub back: DirectionalLight,
    pub ambient: AmbientLight,
}

#[repr(C, align(16))]
struct Tonemapping {
    pub expouse: f32,
}

const CORNERS: [glam::Vec3; 8] = [
    vec3(-1.0, -1.0, -1.0),
    vec3(-1.0, 1.0, -1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(1.0, -1.0, -1.0),
    vec3(1.0, -1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(-1.0, 1.0, 1.0),
    vec3(-1.0, -1.0, 1.0),
];

const FACES: [[u16; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [4, 3, 2, 5],
    [0, 7, 6, 1],
    [1, 6, 5, 2],
    [7, 0, 3, 4],
];

#[repr(C, packed)]
struct SkyboxVertex {
    positon: glam::Vec3,
}

pub const SKY_VERTEX_LAYOUT: [InputVertexStreamDesc; 1] = [InputVertexStreamDesc {
    attributes: &[InputVertexAttributeDesc {
        format: Format::RGB32_SFLOAT,
        locaion: 0,
        binding: 0,
        offset: 0,
    }],
    stride: mem::size_of::<SkyboxVertex>(),
}];

impl<'a> RenderDemo<'a> {
    fn create_draw_stream(&self, context: &FrameContext, rotation: f32) -> DrawStream {
        puffin::profile_function!();
        let mut stream = DrawStream::new(
            self.scene_bind_group,
            context.get_backbuffer_size().into(),
            0,
        );
        stream.set_pipeline(self.pipeline);
        for model in self.model.models.values() {
            let mut bones = Vec::with_capacity(model.bones.len());
            for (index, bone) in model.bones.iter().enumerate() {
                let parent = model.bone_parents[index];
                if parent == u32::MAX {
                    bones.push(glam::Mat4::from_rotation_y(rotation) * glam::Mat4::from(*bone));
                } else {
                    bones.push(bones[parent as usize] * glam::Mat4::from(*bone))
                }
            }
            let temp = context.get_temp_buffer_offset(&bones).unwrap();
            stream.set_dynamic_buffer_offset(0, Some(temp));
            stream.set_bind_group(3, Some(self.draw_bind_group));
            for (bone_idx, mesh_idx) in &model.instances {
                let mesh = &model.static_meshes[*mesh_idx as usize];
                stream.set_vertex_buffer(0, Some(mesh.vertices));
                stream.set_vertex_buffer(1, Some(mesh.attributes));
                stream.set_index_buffer(Some(mesh.indices));
                for submesh in &mesh.submeshes {
                    stream.set_bind_group(
                        1,
                        Some(mesh.resolved_materials[submesh.material_index].main_bind_group),
                    );
                    stream.set_bind_group(2, Some(submesh.object_bind_group));
                    stream.draw(
                        submesh.first_index,
                        submesh.vertex_offset,
                        submesh.index_count,
                        1,
                        *bone_idx,
                    );
                }
            }
        }
        stream
    }

    fn full_screen_quad(
        &self,
        context: &FrameContext,
        pipeline: RasterPipelineHandle,
        bind_group: BindGroupHandle,
    ) -> DrawStream {
        let mut stream = DrawStream::new(bind_group, context.get_backbuffer_size().into(), 0);
        let vertices = [
            BasicVertex::new(vec3(-1.0, -1.0, 0.0), vec2(0.0, 0.0)),
            BasicVertex::new(vec3(1.0, -1.0, 0.0), vec2(1.0, 0.0)),
            BasicVertex::new(vec3(1.0, 1.0, 0.0), vec2(1.0, 1.0)),
            BasicVertex::new(vec3(-1.0, 1.0, 0.0), vec2(0.0, 1.0)),
        ];
        let indices = [0u16, 1u16, 2u16, 0u16, 3u16, 2u16];
        let vb = context.get_temp_buffer(&vertices).unwrap();
        let ib = context.get_temp_buffer(&indices).unwrap();
        stream.set_vertex_buffer(0, Some(vb));
        stream.set_index_buffer(Some(ib));
        stream.set_pipeline(pipeline);
        stream.draw(0, 0, 6, 1, 0);

        stream
    }

    fn draw_skybox(&self, context: &FrameContext, view: glam::Mat4) -> DrawStream {
        let mut stream = DrawStream::new(
            self.skybox_bind_group,
            context.get_backbuffer_size().into(),
            0,
        );
        let mut vertices = Vec::new();
        for index in FACES {
            vertices.push(CORNERS[index[0] as usize]);
            vertices.push(CORNERS[index[1] as usize]);
            vertices.push(CORNERS[index[2] as usize]);
            vertices.push(CORNERS[index[3] as usize]);
        }
        let mut indices = Vec::new();
        for face in FACES {
            indices.push(face[0]);
            indices.push(face[1]);
            indices.push(face[2]);
            indices.push(face[0]);
            indices.push(face[3]);
            indices.push(face[2]);
        }
        let vb = context.get_temp_buffer(&vertices).unwrap();
        let ib = context.get_temp_buffer(&indices).unwrap();
        let (_, rotation, translation) = view.inverse().to_scale_rotation_translation();
        let offset = context
            .get_temp_buffer_offset(
                &[glam::Mat4::from_rotation_translation(rotation, translation)
                    * glam::Mat4::from_scale(vec3(99.0, 99.0, 99.0))],
            )
            .unwrap();
        stream.set_pipeline(self.skybox_pipeline);
        stream.set_bind_group(1, None);
        stream.set_bind_group(2, None);
        stream.set_bind_group(3, Some(self.draw_bind_group));
        stream.set_dynamic_buffer_offset(0, Some(offset));
        stream.set_vertex_buffer(0, Some(vb));
        stream.set_vertex_buffer(1, None);
        stream.set_index_buffer(Some(ib));
        stream.draw(0, 0, indices.len() as _, 1, 0);

        stream
    }
}

impl<'a> Client for RenderDemo<'a> {
    fn tick(&mut self, _context: UpdateContext, dt: GameTime) -> dess_runner::ClientState {
        self.rotation += 0.25 * dt.delta_time;
        dess_runner::ClientState::Continue
    }

    fn render(&self, context: &FrameContext) -> Result<(), dess_backend::BackendError> {
        let color = context
            .get_temporary_image(
                Format::RGBA16_SFLOAT,
                ImageUsage::ColorTarget | ImageUsage::Sampled,
                TemporaryImageDims::Backbuffer,
            )
            .unwrap();
        let depth = context
            .get_temporary_image(
                Format::D24,
                ImageUsage::DepthStencilTarget,
                TemporaryImageDims::Backbuffer,
            )
            .unwrap();
        let color_target = RenderTarget::new(color.as_color().unwrap())
            .clear(ClearRenderTarget::Color([0.0, 0.0, 0.0, 1.0]));
        let depth_target = RenderTarget::new(depth.as_depth().unwrap())
            .clear(ClearRenderTarget::DepthStencil(1.0, 0));
        let eye_position = vec3(0.0, 0.15, 0.6);
        let view = glam::Mat4::look_at_rh(eye_position, vec3(0.0, 0.0, 0.0), glam::Vec3::Y);
        context
            .with_bind_groups(|ctx| {
                let scene = SceneUniform {
                    view,
                    projection: glam::Mat4::perspective_rh(
                        PI / 4.0,
                        context.get_backbuffer_aspect_ratio(),
                        0.1,
                        100.0,
                    ),
                    eye_position,
                };
                ctx.bind_uniform(self.scene_bind_group, 0, &scene)?;
                let light = LightUniform {
                    main: DirectionalLight {
                        direction: vec3a(0.0, 1.0, -4.0).normalize(),
                        color: vec3a(10.0, 10.0, 20.0),
                    },
                    fill: DirectionalLight {
                        direction: vec3a(1.0, 4.0, 0.0).normalize(),
                        color: vec3a(15.0, 10.0, 10.0),
                    },
                    back: DirectionalLight {
                        direction: vec3a(-1.0, 1.0, 1.0).normalize(),
                        color: vec3a(5.0, 5.0, 5.0),
                    },
                    ambient: AmbientLight {
                        top: vec3a(5.0, 5.0, 10.0),
                        middle: vec3a(1.0, 1.0, 1.0),
                        bottom: vec3a(8.0, 2.0, 2.0),
                    },
                };
                ctx.bind_uniform(self.scene_bind_group, 1, &light)?;

                ctx.bind_image(
                    self.tonemapping_bind_group,
                    0,
                    color.as_handle(),
                    ImageLayout::ShaderRead,
                )?;
                ctx.bind_uniform(
                    self.tonemapping_bind_group,
                    1,
                    &Tonemapping { expouse: 0.05 },
                )?;

                ctx.bind_uniform(self.skybox_bind_group, 0, &scene)?;
                ctx.bind_uniform(self.skybox_bind_group, 1, &light)?;

                Ok(())
            })
            .unwrap();
        let main_stream = self.create_draw_stream(context, self.rotation);
        let skybox = self.draw_skybox(context, view);
        let tonemapping =
            self.full_screen_quad(context, self.tonemapping, self.tonemapping_bind_group);
        context.execute(
            self.main_pass,
            &[color_target],
            Some(depth_target),
            context.get_backbuffer_size(),
            [main_stream, skybox].into_iter(),
        );

        let backbuffer = context.get_backbuffer();

        context.execute(
            self.tonemapping_pass,
            &[backbuffer],
            None,
            context.get_backbuffer_size(),
            [tonemapping].into_iter(),
        );

        Ok(())
    }

    fn hidden(&mut self, _value: bool) {}

    fn init(&mut self, context: InitContext) {
        let program = context
            .resource_manager
            .get_or_load_program(&[
                ShaderSource::vertex("shaders/pbr_vs.hlsl"),
                ShaderSource::fragment("shaders/pbr_ps.hlsl"),
            ])
            .unwrap();

        self.main_pass = context
            .render_device
            .create_render_pass(RenderPassLayout {
                depth_target: Some(
                    RenderTargetDesc::new(Format::D24)
                        .clear_input()
                        .initial_layout(ImageLayout::None),
                ),
                color_targets: &[RenderTargetDesc::new(Format::RGBA16_SFLOAT)
                    .clear_input()
                    .store_output()
                    .initial_layout(ImageLayout::None)
                    .next_layout(ImageLayout::ShaderRead)],
                subpasses: &[SubpassLayout {
                    depth_write: true,
                    depth_read: false,
                    color_writes: &[0],
                    color_reads: &[],
                }],
            })
            .unwrap();

        self.pipeline = context
            .pipeline_cache
            .get_or_register_raster_pipeline(
                RasterPipelineCreateDesc::new(
                    program,
                    self.main_pass,
                    &DRAW_PIPELINE_LAYOUT,
                    &BASIC_MESH_LAYOUT,
                )
                .depth_test(DepthCompareOp::Less)
                .cull(dess_backend::CullMode::Back)
                .depth_write(),
            )
            .unwrap();
        let model = context
            .resource_manager
            .request_model(GltfSource::new("models/ABeautifulGame/ABeautifulGame.gltf").get_ref());
        context
            .resource_manager
            .with_context(|ctx| {
                self.model = ctx.resolve_model(model)?;
                Ok(())
            })
            .unwrap();
        self.scene_bind_group = context
            .render_device
            .create_bind_group(&MAIN_PASS_BIND_LAYOUT)
            .unwrap();
        self.draw_bind_group = context
            .render_device
            .create_bind_group(&DRAW_CALL_BIND_LAYOUT)
            .unwrap();

        let program = context
            .resource_manager
            .get_or_load_program(&[
                ShaderSource::vertex("shaders/screen_quad_vs.hlsl"),
                ShaderSource::fragment("shaders/tonemapping_ps.hlsl"),
            ])
            .unwrap();
        self.tonemapping_pass = context
            .render_device
            .create_render_pass(RenderPassLayout {
                depth_target: None,
                color_targets: &[RenderTargetDesc::new(Format::BGRA8_UNORM)
                    .store_output()
                    .initial_layout(ImageLayout::None)
                    .next_layout(ImageLayout::Present)],
                subpasses: &[SubpassLayout {
                    depth_write: false,
                    depth_read: false,
                    color_writes: &[0],
                    color_reads: &[],
                }],
            })
            .unwrap();
        self.tonemapping = context
            .pipeline_cache
            .get_or_register_raster_pipeline(RasterPipelineCreateDesc::new(
                program,
                self.tonemapping_pass,
                &POSTPORCESSING_PIPELINE_LAYOUT,
                &BASIC_VERTEX_LAYOUT,
            ))
            .unwrap();
        self.tonemapping_bind_group = context
            .render_device
            .create_bind_group(&TONEMAPPING_PASS_BIND_LAYOUT)
            .unwrap();

        let program = context
            .resource_manager
            .get_or_load_program(&[
                ShaderSource::vertex("shaders/sky_vs.hlsl"),
                ShaderSource::fragment("shaders/sky_ps.hlsl"),
            ])
            .unwrap();
        self.skybox_pipeline = context
            .pipeline_cache
            .get_or_register_raster_pipeline(
                RasterPipelineCreateDesc::new(
                    program,
                    self.main_pass,
                    &SKYBOX_PIPELINE_LAYOUT,
                    &SKY_VERTEX_LAYOUT,
                )
                .depth_test(DepthCompareOp::Less),
            )
            .unwrap();

        self.skybox_bind_group = context
            .render_device
            .create_bind_group(&SKYBOX_BIND_LAYOUT)
            .unwrap();

        context
            .render_device
            .with_bind_groups(|ctx| {
                ctx.bind_dynamic_storage_buffer(
                    self.draw_bind_group,
                    0,
                    context.render_device.temp_buffer(),
                    mem::size_of::<glam::Mat4>() * MAX_MATRICES_PER_DRAW,
                )?;

                ctx.bind_dynamic_storage_buffer(
                    self.skybox_bind_group,
                    0,
                    context.render_device.temp_buffer(),
                    mem::size_of::<glam::Mat4>(),
                )?;

                Ok(())
            })
            .unwrap();
    }
}

fn main() {
    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    puffin::set_scopes_on(true);
    let mut runner = Runner::new(RenderDemo::default(), "Dess Engine - Test demo");
    runner.run();
}
