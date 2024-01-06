use std::{f32::consts::PI, marker::PhantomData, mem, sync::Arc};

use dess_assets::{ContentSource, GltfSource, ShaderSource};
use dess_backend::{
    BindGroupLayoutDesc, BindType, BindingDesc, ClearRenderTarget, DepthCompareOp, DrawStream,
    Format, Image, ImageAspect, ImageBarrier, ImageLayout, ImageUsage, RasterPipelineCreateDesc,
    RasterPipelineHandle, RenderPassLayout, ShaderStage, EMPTY_BIND_LAYOUT,
    {BindGroupHandle, RenderTarget},
};
use dess_common::GameTime;
use dess_engine::{
    render::{BasicVertex, BASIC_MESH_LAYOUT, BASIC_VERTEX_LAYOUT},
    ModelCollection, PoolImageDesc, RelativeImageSize, ResourceLoader, MESH_PBR_MATERIAL_LAYOUT,
    PACKED_MESH_OBJECT_LAYOUT,
};
use dess_runner::{Client, InitContext, RenderContext, Runner, UpdateContext};
use glam::{vec2, vec3, vec3a, Quat, Vec3};

const MAX_MATRICES_PER_DRAW: usize = 256;

#[derive(Default)]
struct RenderDemo<'a> {
    model: Arc<ModelCollection>,
    pipeline: RasterPipelineHandle,
    tonemapping: RasterPipelineHandle,
    scene_bind_group: BindGroupHandle,
    draw_bind_group: BindGroupHandle,
    tonemapping_bind_group: BindGroupHandle,
    rotation: f32,
    _phantom: PhantomData<&'a ()>,
}

static MAIN_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color_attachments: &[Format::RGBA16_SFLOAT],
    depth_attachment: Some(Format::D24),
};

static TONEMAPPING_PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color_attachments: &[Format::BGRA8_UNORM],
    depth_attachment: None,
};

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

impl<'a> RenderDemo<'a> {
    fn create_draw_stream(&self, context: &RenderContext<'a>, rotation: f32) -> DrawStream {
        puffin::profile_function!();
        let mut stream = DrawStream::new(self.scene_bind_group);
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
            let temp = context.frame.temp_allocate(&bones).unwrap();
            stream.set_dynamic_buffer_offset(0, Some(temp.offset));
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
        context: &RenderContext,
        pipeline: RasterPipelineHandle,
        bind_group: BindGroupHandle,
    ) -> DrawStream {
        let mut stream = DrawStream::new(bind_group);
        let vertices = [
            BasicVertex::new(vec3(-1.0, -1.0, 0.0), vec2(0.0, 0.0)),
            BasicVertex::new(vec3(1.0, -1.0, 0.0), vec2(1.0, 0.0)),
            BasicVertex::new(vec3(1.0, 1.0, 0.0), vec2(1.0, 1.0)),
            BasicVertex::new(vec3(-1.0, 1.0, 0.0), vec2(0.0, 1.0)),
        ];
        let indices = [0u16, 1u16, 2u16, 0u16, 3u16, 2u16];
        let vb = context.frame.temp_allocate(&vertices).unwrap();
        let ib = context.frame.temp_allocate(&indices).unwrap();
        stream.set_vertex_buffer(0, Some(vb));
        stream.set_index_buffer(Some(ib));
        stream.set_pipeline(pipeline);
        stream.draw(0, 0, 6, 1, 0);

        stream
    }
}
impl<'a> Client for RenderDemo<'a> {
    fn tick(&mut self, _context: UpdateContext, dt: GameTime) -> dess_runner::ClientState {
        self.rotation += 0.25 * dt.delta_time;
        dess_runner::ClientState::Continue
    }

    fn render(&self, context: RenderContext) -> Result<(), dess_backend::BackendError> {
        let color = context
            .get_temporary_render_target(
                Format::RGBA16_SFLOAT,
                ImageUsage::ColorTarget | ImageUsage::Sampled,
                ImageAspect::Color,
                RelativeImageSize::Backbuffer,
            )
            .unwrap();
        let depth = context
            .get_temporary_render_target(
                Format::D24,
                ImageUsage::DepthStencilTarget,
                ImageAspect::Depth,
                RelativeImageSize::Backbuffer,
            )
            .unwrap();
        let color_target = RenderTarget::new(
            Format::RGBA16_SFLOAT,
            color.view(),
            ImageLayout::ColorTarget,
        )
        .store_output()
        .clear_input(ClearRenderTarget::Color([0.0, 0.0, 0.0, 1.0]));
        let depth_target =
            RenderTarget::new(Format::D24, depth.view(), ImageLayout::DepthStencilTarget)
                .clear_input(ClearRenderTarget::DepthStencil(1.0, 0));
        let eye_position = vec3(0.0, 0.3, 1.8);
        context
            .device
            .with_bind_groups(|ctx| {
                let scene = SceneUniform {
                    view: glam::Mat4::look_at_rh(eye_position, vec3(0.0, 0.0, 0.0), -glam::Vec3::Y),
                    projection: glam::Mat4::perspective_rh(
                        PI / 4.0,
                        context.frame.render_area.aspect_ratio(),
                        0.1,
                        100.0,
                    ),
                    eye_position,
                };
                ctx.bind_uniform(self.scene_bind_group, 0, &scene)?;
                let light = LightUniform {
                    main: DirectionalLight {
                        direction: vec3a(0.0, 1.0, 1.0).normalize(),
                        color: vec3a(0.8, 0.8, 1.0),
                    },
                    fill: DirectionalLight {
                        direction: vec3a(1.0, 1.0, 0.0).normalize(),
                        color: vec3a(0.6, 0.5, 0.5),
                    },
                    back: DirectionalLight {
                        direction: vec3a(-1.0, 1.0, -1.0).normalize(),
                        color: vec3a(0.3, 0.3, 0.3),
                    },
                    ambient: AmbientLight {
                        top: vec3a(0.7, 0.7, 0.9),
                        middle: vec3a(0.4, 0.6, 0.4),
                        bottom: vec3a(0.5, 0.4, 0.3),
                    },
                };
                ctx.bind_uniform(self.scene_bind_group, 1, &light)?;

                ctx.bind_image(
                    self.tonemapping_bind_group,
                    0,
                    color.image(),
                    ImageLayout::ShaderRead,
                )?;

                Ok(())
            })
            .unwrap();
        let main_stream = self.create_draw_stream(&context, self.rotation);
        let tonemapping =
            self.full_screen_quad(&context, self.tonemapping, self.tonemapping_bind_group);
        context.frame.execute(
            context.frame.render_area,
            &[color_target],
            Some(depth_target),
            [main_stream].into_iter(),
            &[ImageBarrier::color_to_attachment(color.image())],
        );

        let backbuffer = RenderTarget::new(
            Format::BGRA8_UNORM,
            context.frame.target_view,
            ImageLayout::ColorTarget,
        )
        .store_output();

        context.frame.execute(
            context.frame.render_area,
            &[backbuffer],
            None,
            [tonemapping].into_iter(),
            &[ImageBarrier::color_attachment_to_sampled(color.image())],
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

        self.pipeline = context
            .pipeline_cache
            .get_or_register_raster_pipeline(
                RasterPipelineCreateDesc::new(
                    program,
                    &MAIN_PASS_LAYOUT,
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
            .request_model(GltfSource::new("models/PBR/gun.gltf").get_ref());
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
        context
            .render_device
            .with_bind_groups(|ctx| {
                ctx.bind_dynamic_storage_buffer(
                    self.draw_bind_group,
                    0,
                    context.render_device.temp_buffer(),
                    mem::size_of::<glam::Mat4>() * MAX_MATRICES_PER_DRAW,
                )
            })
            .unwrap();

        let program = context
            .resource_manager
            .get_or_load_program(&[
                ShaderSource::vertex("shaders/screen_quad_vs.hlsl"),
                ShaderSource::fragment("shaders/tonemapping_ps.hlsl"),
            ])
            .unwrap();
        self.tonemapping = context
            .pipeline_cache
            .get_or_register_raster_pipeline(RasterPipelineCreateDesc::new(
                program,
                &TONEMAPPING_PASS_LAYOUT,
                &POSTPORCESSING_PIPELINE_LAYOUT,
                &BASIC_VERTEX_LAYOUT,
            ))
            .unwrap();
        self.tonemapping_bind_group = context
            .render_device
            .create_bind_group(&TONEMAPPING_PASS_BIND_LAYOUT)
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
