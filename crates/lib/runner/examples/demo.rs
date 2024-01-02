use std::{f32::consts::PI, marker::PhantomData, mem, sync::Arc};

use dess_assets::{ContentSource, GltfSource, ShaderSource};
use dess_backend::{
    BindGroupLayoutDesc, BindType, BindingDesc, ClearRenderTarget, DepthCompareOp, DrawStream,
    Format, ImageAspect, ImageLayout, ImageUsage, RasterPipelineCreateDesc, RasterPipelineHandle,
    RenderPassLayout, ShaderStage, {BindGroupHandle, RenderTarget},
};
use dess_common::GameTime;
use dess_engine::{
    render::BASIC_MESH_LAYOUT, ModelCollection, PoolImageDesc, RelativeImageSize, ResourceLoader,
    MESH_PBR_MATERIAL_LAYOUT, PACKED_MESH_OBJECT_LAYOUT,
};
use dess_runner::{Client, InitContext, RenderContext, Runner, UpdateContext};
use glam::vec3;

#[derive(Default)]
struct ClearBackbuffer<'a> {
    model: Arc<ModelCollection>,
    pipeline: RasterPipelineHandle,
    scene_bind_group: BindGroupHandle,
    draw_bind_group: BindGroupHandle,
    _phantom: PhantomData<&'a ()>,
}

static PASS_LAYOUT: RenderPassLayout = RenderPassLayout {
    color_attachments: &[Format::BGRA8_UNORM],
    depth_attachment: Some(Format::D24),
};

const PASS_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
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
            name: "base_sampler",
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
        ty: BindType::DynamicUniformBuffer,
        count: 1,
    }],
};

const DRAW_PIPELINE_LAYOUT: [BindGroupLayoutDesc; 4] = [
    PASS_BIND_LAYOUT,
    MESH_PBR_MATERIAL_LAYOUT,
    PACKED_MESH_OBJECT_LAYOUT,
    DRAW_CALL_BIND_LAYOUT,
];

#[repr(C, align(16))]
struct SceneUniform {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

impl<'a> ClearBackbuffer<'a> {
    fn create_draw_stream(&self, context: &RenderContext<'a>, rotation: f32) -> DrawStream {
        puffin::profile_function!();
        let mut stream = DrawStream::new(self.scene_bind_group);
        stream.set_pipeline(self.pipeline);
        for model in self.model.models.values() {
            let mut bones = Vec::with_capacity(model.bones.len());
            for (index, bone) in model.bones.iter().enumerate() {
                let parent = model.bone_parents[index];
                if parent == u32::MAX {
                    bones.push(glam::Mat4::from_rotation_z(rotation) * glam::Mat4::from(*bone));
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
}
impl<'a> Client for ClearBackbuffer<'a> {
    fn tick(&mut self, _context: UpdateContext, _dt: GameTime) -> dess_runner::ClientState {
        dess_runner::ClientState::Continue
    }

    fn render(&self, context: RenderContext) -> Result<(), dess_backend::BackendError> {
        let depth = context
            .resource_pool
            .temp_image(
                [
                    context.frame.render_area.width,
                    context.frame.render_area.height,
                ],
                PoolImageDesc {
                    format: Format::D24,
                    aspect: ImageAspect::Depth,
                    usage: ImageUsage::DepthStencilTarget,
                    resolution: RelativeImageSize::Backbuffer,
                },
            )
            .unwrap();
        let color_target = RenderTarget::new(
            Format::BGRA8_UNORM,
            context.frame.target_view,
            ImageLayout::ColorTarget,
        )
        .store_output()
        .clear_input(ClearRenderTarget::Color([0.125, 0.125, 0.125, 1.0]));
        let depth_target =
            RenderTarget::new(Format::D24, depth.view(), ImageLayout::DepthStencilTarget)
                .clear_input(ClearRenderTarget::DepthStencil(1.0, 0));
        context
            .device
            .with_bind_groups(|ctx| {
                let scene = SceneUniform {
                    view: glam::Mat4::look_at_rh(
                        vec3(0.0, 0.25, 0.2),
                        glam::Vec3::ZERO,
                        glam::Vec3::Z,
                    ),
                    projection: glam::Mat4::perspective_rh(
                        PI / 2.0,
                        context.frame.render_area.aspect_ratio(),
                        0.1,
                        100.0,
                    ),
                };
                ctx.bind_uniform(self.scene_bind_group, 0, &scene)?;

                Ok(())
            })
            .unwrap();
        let stream = self.create_draw_stream(&context, 0.0);
        context.frame.execute(
            context.frame.render_area,
            &[color_target],
            Some(depth_target),
            [stream].into_iter(),
            &[],
        );

        Ok(())
    }

    fn hidden(&mut self, _value: bool) {}

    fn init(&mut self, context: InitContext) {
        let program = context
            .resource_manager
            .get_or_load_program(&[
                ShaderSource::vertex("shaders/unlit_vs.hlsl"),
                ShaderSource::fragment("shaders/unlit_ps.hlsl"),
            ])
            .unwrap();

        self.pipeline = context
            .pipeline_cache
            .get_or_register_raster_pipeline(
                RasterPipelineCreateDesc::new(
                    program,
                    &PASS_LAYOUT,
                    &DRAW_PIPELINE_LAYOUT,
                    &BASIC_MESH_LAYOUT,
                )
                .depth_test(DepthCompareOp::LessOrEqual)
                .cull(dess_backend::CullMode::Back)
                .depth_write(),
            )
            .unwrap();
        let model = context
            .resource_manager
            .request_model(GltfSource::new("models/PBR/ABeautifulGame.gltf").get_ref());
        context
            .resource_manager
            .with_context(|ctx| {
                self.model = ctx.resolve_model(model)?;
                Ok(())
            })
            .unwrap();
        self.scene_bind_group = context
            .render_device
            .create_bind_group(&PASS_BIND_LAYOUT)
            .unwrap();
        self.draw_bind_group = context
            .render_device
            .create_bind_group(&DRAW_CALL_BIND_LAYOUT)
            .unwrap();
        context
            .render_device
            .with_bind_groups(|ctx| {
                ctx.bind_dynamic_uniform_buffer(
                    self.draw_bind_group,
                    0,
                    context.render_device.temp_buffer(),
                    mem::size_of::<glam::Mat4>(),
                )
            })
            .unwrap();
    }
}

fn main() {
    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    puffin::set_scopes_on(true);
    let mut runner = Runner::new(
        ClearBackbuffer::default(),
        "Dess Engine - Clear backbuffer example",
    );
    runner.run();
}
