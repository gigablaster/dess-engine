use std::{slice, sync::atomic::Ordering};

use arrayvec::ArrayVec;
use ash::vk;
use log::{debug, error};

use crate::{BackendError, BackendResult};

use super::{Device, Index, PipelineHandle, ProgramHandle, Shader};

#[derive(Debug, Clone, Copy)]
pub struct BlendDesc {
    pub src_blend: vk::BlendFactor,
    pub dst_blend: vk::BlendFactor,
    pub op: vk::BlendOp,
}

#[derive(Debug)]
pub struct VertexAttributeDesc {
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub stride: usize,
}

/// Data to create pipeline.
///
/// Contains all data to create new pipeline.
#[derive(Debug, Clone)]
pub struct PipelineCreateDesc {
    /// Blend data, None if opaque. Order: color, alpha
    pub blend: Option<(BlendDesc, BlendDesc)>,
    /// Depth comparison op, None if no depth test is happening
    pub depth_test: Option<vk::CompareOp>,
    /// true if we write into depth
    pub depth_write: bool,
    /// Culling information, None if we don't do culling
    pub cull: Option<(vk::CullModeFlags, vk::FrontFace)>,
    /// Vertex layout
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub strides: &'static [(usize, vk::VertexInputRate)],
}

pub trait PipelineVertex: Sized {
    fn attributes() -> &'static [vk::VertexInputAttributeDescription];
    fn strides() -> &'static [(usize, vk::VertexInputRate)];
}

impl PipelineCreateDesc {
    pub fn new<T: PipelineVertex>() -> Self {
        Self {
            blend: None,
            depth_test: Some(vk::CompareOp::LESS),
            depth_write: true,
            cull: Some((vk::CullModeFlags::BACK, vk::FrontFace::CLOCKWISE)),
            attributes: T::attributes(),
            strides: T::strides(),
        }
    }

    pub fn blend(mut self, color: BlendDesc, alpha: BlendDesc) -> Self {
        self.blend = Some((color, alpha));

        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);

        self
    }

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;

        self
    }

    pub fn cull(mut self, mode: vk::CullModeFlags, front: vk::FrontFace) -> Self {
        self.cull = Some((mode, front));

        self
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PipelineCreatePayload {
    pub shaders: ArrayVec<Shader, 2>,
    pub layout: vk::PipelineLayout,
    pub desc: PipelineCreateDesc,
    pub color_attachments: ArrayVec<vk::Format, 8>,
    pub depth_attachment: Option<vk::Format>,
}

impl Device {
    pub fn create_pipeline(
        &self,
        program: ProgramHandle,
        desc: PipelineCreateDesc,
        color_attachments: &[vk::Format],
        depth_attachment: Option<vk::Format>,
    ) -> BackendResult<PipelineHandle> {
        let index = {
            let mut pipelines = self.pipelines.write();
            let index = pipelines.len();
            pipelines.push((vk::Pipeline::null(), vk::PipelineLayout::null()));

            Index::new(index)
        };
        {
            let programs = self.program_storage.read();
            let program = programs
                .get(program.index())
                .ok_or(BackendError::InvalidHandle)?;
            let payload = PipelineCreatePayload {
                shaders: program.shaders.clone(),
                layout: program.pipeline_layout,
                desc,
                color_attachments: color_attachments.iter().copied().collect(),
                depth_attachment,
            };
            let sender = self.pipeline_compiled_sender.clone();
            let device = self.raw.clone();
            let cache = self.pipeline_cache;
            self.pipelines_in_fly.fetch_add(1, Ordering::SeqCst);
            rayon::spawn(
                move || match Self::build_pipeline(&device, cache, payload) {
                    Ok(result) => sender.send((index, result.0, result.1)).unwrap(),
                    Err(err) => {
                        error!("Failed to compiled pipeline: {:?}", err);
                        sender
                            .send((index, vk::Pipeline::null(), vk::PipelineLayout::null()))
                            .unwrap();
                    }
                },
            );
        }

        Ok(index)
    }

    fn build_pipeline(
        device: &ash::Device,
        cache: vk::PipelineCache,
        desc: PipelineCreatePayload,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), BackendError> {
        debug!("Compile pipeline {:?}", desc);
        let shader_create_info = desc
            .shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(shader.stage)
                    .module(shader.raw)
                    .name(&shader.entry)
                    .build()
            })
            .collect::<Vec<_>>();

        let assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states)
            .build();

        let vertex_binding_desc = desc
            .desc
            .strides
            .iter()
            .enumerate()
            .map(|(index, _)| {
                vk::VertexInputBindingDescription::builder()
                    .stride(desc.desc.strides[index].0 as _)
                    .binding(desc.desc.attributes[index].binding)
                    .input_rate(desc.desc.strides[index].1)
                    .build()
            })
            .collect::<Vec<_>>();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_desc)
            .vertex_attribute_descriptions(desc.desc.attributes)
            .build();

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(slice::from_ref(&vk::Viewport::default()))
            .scissors(slice::from_ref(&vk::Rect2D::default()))
            .build();

        let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_bias_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let rasterizer_state = if let Some(desc) = desc.desc.cull {
            rasterizer_state.cull_mode(desc.0).front_face(desc.1)
        } else {
            rasterizer_state.cull_mode(vk::CullModeFlags::NONE)
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let depthstencil_state = if let Some(op) = desc.desc.depth_test {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(op)
                .stencil_test_enable(false)
                .depth_test_enable(true)
                .depth_write_enable(desc.desc.depth_write)
                .build()
        } else {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::NEVER)
                .stencil_test_enable(false)
                .depth_test_enable(false)
                .depth_write_enable(desc.desc.depth_write)
                .build()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachment = if let Some((color, alpha)) = desc.desc.blend {
            color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(color.src_blend)
                .dst_color_blend_factor(color.dst_blend)
                .color_blend_op(color.op)
                .src_alpha_blend_factor(alpha.src_blend)
                .dst_alpha_blend_factor(alpha.dst_blend)
                .alpha_blend_op(alpha.op)
        } else {
            color_blend_attachment.blend_enable(false)
        }
        .build();
        let blending_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(slice::from_ref(&color_blend_attachment))
            .logic_op_enable(false)
            .build();

        let rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&desc.color_attachments);
        let mut rendering_info = if let Some(depth) = desc.depth_attachment {
            rendering_info.depth_attachment_format(depth)
        } else {
            rendering_info
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(desc.layout)
            .stages(&shader_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&blending_state)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly_state_create_info)
            .rasterization_state(&rasterizer_state)
            .depth_stencil_state(&depthstencil_state)
            .push_next(&mut rendering_info)
            .build();

        let pipeline = unsafe {
            device.create_graphics_pipelines(cache, slice::from_ref(&pipeline_create_info), None)
        }
        .map_err(|(_, error)| BackendError::from(error))?[0];

        Ok((pipeline, desc.layout))
    }

    pub fn are_pipelines_compiling(&self) -> bool {
        self.pipelines_in_fly.load(Ordering::Acquire) > 0
    }
}
