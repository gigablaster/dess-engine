// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{collections::HashMap, fmt::Debug, hash::Hash, slice, sync::Arc};

use crate::BackendResult;

use super::{pipeline_cache::PipelineCache, Device, ImageDesc, Program};
use arrayvec::ArrayVec;
use ash::vk::{self};
use dess_common::TempList;
use log::error;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Default)]
pub struct RenderPassLayout<'a> {
    pub color_attachments: &'a [vk::Format],
    pub depth_attachment: Option<vk::Format>,
}

impl<'a> RenderPassLayout<'a> {
    pub fn new(color_attachments: &'a [vk::Format], depth_attachment: Option<vk::Format>) -> Self {
        Self {
            color_attachments,
            depth_attachment,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderPassAttachmentDesc {
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    samples: vk::SampleCountFlags,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

impl RenderPassAttachmentDesc {
    pub fn color(format: vk::Format) -> Self {
        Self {
            format,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    pub fn depth(format: vk::Format) -> Self {
        Self {
            format,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }

    pub fn garbage_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    pub fn clear_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self
    }

    pub fn discard_output(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn multisampling(mut self, value: vk::SampleCountFlags) -> Self {
        self.samples = value;
        self
    }

    pub fn initial_layout(mut self, value: vk::ImageLayout) -> Self {
        self.initial_layout = value;
        self
    }

    pub fn final_layout(mut self, value: vk::ImageLayout) -> Self {
        self.final_layout = value;
        self
    }

    fn build(self) -> vk::AttachmentDescription {
        vk::AttachmentDescription {
            format: self.format,
            samples: self.samples,
            load_op: self.load_op,
            store_op: self.store_op,
            initial_layout: self.initial_layout,
            final_layout: self.final_layout,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            ..Default::default()
        }
    }
}

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
    /// Shader program
    pub program: Arc<Program>,
    /// Subpass in current render pass
    pub subpass: u32,
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
    pub fn new<T: PipelineVertex>(program: &Arc<Program>, subpass: u32) -> Self {
        Self {
            program: program.clone(),
            subpass,
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

/// Pipeline cache builder
///
/// Collect and builds all pipelines using rayon. Only way to create new pipelines,
/// we can push multiple of them into RenderPass, but new pipelines ones will override old
/// ones. Best way is to collect all shaders that should be used on for each render pass
/// and build them all at once.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    pipelines: Vec<PipelineCreateDesc>,
}

impl PipelineBuilder {
    fn build(
        self,
        pipeline_cache: &PipelineCache,
    ) -> BackendResult<Vec<(vk::Pipeline, vk::PipelineLayout)>> {
        let cache = RwLock::new(vec![
            (vk::Pipeline::null(), vk::PipelineLayout::null());
            self.pipelines.len()
        ]);
        rayon::scope(|s| {
            for index in 0..self.pipelines.len() {
                let cache = &cache;
                let pipelines = &self.pipelines;
                s.spawn(move |_| {
                    if let Err(err) = Self::build_pipeline(
                        render_pass,
                        pipeline_cache.raw(),
                        &pipelines[index],
                        cache,
                        index,
                    ) {
                        error!("Failed to compile pipeline: {:?}", err);
                    }
                })
            }
        });

        let cache = cache.into_inner();
        if cache.iter().any(|x| x.0 == vk::Pipeline::null()) {
            Err(BackendError::Fail)
        } else {
            Ok(cache)
        }
    }

    fn build_pipeline(
        render_pass: vk::RenderPass,
        cache: vk::PipelineCache,
        desc: &PipelineCreateDesc,
        pipelines: &RwLock<Vec<(vk::Pipeline, vk::PipelineLayout)>>,
        index: usize,
    ) -> Result<(), BackendError> {
        let shader_create_info = desc
            .program
            .shaders()
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
            .strides
            .iter()
            .enumerate()
            .map(|(index, _)| {
                vk::VertexInputBindingDescription::builder()
                    .stride(desc.strides[index].0 as _)
                    .binding(desc.attributes[index].binding)
                    .input_rate(desc.strides[index].1)
                    .build()
            })
            .collect::<Vec<_>>();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_desc)
            .vertex_attribute_descriptions(desc.attributes)
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

        let rasterizer_state = if let Some(desc) = desc.cull {
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

        let depthstencil_state = if let Some(op) = desc.depth_test {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(op)
                .stencil_test_enable(false)
                .depth_test_enable(true)
                .depth_write_enable(desc.depth_write)
                .build()
        } else {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::NEVER)
                .stencil_test_enable(false)
                .depth_test_enable(false)
                .depth_write_enable(desc.depth_write)
                .build()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachment = if let Some((color, alpha)) = desc.blend {
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

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .render_pass(render_pass)
            .layout(desc.program.pipeline_layout())
            .stages(&shader_create_info)
            .subpass(desc.subpass)
            .dynamic_state(&dynamic_state_create_info)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&blending_state)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly_state_create_info)
            .rasterization_state(&rasterizer_state)
            .depth_stencil_state(&depthstencil_state)
            .build();

        let pipeline = unsafe {
            desc.program.device().raw().create_graphics_pipelines(
                cache,
                slice::from_ref(&pipeline_create_info),
                None,
            )
        }
        .map_err(|(_, error)| BackendError::from(error))?[0];

        pipelines.write()[index] = (pipeline, desc.program.pipeline_layout());

        Ok(())
    }

    /// Adds new pipeline and returns it's index so we can get it later from render pass
    /// itself. Returns handle to be used with RenderPass::resolve.
    pub fn push(&mut self, desc: PipelineCreateDesc) -> Index<vk::Pipeline> {
        let last = self.pipelines.len() as u32;
        self.pipelines.push(desc);

        Index::<vk::Pipeline>::new(last)
    }
}

#[derive(Debug)]
/// Describes render pass
pub struct RenderPass {
    device: Arc<Device>,
    raw: vk::RenderPass,
    fbo_cache: FboCache,
    pipeline_cache: Vec<(vk::Pipeline, vk::PipelineLayout)>,
}

/// Render pass
///
/// Keeps all information that is needed for render pass. Including all FBOs and all pipelines.
/// FBOs can be cleaned once render target has changed.
impl RenderPass {
    /// Creates render pass from device, description and possible pipelines
    ///
    /// Only way to create pipelines in this engine, render passes should be created
    /// at start of the game.
    pub fn new(
        device: &Arc<Device>,
        layout: RenderPassLayout,
        pipelines: PipelineBuilder,
        cache: &PipelineCache,
    ) -> Result<Self, BackendError> {
        let attachments = layout
            .color_attachments
            .iter()
            .map(|desc| desc.build())
            .chain(layout.depth_attachment.iter().map(|desc| desc.build()))
            .collect::<Vec<_>>();

        let color_attacmnet_refs = (0..layout.color_attachments.len())
            .map(|index| vk::AttachmentReference {
                attachment: index as _,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();

        let mut subpass_desc = vk::SubpassDescription::builder()
            .color_attachments(&color_attacmnet_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let depth_attachment_ref = if layout.depth_attachment.is_some() {
            Some(vk::AttachmentReference {
                attachment: color_attacmnet_refs.len() as _,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            })
        } else {
            None
        };

        if let Some(depth_attachment_ref) = depth_attachment_ref.as_ref() {
            subpass_desc = subpass_desc.depth_stencil_attachment(depth_attachment_ref);
        }

        let subpasses = [subpass_desc.build(); 1];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .build();

        let render_pass = unsafe { device.raw().create_render_pass(&render_pass_info, None) }?;

        let all_attachments = layout
            .color_attachments
            .iter()
            .chain(layout.depth_attachment)
            .copied()
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        Ok(Self {
            device: device.clone(),
            raw: render_pass,
            fbo_cache: FboCache::new(render_pass, &all_attachments),
            pipeline_cache: pipelines.build(render_pass, cache)?,
        })
    }

    /// Removes all FBOs, so new one will be created
    pub fn clear_fbos(&self) {
        self.fbo_cache.clear(&self.device);
    }

    pub fn raw(&self) -> vk::RenderPass {
        self.raw
    }

    /// Creates new FBO, or returns already created with same key.
    pub fn get_or_create_fbo(&self, key: FboCacheKey) -> Result<vk::Framebuffer, BackendError> {
        self.fbo_cache.get_or_create(&self.device, key)
    }

    /// returns vulkan pipeline by handle. Handles are created by PipelineCacheBuilder.
    /// Doesn't do any valudation if handle belongs to this render pass.
    pub fn resolve_pipeline(
        &self,
        index: Index<(vk::Pipeline, vk::PipelineLayout)>,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        self.pipeline_cache[index.value() as usize]
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        self.clear_fbos();
        self.pipeline_cache
            .drain(..)
            .for_each(|pipeline| unsafe { self.device.raw().destroy_pipeline(pipeline.0, None) });
        unsafe { self.device.raw().destroy_render_pass(self.raw, None) };
    }
}
