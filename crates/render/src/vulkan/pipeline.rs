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

use std::{ffi::CStr, mem::size_of, slice};

use ash::vk;

use crate::{BackendError, BackendResult};

use super::{FreeGpuResource, RenderPass, Shader};

pub trait PipelineVertex: Sized {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription];
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

pub enum PipelineBlend {
    Opaque,
    AlphaBlend,
    PremultipliedAlpha,
    Additive,
}

pub struct PipelineDesc<'a> {
    pub render_pass: &'a RenderPass,
    pub subpass: u32,
    pub depth_write: bool,
    pub depth_test: bool,
    pub face_cull: bool,
    pub blend: PipelineBlend,
    pub shaders: Vec<&'a Shader>,
}

impl<'a> PipelineDesc<'a> {
    pub fn new(render_pass: &'a RenderPass) -> Self {
        Self {
            render_pass,
            subpass: 0,
            depth_write: true,
            depth_test: true,
            face_cull: true,
            blend: PipelineBlend::Opaque,
            shaders: Vec::new(),
        }
    }

    pub fn subpass(mut self, value: u32) -> Self {
        self.subpass = value;
        self
    }

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;
        self
    }

    pub fn face_cull(mut self, value: bool) -> Self {
        self.face_cull = value;
        self
    }

    pub fn add_shader(mut self, shader: &'a Shader) -> Self {
        self.shaders.push(shader);
        self
    }
}

pub fn create_pipeline_cache(device: &ash::Device) -> BackendResult<vk::PipelineCache> {
    let cache_create_info = vk::PipelineCacheCreateInfo::builder().build();

    let cache = unsafe { device.create_pipeline_cache(&cache_create_info, None) }?;

    Ok(cache)
}

impl Pipeline {
    pub fn new<V: PipelineVertex>(
        device: &ash::Device,
        cache: &vk::PipelineCache,
        desc: PipelineDesc,
    ) -> BackendResult<Self> {
        let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let shader_create_info = desc
            .shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(shader.stage)
                    .module(shader.raw)
                    .name(entry)
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

        let vertex_binding_desc = vk::VertexInputBindingDescription::builder()
            .stride(size_of::<V>() as _)
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(slice::from_ref(&vertex_binding_desc))
            .vertex_attribute_descriptions(V::attribute_description())
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

        let rasterizer_state = if desc.face_cull {
            rasterizer_state
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
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

        let depthstencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_compare_op(vk::CompareOp::LESS)
            .stencil_test_enable(false)
            .depth_test_enable(desc.depth_test)
            .depth_write_enable(desc.depth_write)
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachment = match desc.blend {
            PipelineBlend::Opaque => color_blend_attachment.blend_enable(false),
            PipelineBlend::AlphaBlend => color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD),
            PipelineBlend::PremultipliedAlpha => color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD),
            PipelineBlend::Additive => color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_COLOR)
                .dst_color_blend_factor(vk::BlendFactor::ONE)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD),
        }
        .build();
        let blending_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(slice::from_ref(&color_blend_attachment))
            .logic_op_enable(false)
            .build();

        let mut descriptor_layouts = Vec::new();
        desc.shaders
            .iter()
            .for_each(|shader| descriptor_layouts.extend_from_slice(&shader.layouts));

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .build();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }?;

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .render_pass(desc.render_pass.raw)
            .layout(pipeline_layout)
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
            device.create_graphics_pipelines(*cache, slice::from_ref(&pipeline_create_info), None)
        }
        .map_err(|_| BackendError::Other("Shit happened".into()))?[0];

        Ok(Self {
            pipeline_layout,
            pipeline,
        })
    }
}

impl FreeGpuResource for Pipeline {
    fn free(&self, devive: &ash::Device) {
        unsafe {
            devive.destroy_pipeline(self.pipeline, None);
            devive.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
