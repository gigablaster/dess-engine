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

use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    mem::size_of,
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use rspirv_reflect::{BindingCount, DescriptorInfo};

use super::{BackendError, BackendResult, Device, FreeGpuResource, RenderPass, SamplerDesc};

const MAX_SAMPLERS: usize = 16;

type DescriptorSetLayouts = Vec<vk::DescriptorSetLayout>;
type DescriptorSetIndex = (u32, u32);
type DescriptorSetMap = HashMap<String, DescriptorSetIndex>;

pub struct ShaderDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::VERTEX,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::FRAGMENT,
            entry: "main",
            code,
        }
    }

    pub fn entry(mut self, entry: &'a str) -> Self {
        self.entry = entry;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Shader {
    pub raw: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub layouts: DescriptorSetLayouts,
    pub names: DescriptorSetMap,
    pub entry: CString,
}

impl Shader {
    pub fn new(device: &Arc<Device>, desc: ShaderDesc, name: Option<&str>) -> BackendResult<Self> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(desc.code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.raw.create_shader_module(&shader_create_info, None) }?;
        if let Some(name) = name {
            device.set_object_name(shader, name)?;
        }
        let (names, layouts) = Self::create_descriptor_set_layouts(device, desc.stage, desc.code)?;
        Ok(Self {
            stage: desc.stage,
            raw: shader,
            layouts,
            names,
            entry: CString::new(desc.entry).unwrap(),
        })
    }

    fn create_descriptor_set_layouts(
        device: &Device,
        stage: vk::ShaderStageFlags,
        code: &[u8],
    ) -> BackendResult<(DescriptorSetMap, DescriptorSetLayouts)> {
        let info = rspirv_reflect::Reflection::new_from_spirv(code)?;
        let sets = info.get_descriptor_sets()?;
        let set_count = sets.keys().map(|index| *index + 1).max().unwrap_or(0);
        let mut layouts = DescriptorSetLayouts::with_capacity(8);
        let create_flags = vk::DescriptorSetLayoutCreateFlags::empty();
        let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
        let mut names = DescriptorSetMap::with_capacity(8);
        for set_index in 0..set_count {
            let set = sets.get(&set_index);
            if let Some(set) = set {
                let mut bindings = Vec::with_capacity(set.len());
                for (index, binding) in set.iter() {
                    match binding.ty {
                        rspirv_reflect::DescriptorType::UNIFORM_BUFFER
                        | rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | rspirv_reflect::DescriptorType::STORAGE_IMAGE
                        | rspirv_reflect::DescriptorType::STORAGE_BUFFER
                        | rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                            bindings.push(Self::create_uniform_binding(*index, stage, binding))
                        }
                        rspirv_reflect::DescriptorType::SAMPLED_IMAGE => bindings
                            .push(Self::create_sampled_image_binding(*index, stage, binding)),
                        rspirv_reflect::DescriptorType::SAMPLER
                        | rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                            let sampler_index = samplers.len();
                            samplers.push(
                                device
                                    .get_sampler(SamplerDesc {
                                        texel_filter: vk::Filter::LINEAR,
                                        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                                        address_mode: vk::SamplerAddressMode::REPEAT,
                                    })
                                    .unwrap(),
                            );
                            bindings.push(Self::create_sampler_binding(
                                *index,
                                stage,
                                binding,
                                &samplers[sampler_index],
                            ));
                        }
                        _ => unimplemented!("{:?}", binding),
                    };
                    names.insert(binding.name.clone(), (set_index, *index));
                }
                let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(create_flags)
                    .bindings(&bindings)
                    .build();
                let layout =
                    unsafe { device.raw.create_descriptor_set_layout(&layout_info, None) }?;
                layouts.push(layout);
            } else {
                let layout = unsafe {
                    device.raw.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder().build(),
                        None,
                    )
                }?;
                layouts.push(layout);
            }
        }

        Ok((names, layouts))
    }

    fn create_uniform_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_type = match binding.ty {
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER => vk::DescriptorType::UNIFORM_BUFFER,
            rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER
            }
            rspirv_reflect::DescriptorType::STORAGE_IMAGE => vk::DescriptorType::STORAGE_IMAGE,
            rspirv_reflect::DescriptorType::STORAGE_BUFFER => vk::DescriptorType::STORAGE_BUFFER,
            rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            }
            _ => unimplemented!("{:?}", binding),
        };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(index)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .stage_flags(stage)
            .build()
    }

    fn create_sampled_image_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_count = match binding.binding_count {
            BindingCount::One => 1,
            BindingCount::StaticSized(size) => size,
            _ => unimplemented!("{:?}", binding.binding_count),
        };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(index)
            .descriptor_count(descriptor_count as _)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(stage)
            .build()
    }

    fn create_sampler_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
        sampler: &vk::Sampler,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_count = match binding.binding_count {
            BindingCount::One => 1,
            BindingCount::StaticSized(size) => size,
            _ => unimplemented!("{:?}", binding.binding_count),
        };
        let ty = match binding.ty {
            rspirv_reflect::DescriptorType::SAMPLER => vk::DescriptorType::SAMPLER,
            rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
            _ => unimplemented!(),
        };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(index)
            .descriptor_count(descriptor_count as _)
            .descriptor_type(ty)
            .stage_flags(stage)
            .immutable_samplers(slice::from_ref(sampler))
            .build()
    }
}

impl FreeGpuResource for Shader {
    fn free(&self, device: &ash::Device) {
        self.layouts.iter().for_each(|layout| {
            unsafe { device.destroy_descriptor_set_layout(*layout, None) };
        });
        unsafe { device.destroy_shader_module(self.raw, None) }
    }
}

pub trait PipelineVertex: Sized {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription];
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub names: DescriptorSetMap,
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

        let descriptor_layouts = Self::combine_layouts(
            &desc
                .shaders
                .iter()
                .map(|shader| &shader.layouts)
                .collect::<Vec<_>>(),
        );

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

        let mut names = DescriptorSetMap::with_capacity(8);
        desc.shaders.iter().for_each(|shader| {
            shader.names.iter().for_each(|(name, index)| {
                names.insert(name.into(), *index);
            })
        });

        Ok(Self {
            pipeline_layout,
            pipeline,
            names,
        })
    }

    pub fn descriptor_index(&self, name: &str) -> Option<DescriptorSetIndex> {
        self.names.get(name).copied()
    }

    fn combine_layouts(layouts: &[&DescriptorSetLayouts]) -> DescriptorSetLayouts {
        let count = layouts.iter().map(|x| x.len()).max().unwrap_or(0);
        let mut result = DescriptorSetLayouts::with_capacity(count);
        result.resize(count, vk::DescriptorSetLayout::null());
        for layout in layouts {
            for index in 0..layout.len() {
                if layout[index] != vk::DescriptorSetLayout::null() {
                    result[index] = layout[index];
                }
            }
        }

        result
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
