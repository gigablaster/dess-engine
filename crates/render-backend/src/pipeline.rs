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
    collections::{btree_map::Entry, BTreeMap, HashMap},
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

type DescriptorSetLayout = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = BTreeMap<u32, DescriptorSetLayout>;

#[derive(Debug, Default, Clone)]
pub struct DescriptorSetInfo {
    pub types: HashMap<u32, vk::DescriptorType>,
    pub names: HashMap<String, u32>,
    pub layout: vk::DescriptorSetLayout,
}

impl DescriptorSetInfo {
    pub fn new(
        device: &Device,
        stage: vk::ShaderStageFlags,
        set: &BTreeMap<u32, DescriptorInfo>,
    ) -> BackendResult<Self> {
        let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
        let mut bindings = HashMap::with_capacity(set.len());
        for (index, binding) in set.iter() {
            match binding.ty {
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER
                | rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER
                | rspirv_reflect::DescriptorType::STORAGE_IMAGE
                | rspirv_reflect::DescriptorType::STORAGE_BUFFER
                | rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    bindings.insert(*index, Self::create_uniform_binding(*index, stage, binding));
                }
                rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                    bindings.insert(
                        *index,
                        Self::create_sampled_image_binding(*index, stage, binding),
                    );
                }
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
                    bindings.insert(
                        *index,
                        Self::create_sampler_binding(
                            *index,
                            stage,
                            binding,
                            &samplers[sampler_index],
                        ),
                    );
                }
                _ => unimplemented!("{:?}", binding),
            };
        }

        let layoyt = bindings.values().copied().collect::<Vec<_>>();
        let mut types = HashMap::with_capacity(set.len());
        bindings.into_iter().for_each(|(index, binding)| {
            types.insert(index, binding.descriptor_type);
        });
        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layoyt)
            .build();
        let layout = unsafe {
            device
                .raw
                .create_descriptor_set_layout(&layout_create_info, None)
        }?;

        let mut names = HashMap::with_capacity(set.len());
        set.iter().for_each(|(index, info)| {
            names.insert(info.name.clone(), *index);
        });

        Ok(Self {
            types,
            names,
            layout,
        })
    }

    fn create_uniform_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_type = match binding.ty {
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER => vk::DescriptorType::UNIFORM_BUFFER,
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
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

impl FreeGpuResource for DescriptorSetInfo {
    fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
    }
}
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
    pub entry: CString,
    pub descriptor_sets: StageDescriptorSetLayouts,
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
        Ok(Self {
            stage: desc.stage,
            raw: shader,
            entry: CString::new(desc.entry).unwrap(),
            descriptor_sets: rspirv_reflect::Reflection::new_from_spirv(desc.code)?
                .get_descriptor_sets()?,
        })
    }
}

impl FreeGpuResource for Shader {
    fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }
}

pub trait PipelineVertex: Sized {
    fn attribute_description() -> &'static [vk::VertexInputAttributeDescription];
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub sets: Vec<DescriptorSetInfo>,
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
        device: &Device,
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

        let sets = Self::create_descriptor_set_layouts(
            device,
            vk::ShaderStageFlags::ALL_GRAPHICS,
            &desc.shaders,
        )?;

        let descriptor_layouts = sets.iter().map(|set| set.layout).collect::<Vec<_>>();

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .build();

        let pipeline_layout =
            unsafe { device.raw.create_pipeline_layout(&layout_create_info, None) }?;

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
            device.raw.create_graphics_pipelines(
                *cache,
                slice::from_ref(&pipeline_create_info),
                None,
            )
        }
        .map_err(|_| BackendError::Other("Shit happened".into()))?[0];

        Ok(Self {
            pipeline_layout,
            pipeline,
            sets,
        })
    }

    fn create_descriptor_set_layouts(
        device: &Device,
        stages: vk::ShaderStageFlags,
        shaders: &[&Shader],
    ) -> BackendResult<Vec<DescriptorSetInfo>> {
        let sets = shaders
            .iter()
            .map(|shader| shader.descriptor_sets.clone())
            .collect::<Vec<_>>();
        let sets = Self::merge_shader_stage_layouts(sets);
        let set_count = sets.keys().map(|index| *index + 1).max().unwrap_or(0);
        let mut layouts = Vec::with_capacity(set_count as _);
        for set_index in 0..set_count {
            let set = sets.get(&set_index);
            match set {
                Some(set) => {
                    layouts.push(DescriptorSetInfo::new(device, stages, set)?);
                }
                None => {
                    layouts.push(DescriptorSetInfo::default());
                }
            }
        }

        Ok(layouts)
    }

    fn merge_shader_stage_layouts(
        stages: Vec<StageDescriptorSetLayouts>,
    ) -> StageDescriptorSetLayouts {
        let mut stages = stages.into_iter();
        let mut result = stages.next().unwrap_or_default();

        for stage in stages {
            Self::merge_shader_stage_layout_pair(stage, &mut result);
        }

        result
    }

    fn merge_shader_stage_layout_pair(
        src: StageDescriptorSetLayouts,
        dst: &mut StageDescriptorSetLayouts,
    ) {
        for (set_idx, set) in src.into_iter() {
            match dst.entry(set_idx) {
                Entry::Occupied(mut existing) => {
                    let existing = existing.get_mut();
                    for (binding_idx, binding) in set {
                        match existing.entry(binding_idx) {
                            Entry::Occupied(existing) => {
                                let existing = existing.get();
                                assert_eq!(
                                    existing.ty, binding.ty,
                                    "binding idx: {}, name: {:?}",
                                    binding_idx, binding.name
                                );
                                assert_eq!(
                                    existing.name, binding.name,
                                    "binding idx: {}, name: {:?}",
                                    binding_idx, binding.name
                                );
                            }
                            Entry::Vacant(vacant) => {
                                vacant.insert(binding);
                            }
                        }
                    }
                }
                Entry::Vacant(vacant) => {
                    vacant.insert(set);
                }
            }
        }
    }
}

impl FreeGpuResource for Pipeline {
    fn free(&self, device: &ash::Device) {
        self.sets
            .iter()
            .for_each(|set| unsafe { device.destroy_descriptor_set_layout(set.layout, None) });
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
