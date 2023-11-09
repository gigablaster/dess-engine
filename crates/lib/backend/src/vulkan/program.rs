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
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use gpu_descriptor::DescriptorTotalCount;
use rspirv_reflect::{BindingCount, DescriptorInfo};

use crate::{GpuResource, RenderError};

use super::{Device, SamplerDesc};

const MAX_SAMPLERS: usize = 16;
const MAX_SETS: usize = 4;
pub const PASS_BINDING_SLOT: u32 = 0;
pub const MATERIAL_BINDING_SLOT: u32 = 1;
pub const DYNAMIC_BINDING_SLOT: u32 = 3;

// Slots are
// 0 - per pass
// 1 - per material
// 2 - per object (optional)
// 3 - dynamic shit
static DESCRIPTORS_PER_SLOT: [u32; MAX_SETS] = [4, 64, 256, 512];

type DescriptorSetLayout = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = BTreeMap<u32, DescriptorSetLayout>;

#[derive(Debug, Default)]
pub struct DescriptorSetInfo {
    pub layout: vk::DescriptorSetLayout,
    pub descriptor_count: DescriptorTotalCount,
    pub types: HashMap<u32, vk::DescriptorType>,
    pub names: HashMap<String, u32>,
}

impl DescriptorSetInfo {
    pub fn new(
        device: &Device,
        set_index: u32,
        stage: vk::ShaderStageFlags,
        set: &BTreeMap<u32, DescriptorInfo>,
        expected_count: u32,
    ) -> Result<Self, RenderError> {
        let mut uniform_buffers_count = 0;
        let mut dynamic_uniform_buffers_count = 0;
        let mut combined_samplers_count = 0;
        let mut sampled_images_count = 0;
        let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
        let mut bindings = HashMap::with_capacity(set.len());
        for (index, binding) in set.iter() {
            match binding.ty {
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER
                | rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER
                | rspirv_reflect::DescriptorType::STORAGE_IMAGE
                | rspirv_reflect::DescriptorType::STORAGE_BUFFER
                | rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    let binding = Self::create_uniform_binding(
                        *index,
                        stage,
                        binding,
                        set_index == DYNAMIC_BINDING_SLOT, // Force last slot to have dynamic bindings
                    );

                    bindings.insert(*index, binding);
                    if binding.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                        dynamic_uniform_buffers_count += 1
                    } else {
                        uniform_buffers_count += 1;
                    }
                }
                rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                    bindings.insert(
                        *index,
                        Self::create_sampled_image_binding(*index, stage, binding),
                    );
                    sampled_images_count += 1;
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
                    combined_samplers_count += 1;
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
                .raw()
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
            descriptor_count: DescriptorTotalCount {
                sampler: 0,
                combined_image_sampler: combined_samplers_count * expected_count,
                sampled_image: sampled_images_count * expected_count,
                storage_image: 0,
                uniform_texel_buffer: 0,
                storage_texel_buffer: 0,
                uniform_buffer: uniform_buffers_count * expected_count,
                storage_buffer: 0,
                uniform_buffer_dynamic: dynamic_uniform_buffers_count * expected_count,
                storage_buffer_dynamic: 0,
                input_attachment: 0,
                acceleration_structure: 0,
                inline_uniform_block_bytes: 0,
                inline_uniform_block_bindings: 0,
            },
        })
    }

    fn create_uniform_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
        force_dynamic: bool,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_type = match binding.ty {
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER if !force_dynamic => {
                vk::DescriptorType::UNIFORM_BUFFER
            }
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER if force_dynamic => {
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
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

impl GpuResource for DescriptorSetInfo {
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

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: vk::ShaderStageFlags::COMPUTE,
            entry: "main",
            code,
        }
    }

    pub fn entry(mut self, entry: &'a str) -> Self {
        self.entry = entry;
        self
    }
}

#[derive(Debug)]
pub(crate) struct Shader {
    pub raw: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub entry: CString,
    pub descriptor_sets: StageDescriptorSetLayouts,
}

impl Shader {
    pub fn new(device: &ash::Device, desc: &ShaderDesc) -> Result<Self, RenderError> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(desc.code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            stage: desc.stage,
            raw: shader,
            entry: CString::new(desc.entry).unwrap(),
            descriptor_sets: rspirv_reflect::Reflection::new_from_spirv(desc.code)?
                .get_descriptor_sets()?,
        })
    }
}

impl GpuResource for Shader {
    fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }
}

/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    device: Arc<Device>,
    pipeline_layout: vk::PipelineLayout,
    sets: ArrayVec<DescriptorSetInfo, MAX_SETS>,
    shaders: Vec<Shader>,
}

impl Program {
    pub fn new(device: &Arc<Device>, shaders: &[ShaderDesc]) -> Result<Self, RenderError> {
        let mut stages = vk::ShaderStageFlags::empty();
        shaders.iter().for_each(|x| {
            stages |= x.stage;
        });

        let shaders = shaders
            .iter()
            .map(|desc| Shader::new(device.raw(), desc).unwrap())
            .collect::<Vec<_>>();

        let sets = Self::create_descriptor_set_layouts(device, stages, &shaders)?;

        let descriptor_layouts = sets
            .iter()
            .map(|set| set.layout)
            .collect::<ArrayVec<_, MAX_SETS>>();

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .build();

        let pipeline_layout = unsafe {
            device
                .raw()
                .create_pipeline_layout(&layout_create_info, None)
        }?;

        Ok(Self {
            device: device.clone(),
            pipeline_layout,
            sets,
            shaders,
        })
    }

    fn create_descriptor_set_layouts(
        device: &Device,
        stages: vk::ShaderStageFlags,
        shaders: &[Shader],
    ) -> Result<ArrayVec<DescriptorSetInfo, MAX_SETS>, RenderError> {
        let sets = shaders
            .iter()
            .map(|shader| shader.descriptor_sets.clone())
            .collect::<Vec<_>>();
        let sets = Self::merge_shader_stage_layouts(sets);
        let set_count = sets.keys().map(|index| *index + 1).max().unwrap_or(0);
        let mut layouts = ArrayVec::new();
        for set_index in 0..set_count {
            let set = sets.get(&set_index);
            match set {
                Some(set) => {
                    layouts.push(DescriptorSetInfo::new(
                        device,
                        set_index,
                        stages,
                        set,
                        DESCRIPTORS_PER_SLOT[set_index as usize],
                    )?);
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

    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub(crate) fn shaders(&self) -> &[Shader] {
        &self.shaders
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn descriptor_set(&self, index: usize) -> &DescriptorSetInfo {
        &self.sets[index]
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.shaders
            .drain(..)
            .for_each(|shader| shader.free(self.device.raw()));
        unsafe {
            self.device
                .raw()
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
        self.sets.iter().for_each(|set| set.free(self.device.raw()));
    }
}
