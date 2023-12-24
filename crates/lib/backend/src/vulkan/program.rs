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
    mem::discriminant,
    slice,
};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use rspirv_reflect::{BindingCount, DescriptorInfo};

use crate::{BackendError, BackendResult, ShaderStage};

use super::{Device, Index, ProgramHandle, SamplerDesc};

const MAX_SAMPLERS: usize = 16;
const MAX_SETS: usize = 4;
pub const PER_PASS_BINDING_SLOT: usize = 0;
pub const PER_MATERIAL_BINDING_SLOT: usize = 1;
pub const PER_OBJECT_BINDING_SLOT: usize = 2;
pub const PER_DRAW_BINDING_SLOT: usize = 3;

// Slots are
// 0 - per pass
// 1 - per material
// 2 - per object (optional)
// 3 - dynamic shit
static DESCRIPTORS_PER_SLOT: [u32; MAX_SETS] = [4, 64, 256, 512];

type DescriptorSetLayout = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = BTreeMap<u32, DescriptorSetLayout>;

#[derive(Debug, Default)]
pub(crate) struct DescriptorSetInfo {
    pub layout: vk::DescriptorSetLayout,
    pub types: HashMap<u32, vk::DescriptorType>,
    pub names: HashMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorSetCreateInfo {
    pub stage: ShaderStage,
    pub expected_count: u32,
    set: HashMap<u32, DescriptorInfo>,
}

impl std::hash::Hash for DescriptorSetCreateInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.stage.hash(state);
        self.expected_count.hash(state);
        self.set.iter().for_each(|(index, info)| {
            index.hash(state);
            discriminant(&info.binding_count);
            if let BindingCount::StaticSized(count) = info.binding_count {
                count.hash(state);
            }
            info.name.hash(state);
            info.ty.0.hash(state);
        });
    }
}

pub enum DescriptorType {
    SampledImage,
    StorageImage,
    CombinedImageSampler,
    UniformBuffer,
    DynamicUniformBuffer,
    StorageBuffer,
    DynamicStorageBuffer,
}

impl DescriptorSetCreateInfo {
    pub fn stage(mut self, stage: ShaderStage) -> Self {
        self.stage = stage;

        self
    }

    pub fn expected_count(mut self, count: u32) -> Self {
        self.expected_count = count;
        self
    }

    pub fn descriptor(mut self, index: usize, name: &str, ty: DescriptorType) -> Self {
        let ty = match ty {
            DescriptorType::UniformBuffer => rspirv_reflect::DescriptorType::UNIFORM_BUFFER,
            DescriptorType::DynamicUniformBuffer => {
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
            DescriptorType::StorageBuffer => rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            DescriptorType::DynamicStorageBuffer => {
                rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC
            }
            DescriptorType::SampledImage => rspirv_reflect::DescriptorType::SAMPLED_IMAGE,
            DescriptorType::StorageImage => rspirv_reflect::DescriptorType::STORAGE_IMAGE,
            DescriptorType::CombinedImageSampler => {
                rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
        };
        self.set.insert(
            index as u32,
            rspirv_reflect::DescriptorInfo {
                ty,
                binding_count: rspirv_reflect::BindingCount::One,
                name: name.to_owned(),
            },
        );

        self
    }
}

impl DescriptorSetInfo {
    pub(crate) fn from_desc(
        device: &ash::Device,
        desc: &DescriptorSetCreateInfo,
        inmuatable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    ) -> BackendResult<Self> {
        Self::new(
            device,
            desc.stage.into(),
            &desc.set,
            false,
            inmuatable_samplers,
        )
    }

    pub(crate) fn new(
        device: &ash::Device,
        stage: vk::ShaderStageFlags,
        set: &HashMap<u32, DescriptorInfo>,
        dynamic: bool,
        inmuatable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    ) -> Result<Self, BackendError> {
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
                    let binding = Self::create_buffer_binding(*index, stage, binding, dynamic);

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
                        inmuatable_samplers
                            .get(&Self::get_suitable_sampler_desc(binding))
                            .unwrap(),
                    );
                    bindings.insert(
                        *index,
                        Self::create_sampler_binding(
                            *index,
                            stage,
                            binding,
                            samplers[sampler_index],
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
        let layout = unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }?;

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

    fn create_buffer_binding(
        index: u32,
        stage: vk::ShaderStageFlags,
        binding: &DescriptorInfo,
        dynamic: bool,
    ) -> vk::DescriptorSetLayoutBinding {
        let descriptor_type = match binding.ty {
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER if dynamic => {
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            }
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER => vk::DescriptorType::UNIFORM_BUFFER,
            rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER
            }
            rspirv_reflect::DescriptorType::STORAGE_IMAGE => vk::DescriptorType::STORAGE_IMAGE,
            rspirv_reflect::DescriptorType::STORAGE_BUFFER if dynamic => {
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            }
            rspirv_reflect::DescriptorType::STORAGE_BUFFER => vk::DescriptorType::STORAGE_BUFFER,
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

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
    }

    fn get_suitable_sampler_desc(binding: &DescriptorInfo) -> SamplerDesc {
        let address_mode = if binding.name.ends_with("_e") {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        } else if binding.name.ends_with("_m") {
            vk::SamplerAddressMode::MIRRORED_REPEAT
        } else {
            vk::SamplerAddressMode::REPEAT
        };
        SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode,
            anisotropy_level: 16, // TODO:: control anisotropy level
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: ShaderStage,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: ShaderStage, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

    pub fn vertex(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Vertex,
            entry: "main",
            code,
        }
    }

    pub fn fragment(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Fragment,
            entry: "main",
            code,
        }
    }

    pub fn compute(code: &'a [u8]) -> Self {
        Self {
            stage: ShaderStage::Compute,
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
pub(crate) struct Shader {
    pub raw: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub entry: CString,
    pub descriptor_sets: StageDescriptorSetLayouts,
}

impl Shader {
    pub fn new(device: &ash::Device, desc: &ShaderDesc) -> Result<Self, BackendError> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(desc.code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            raw: shader,
            stage: desc.stage.into(),
            entry: CString::new(desc.entry).unwrap(),
            descriptor_sets: rspirv_reflect::Reflection::new_from_spirv(desc.code)?
                .get_descriptor_sets()?,
        })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }
}

pub const MAX_SHADERS: usize = 2;

/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) sets: ArrayVec<DescriptorSetInfo, MAX_SETS>,
    pub(crate) shaders: ArrayVec<Shader, MAX_SHADERS>,
}

impl Program {
    pub(crate) fn new(
        device: &ash::Device,
        shaders: &[ShaderDesc],
        inmuatable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    ) -> Result<Self, BackendError> {
        let mut stages = vk::ShaderStageFlags::empty();
        shaders.iter().for_each(|x| {
            stages |= x.stage.into();
        });

        let shaders = shaders
            .iter()
            .map(|desc| Shader::new(device, desc).unwrap())
            .collect::<ArrayVec<_, MAX_SHADERS>>();

        let sets =
            Self::create_descriptor_set_layouts(device, stages, &shaders, inmuatable_samplers)?;

        let descriptor_layouts = sets
            .iter()
            .map(|set| set.layout)
            .collect::<ArrayVec<_, MAX_SETS>>();

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_layouts)
            .build();

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }?;

        Ok(Self {
            pipeline_layout,
            sets,
            shaders,
        })
    }

    fn create_descriptor_set_layouts(
        device: &ash::Device,
        stages: vk::ShaderStageFlags,
        shaders: &[Shader],
        inmuatable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    ) -> Result<ArrayVec<DescriptorSetInfo, MAX_SETS>, BackendError> {
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
                    let set = set
                        .iter()
                        .map(|(a, b)| (*a, b.clone()))
                        .collect::<HashMap<_, _>>();
                    layouts.push(DescriptorSetInfo::new(
                        device,
                        stages,
                        &set,
                        set_index == PER_DRAW_BINDING_SLOT as u32,
                        inmuatable_samplers,
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

    pub fn free(&self, device: &ash::Device) {
        self.shaders.iter().for_each(|shader| shader.free(device));
        unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
        self.sets.iter().for_each(|set| set.free(device));
    }
}

impl Device {
    pub fn create_program(&self, shaders: &[ShaderDesc]) -> BackendResult<ProgramHandle> {
        let program = Program::new(&self.raw, shaders, &self.samplers)?;
        let mut programs = self.program_storage.write();
        let index = programs.len();
        programs.push(program);
        Ok(Index::new(index))
    }
}
