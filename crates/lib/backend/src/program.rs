// Copyright (C) 2023-2024 gigablaster

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
    collections::{BTreeMap, HashMap},
    slice,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::vk::{self, PipelineLayoutCreateInfo};
use byte_slice_cast::AsSliceOf;
use smol_str::SmolStr;

use crate::{AsVulkan, Result};

use super::{Device, SamplerDesc};

const MAX_SAMPLERS: usize = 16;
pub const MAX_DESCRIPTOR_SETS: usize = 4;
pub const MATERIAL_BINDING_SLOT: u32 = 1;
pub const OBJECT_BINDING_SLOT: u32 = 2;
pub const DYNAMIC_BINDING_SLOT: u32 = 3;

#[derive(Debug)]
pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
    pub types: HashMap<usize, vk::DescriptorType>,
    pub names: HashMap<SmolStr, usize>,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBindingDesc<'a> {
    pub slot: usize,
    pub ty: vk::DescriptorType,
    pub count: usize,
    pub name: &'a str,
}

#[derive(Debug)]
pub struct BindGroupLayoutDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub set: Vec<DescriptorBindingDesc<'a>>,
}

impl DescriptorSetLayout {
    pub fn new(device: &Arc<Device>, set: &BindGroupLayoutDesc) -> Result<Self> {
        let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
        let mut bindings = HashMap::with_capacity(set.set.len());
        for binding in set.set.iter() {
            match binding.ty {
                vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                | vk::DescriptorType::SAMPLED_IMAGE => {
                    bindings.insert(binding.slot, Self::create_binding(set.stage, binding));
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER | vk::DescriptorType::SAMPLER => {
                    let sampler_index = samplers.len();
                    samplers.push(
                        device
                            .sampler(&Self::get_suitable_sampler_desc(binding.name))
                            .unwrap(),
                    );
                    bindings.insert(
                        binding.slot,
                        Self::create_sampler_binding(set.stage, binding, &samplers[sampler_index]),
                    );
                }
                _ => panic!("Not yet implemented {:?}", binding.ty),
            };
        }

        let layoyt = bindings.values().copied().collect::<Vec<_>>();
        let mut types = HashMap::with_capacity(set.set.len());
        bindings.into_iter().for_each(|(index, binding)| {
            types.insert(index, binding.descriptor_type);
        });
        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&layoyt)
            .build();
        let layout = unsafe {
            device
                .get()
                .create_descriptor_set_layout(&layout_create_info, None)
        }?;

        let mut names = HashMap::with_capacity(set.set.len());
        set.set.iter().for_each(|binding| {
            names.insert(binding.name.into(), binding.slot);
        });

        Ok(Self {
            device: device.clone(),
            types,
            names,
            layout,
        })
    }

    fn create_binding(
        stage: vk::ShaderStageFlags,
        binding: &DescriptorBindingDesc,
    ) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(binding.slot as _)
            .descriptor_type(binding.ty)
            .descriptor_count(binding.count as _)
            .stage_flags(stage)
            .build()
    }

    fn create_sampler_binding(
        stage: vk::ShaderStageFlags,
        binding: &DescriptorBindingDesc,
        sampler: &vk::Sampler,
    ) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(binding.slot as _)
            .descriptor_count(binding.count as _)
            .descriptor_type(binding.ty)
            .stage_flags(stage)
            .immutable_samplers(slice::from_ref(sampler))
            .build()
    }

    fn get_suitable_sampler_desc(name: &str) -> SamplerDesc {
        let address_mode = if name.ends_with("_e") {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        } else if name.ends_with("_m") {
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

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .get()
                .destroy_descriptor_set_layout(self.layout, None)
        };
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct ShaderDesc<'a> {
    pub stage: vk::ShaderStageFlags,
    pub entry: &'a str,
    pub code: &'a [u8],
}

impl<'a> ShaderDesc<'a> {
    pub fn new(stage: vk::ShaderStageFlags, code: &'a [u8]) -> Self {
        Self {
            stage,
            entry: "main",
            code,
        }
    }

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

#[derive(Debug, Clone)]
struct DescriptorInfo(vk::DescriptorType, usize, SmolStr);

type DescriptorSetInfo = HashMap<usize, DescriptorInfo>;

#[derive(Debug)]
pub(crate) struct Shader {
    raw: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
    entry: SmolStr,
    sets: HashMap<usize, DescriptorSetInfo>,
}

impl AsVulkan<vk::ShaderModule> for Shader {
    fn as_vk(&self) -> vk::ShaderModule {
        self.raw
    }
}

impl Shader {
    fn new(device: &Device, desc: &ShaderDesc) -> Result<Self> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(desc.code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.get().create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            raw: shader,
            stage: desc.stage,
            entry: SmolStr::new(desc.entry),
            sets: Self::reflect(desc.code)?,
        })
    }

    pub fn free(&self, device: &Device) {
        unsafe { device.get().destroy_shader_module(self.raw, None) }
    }

    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    fn reflect(code: &[u8]) -> Result<HashMap<usize, DescriptorSetInfo>> {
        let info = rspirv_reflect::Reflection::new_from_spirv(code)?;
        let sets = info.get_descriptor_sets()?;
        Ok(sets
            .iter()
            .map(|(index, info)| {
                (
                    *index as usize,
                    Self::extract_descriptor_set(info, *index == DYNAMIC_BINDING_SLOT),
                )
            })
            .collect())
    }

    fn extract_descriptor_set(
        info: &BTreeMap<u32, rspirv_reflect::DescriptorInfo>,
        dynamic: bool,
    ) -> DescriptorSetInfo {
        info.iter()
            .map(|(index, info)| (*index as usize, Self::extract_descriptor(info, dynamic)))
            .collect()
    }

    fn extract_descriptor(info: &rspirv_reflect::DescriptorInfo, dynamic: bool) -> DescriptorInfo {
        let name = SmolStr::from(&info.name);
        let count = Self::binding_count(info);
        match info.ty {
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER if dynamic => {
                DescriptorInfo(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, count, name)
            }
            rspirv_reflect::DescriptorType::UNIFORM_BUFFER => {
                DescriptorInfo(vk::DescriptorType::UNIFORM_BUFFER, count, name)
            }
            rspirv_reflect::DescriptorType::STORAGE_BUFFER if dynamic => {
                DescriptorInfo(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, count, name)
            }
            rspirv_reflect::DescriptorType::STORAGE_BUFFER => {
                DescriptorInfo(vk::DescriptorType::STORAGE_BUFFER, count, name)
            }
            rspirv_reflect::DescriptorType::SAMPLER => {
                DescriptorInfo(vk::DescriptorType::SAMPLER, count, name)
            }
            rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                DescriptorInfo(vk::DescriptorType::SAMPLED_IMAGE, count, name)
            }
            rspirv_reflect::DescriptorType::STORAGE_IMAGE => {
                DescriptorInfo(vk::DescriptorType::STORAGE_IMAGE, count, name)
            }
            rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                DescriptorInfo(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, count, name)
            }
            _ => panic!("Binding of type {:?} isn't supported", info.ty),
        }
    }

    fn binding_count(info: &rspirv_reflect::DescriptorInfo) -> usize {
        match info.binding_count {
            rspirv_reflect::BindingCount::One => 1,
            rspirv_reflect::BindingCount::StaticSized(n) => n,
            rspirv_reflect::BindingCount::Unbounded => {
                panic!("Unbounded bindings aren't supported")
            }
        }
    }
}

/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    device: Arc<Device>,
    shaders: Vec<Shader>,
    sets: HashMap<usize, DescriptorSetLayout>,
    layout: vk::PipelineLayout,
}

impl Program {
    pub fn new(device: &Arc<Device>, shaders: &[ShaderDesc]) -> Result<Self> {
        let mut stages = vk::ShaderStageFlags::empty();
        let shaders = shaders
            .iter()
            .map(|desc| {
                stages |= desc.stage;
                Shader::new(device, &desc).unwrap()
            })
            .collect::<Vec<_>>();
        let sets = shaders
            .iter()
            .map(|shader| &shader.sets)
            .collect::<Vec<_>>();
        let sets = Self::merge_descriptor_sets(&sets);

        let sets = sets
            .iter()
            .map(|(index, set)| (*index, Self::create_layout(device, stages, set).unwrap()))
            .collect::<HashMap<_, _>>();
        let layouts = sets.values().map(|x| x.layout).collect::<Vec<_>>();
        let info = PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .build();
        let layout = unsafe { device.get().create_pipeline_layout(&info, None) }?;
        Ok(Self {
            device: device.clone(),
            shaders,
            sets,
            layout,
        })
    }

    fn create_layout(
        device: &Arc<Device>,
        stage: vk::ShaderStageFlags,
        set: &DescriptorSetInfo,
    ) -> Result<DescriptorSetLayout> {
        let set = set
            .iter()
            .map(|(slot, info)| DescriptorBindingDesc {
                slot: *slot,
                ty: info.0,
                count: info.1,
                name: &info.2,
            })
            .collect::<Vec<_>>();
        DescriptorSetLayout::new(device, &BindGroupLayoutDesc { stage, set })
    }

    pub(crate) fn shaders(&self) -> impl Iterator<Item = &Shader> {
        self.shaders.iter()
    }

    pub(crate) fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn descriptopr_set_layouts(&self) -> impl Iterator<Item = (&usize, &DescriptorSetLayout)> {
        self.sets.iter()
    }

    fn merge_descriptor_sets(
        sets: &[&HashMap<usize, DescriptorSetInfo>],
    ) -> HashMap<usize, DescriptorSetInfo> {
        let mut result = HashMap::new();
        for index in 0..MAX_DESCRIPTOR_SETS {
            let entry = result.entry(index).or_insert(DescriptorSetInfo::new());
            for set in sets {
                if let Some(group) = set.get(&index) {
                    for (index, descriptor) in group {
                        entry.insert(*index, descriptor.clone());
                    }
                }
            }
        }

        result
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.shaders
            .iter()
            .for_each(|shader| shader.free(&self.device));
    }
}
