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

use std::{collections::HashMap, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use smol_str::SmolStr;

use crate::Result;

use super::{Device, SamplerDesc};

const MAX_SAMPLERS: usize = 16;
pub const MAX_DESCRIPTOR_SETS: usize = 4;
pub const DYNAMIC_BINDING_SLOT: usize = 3;
pub const MATERIAL_BINDING_SLOT: usize = 1;

#[derive(Debug)]
pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
    types: HashMap<u32, vk::DescriptorType>,
    names: HashMap<SmolStr, u32>,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBindingDesc<'a> {
    pub slot: usize,
    pub ty: vk::DescriptorType,
    pub count: usize,
    pub name: &'a str,
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutDesc {
    pub stage: vk::ShaderStageFlags,
    pub set: &'static [DescriptorBindingDesc<'static>],
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
            types.insert(index as u32, binding.descriptor_type);
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
            names.insert(binding.name.into(), binding.slot as u32);
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
            .descriptor_type(binding.ty.into())
            .descriptor_count(binding.count as _)
            .stage_flags(stage.into())
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
pub(crate) struct Shader {
    pub raw: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub entry: SmolStr,
}

impl Shader {
    fn new(device: &Device, desc: &ShaderDesc) -> Result<Self> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(desc.code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.get().create_shader_module(&shader_create_info, None) }?;
        Ok(Self {
            raw: shader,
            stage: desc.stage.into(),
            entry: SmolStr::new(desc.entry),
        })
    }

    pub fn free(&self, device: &Device) {
        unsafe { device.get().destroy_shader_module(self.raw, None) }
    }
}

pub const MAX_SHADERS: usize = 3;

pub const EMPTY_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: vk::ShaderStageFlags::empty(),
    set: &[],
};
/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    device: Arc<Device>,
    shaders: ArrayVec<Shader, MAX_SHADERS>,
    sets: ArrayVec<DescriptorSetLayout, MAX_DESCRIPTOR_SETS>,
    layout: vk::PipelineLayout,
}

impl Program {
    pub fn new(device: &Arc<Device>, shaders: &[ShaderDesc]) -> Result<Self> {
        let mut stages = vk::ShaderStageFlags::empty();
        shaders.iter().for_each(|x| {
            stages |= x.stage.into();
        });

        let shaders = shaders
            .iter()
            .map(|desc| Shader::new(device, desc).unwrap())
            .collect::<ArrayVec<_, MAX_SHADERS>>();

        Ok(Self {
            device: device.clone(),
            shaders,
            sets: todo!(),
            layout: todo!(),
        })
    }

    pub(crate) fn shaders(&self) -> impl Iterator<Item = &Shader> {
        self.shaders.iter()
    }

    pub(crate) fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn descriptopr_set_layouts(&self) -> impl Iterator<Item = &DescriptorSetLayout> {
        self.sets.iter()
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.shaders
            .iter()
            .for_each(|shader| shader.free(&self.device));
    }
}
