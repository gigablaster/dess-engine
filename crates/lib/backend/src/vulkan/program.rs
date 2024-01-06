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

use std::{collections::HashMap, ffi::CString, slice};

use arrayvec::ArrayVec;
use ash::vk;
use byte_slice_cast::AsSliceOf;
use smol_str::SmolStr;

use crate::{BackendError, BackendResult, ShaderStage};

use super::{Device, Index, ProgramHandle, SamplerDesc};

const MAX_SAMPLERS: usize = 16;
pub const MAX_BINDING_GROUPS: usize = 4;
pub const DYNAMIC_BINDING_SLOT: usize = 3;
pub const MATERIAL_BINDING_SLOT: usize = 1;

#[derive(Debug, Default)]
pub(crate) struct BindGroupLayout {
    pub layout: vk::DescriptorSetLayout,
    pub types: HashMap<u32, vk::DescriptorType>,
    pub names: HashMap<SmolStr, u32>,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindingDesc<'a> {
    pub slot: usize,
    pub name: &'a str,
    pub ty: BindType,
    pub count: usize,
}

#[derive(Debug, Default, Hash, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutDesc {
    pub stage: ShaderStage,
    pub set: &'static [BindingDesc<'static>],
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum BindType {
    SampledImage,
    StorageImage,
    CombinedImageSampler,
    UniformBuffer,
    DynamicUniformBuffer,
    StorageBuffer,
    DynamicStorageBuffer,
    Sampler,
}

impl From<BindType> for vk::DescriptorType {
    fn from(value: BindType) -> Self {
        match value {
            BindType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            BindType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            BindType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            BindType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            BindType::DynamicUniformBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            BindType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            BindType::DynamicStorageBuffer => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            BindType::Sampler => vk::DescriptorType::SAMPLER,
        }
    }
}

impl BindGroupLayout {
    pub(crate) fn new(
        device: &ash::Device,
        set: &BindGroupLayoutDesc,
        inmuatable_samplers: &HashMap<SamplerDesc, vk::Sampler>,
    ) -> Result<Self, BackendError> {
        let mut samplers = ArrayVec::<_, MAX_SAMPLERS>::new();
        let mut bindings = HashMap::with_capacity(set.set.len());
        for binding in set.set.iter() {
            match binding.ty {
                BindType::UniformBuffer
                | BindType::StorageBuffer
                | BindType::StorageImage
                | BindType::DynamicUniformBuffer
                | BindType::DynamicStorageBuffer
                | BindType::SampledImage => {
                    bindings.insert(binding.slot, Self::create_binding(set.stage, binding));
                }
                BindType::CombinedImageSampler | BindType::Sampler => {
                    let sampler_index = samplers.len();
                    samplers.push(
                        inmuatable_samplers
                            .get(&Self::get_suitable_sampler_desc(binding.name))
                            .unwrap(),
                    );
                    bindings.insert(
                        binding.slot,
                        Self::create_sampler_binding(set.stage, binding, samplers[sampler_index]),
                    );
                }
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
        let layout = unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }?;

        let mut names = HashMap::with_capacity(set.set.len());
        set.set.iter().for_each(|binding| {
            names.insert(binding.name.into(), binding.slot as u32);
        });

        Ok(Self {
            types,
            names,
            layout,
        })
    }

    fn create_binding(stage: ShaderStage, binding: &BindingDesc) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(binding.slot as _)
            .descriptor_type(binding.ty.into())
            .descriptor_count(binding.count as _)
            .stage_flags(stage.into())
            .build()
    }

    fn create_sampler_binding(
        stage: ShaderStage,
        binding: &BindingDesc,
        sampler: &vk::Sampler,
    ) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(binding.slot as _)
            .descriptor_count(binding.count as _)
            .descriptor_type(binding.ty.into())
            .stage_flags(stage.into())
            .immutable_samplers(slice::from_ref(sampler))
            .build()
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_set_layout(self.layout, None) };
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
        })
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }
}

pub const MAX_SHADERS: usize = 3;

pub const EMPTY_BIND_LAYOUT: BindGroupLayoutDesc = BindGroupLayoutDesc {
    stage: ShaderStage::None,
    set: &[],
};
/// Shader program similar to what we had in OpenGL.
///
/// Contains shader modules and layouts needed to create PSOs and descriptor sets.
#[derive(Debug)]
pub struct Program {
    pub(crate) shaders: ArrayVec<Shader, MAX_SHADERS>,
}

impl Program {
    pub(crate) fn new(device: &ash::Device, shaders: &[ShaderDesc]) -> Result<Self, BackendError> {
        let mut stages = vk::ShaderStageFlags::empty();
        shaders.iter().for_each(|x| {
            stages |= x.stage.into();
        });

        let shaders = shaders
            .iter()
            .map(|desc| Shader::new(device, desc).unwrap())
            .collect::<ArrayVec<_, MAX_SHADERS>>();

        Ok(Self { shaders })
    }

    pub fn free(&self, device: &ash::Device) {
        self.shaders.iter().for_each(|shader| shader.free(device));
    }
}

impl Device {
    pub fn create_program(&self, shaders: &[ShaderDesc]) -> BackendResult<ProgramHandle> {
        let program = Program::new(&self.raw, shaders)?;
        let mut programs = self.program_storage.write();
        let index = programs.len();
        programs.push(program);
        Ok(Index::new(index))
    }

    pub fn update_program(
        &self,
        handle: ProgramHandle,
        shaders: &[ShaderDesc],
    ) -> BackendResult<()> {
        let program = Program::new(&self.raw, shaders)?;
        let mut programs = self.program_storage.write();
        programs[handle.index()].free(&self.raw);
        programs[handle.index()] = program;
        // Send all pipelines that use this program to rebuild
        let pipelines = self.pipelines.read();
        let mut to_rebuild = self.raster_pipelines_to_rebuild.lock();
        pipelines.enumerate().for_each(|(pipeline_handle, ..)| {
            let desc = pipelines.get_cold(pipeline_handle).unwrap();
            if desc.program == handle {
                to_rebuild.insert(pipeline_handle);
            }
        });
        unsafe { self.raw.device_wait_idle() }?;
        Ok(())
    }
}
