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

use ash::vk;
use byte_slice_cast::AsSliceOf;
use rspirv_reflect::{BindingCount, DescriptorInfo};

use crate::BackendResult;

#[derive(Debug, Clone)]
pub struct Shader {
    pub raw: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
    pub layouts: Vec<vk::DescriptorSetLayout>,
}

impl Shader {
    fn new(device: &ash::Device, stage: vk::ShaderStageFlags, code: &[u8]) -> BackendResult<Self> {
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(code.as_slice_of::<u32>().unwrap())
            .build();

        let shader = unsafe { device.create_shader_module(&shader_create_info, None) }?;

        Ok(Self {
            stage,
            raw: shader,
            layouts: Self::create_descriptor_set_layouts(device, stage, code)?,
        })
    }

    pub fn vertex(device: &ash::Device, code: &[u8]) -> BackendResult<Self> {
        Self::new(device, vk::ShaderStageFlags::VERTEX, code)
    }

    pub fn fragment(device: &ash::Device, code: &[u8]) -> BackendResult<Self> {
        Self::new(device, vk::ShaderStageFlags::FRAGMENT, code)
    }

    pub fn free(&self, device: &ash::Device) {
        unsafe { device.destroy_shader_module(self.raw, None) }
    }

    fn create_descriptor_set_layouts(
        device: &ash::Device,
        stage: vk::ShaderStageFlags,
        code: &[u8],
    ) -> BackendResult<Vec<vk::DescriptorSetLayout>> {
        let info = rspirv_reflect::Reflection::new_from_spirv(code)?;
        let sets = info.get_descriptor_sets()?;
        let set_count = sets.keys().map(|index| *index + 1).max().unwrap_or(0);
        let mut layouts = Vec::with_capacity(8);
        let create_flags = vk::DescriptorSetLayoutCreateFlags::empty();
        for index in 0..set_count {
            let set = sets.get(&index);
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
                        _ => unimplemented!("{:?}", binding),
                    }
                }
                let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(create_flags)
                    .bindings(&bindings)
                    .build();
                let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
                layouts.push(layout);
            } else {
                let layout = unsafe {
                    device.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder().build(),
                        None,
                    )
                }?;
                layouts.push(layout);
            }
        }

        Ok(layouts)
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
}
