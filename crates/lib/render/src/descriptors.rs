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

use std::{collections::HashSet, mem::size_of, slice, sync::Arc};

use ash::vk;
use dess_common::{Handle, HandleContainer, TempList};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;

use crate::{
    uniforms::Uniforms,
    vulkan::{DescriptorSet, DescriptorSetInfo, Device},
    RenderError,
};

#[derive(Debug, Clone, Copy)]
struct BindImage {
    layout: vk::ImageLayout,
    view: vk::ImageView,
}

#[derive(Debug, Clone, Copy)]
struct BindStaticUniform {
    offset: u32,
    size: u32,
}

#[derive(Debug, Clone, Copy)]
struct BindDynamicUniform {
    buffer: vk::Buffer,
}

#[derive(Debug, Clone, Copy)]
pub struct BindingPoint<T> {
    pub binding: u32,
    pub data: Option<T>,
}

#[derive(Debug)]
pub struct Descriptor {
    pub raw: vk::DescriptorSet,
    data: Box<DescriptorData>,
}

#[derive(Debug)]
pub struct DescriptorData {
    descriptor: Option<DescriptorSet>,
    static_buffers: Vec<BindingPoint<BindStaticUniform>>,
    dynamic_buffers: Vec<BindingPoint<BindDynamicUniform>>,
    images: Vec<BindingPoint<BindImage>>,
    count: DescriptorTotalCount,
    layout: vk::DescriptorSetLayout,
}

impl DescriptorData {
    pub fn is_valid(&self) -> bool {
        self.static_buffers
            .iter()
            .all(|buffer| buffer.data.is_some())
            && self.images.iter().all(|image| image.data.is_some())
            && self.descriptor.is_some()
    }
}

impl Descriptor {
    pub fn is_valid(&self) -> bool {
        self.raw != vk::DescriptorSet::null()
    }
}

pub type DescriptorHandle = Handle<Descriptor>;

pub struct DescriptorCache {
    device: Arc<Device>,
    container: HandleContainer<Descriptor>,
    dirty: HashSet<DescriptorHandle>,
    retired_descriptors: Vec<Descriptor>,
    retired_uniforms: Vec<u32>,
    unforms_to_dealloc: Vec<u32>,
    uniforms: Uniforms,
}

unsafe impl Sync for DescriptorCache {}

impl DescriptorCache {
    pub fn new(device: &Arc<Device>) -> Result<Self, RenderError> {
        Ok(Self {
            device: device.clone(),
            container: HandleContainer::new(),
            dirty: HashSet::new(),
            retired_descriptors: Vec::new(),
            retired_uniforms: Vec::new(),
            unforms_to_dealloc: Vec::new(),
            uniforms: Uniforms::new(device)?,
        })
    }

    pub fn create(&mut self, set: &DescriptorSetInfo) -> Result<DescriptorHandle, RenderError> {
        let static_buffers = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER {
                    Some(BindingPoint::<BindStaticUniform> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let images = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    Some(BindingPoint::<BindImage> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let dynamic_buffers = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(BindingPoint::<BindDynamicUniform> {
                        binding: *index,
                        data: None,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let count = DescriptorTotalCount {
            combined_image_sampler: images.len() as _,
            uniform_buffer: static_buffers.len() as _,
            ..Default::default()
        };
        let handle = self.container.push(Descriptor {
            raw: vk::DescriptorSet::null(),
            data: Box::new(DescriptorData {
                descriptor: None,
                static_buffers,
                dynamic_buffers,
                images,
                count,
                layout: set.layout,
            }),
        });

        self.dirty.insert(handle);
        Ok(handle)
    }

    pub fn get(&self, handle: DescriptorHandle) -> Option<&Descriptor> {
        self.container.get(handle).filter(|&value| value.is_valid())
    }

    pub fn remove(&mut self, handle: DescriptorHandle) {
        if let Some(value) = self.container.remove(handle) {
            value
                .data
                .static_buffers
                .iter()
                .filter_map(|x| x.data)
                .for_each(|x| self.retired_uniforms.push(x.offset));
            self.retired_descriptors.push(value);
        }
    }

    pub fn set_image(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> Result<(), RenderError> {
        if let Some(desc) = self.container.get_mut(handle) {
            let image = desc
                .data
                .images
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = image {
                point.data = Some(BindImage { layout, view });
                self.dirty.insert(handle);
            }
        }

        Ok(())
    }

    pub fn set_uniform<T: Sized>(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        data: &T,
    ) -> Result<(), RenderError> {
        if let Some(desc) = self.container.get_mut(handle) {
            let desc = desc
                .data
                .static_buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = desc {
                if let Some(old) = point.data.replace(BindStaticUniform {
                    offset: self.uniforms.push(data)?,
                    size: size_of::<T>() as u32,
                }) {
                    self.retired_uniforms.push(old.offset);
                }
                self.dirty.insert(handle);
            }
        }
        Ok(())
    }

    pub fn set_dynamic_uniform(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        buffer: vk::Buffer,
    ) -> Result<(), RenderError> {
        if let Some(desc) = self.container.get_mut(handle) {
            let desc = desc
                .data
                .dynamic_buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = desc {
                point.data = Some(BindDynamicUniform { buffer });
                self.dirty.insert(handle);
            }
        }
        Ok(())
    }

    pub fn update_descriptors(&mut self) -> Result<(), RenderError> {
        puffin::profile_scope!("Update descriptors");
        let drop_list = &mut self.device.drop_list();
        let allocator = &mut self.device.descriptor_allocator();
        // Free uniforms from last update
        self.unforms_to_dealloc
            .drain(..)
            .for_each(|x| self.uniforms.dealloc(x));
        // Mark retired unifroms into 'to delete' list
        self.retired_uniforms
            .drain(..)
            .for_each(|x| self.unforms_to_dealloc.push(x));

        let device = self.device.raw();
        {
            puffin::profile_scope!("Allocate new descriptors");

            // Then update all dirty descriptors
            for handle in self.dirty.iter() {
                if let Some(desc) = self.container.get_mut(*handle) {
                    if let Some(old) = desc.data.descriptor.take() {
                        drop_list.free_descriptor_set(old);
                    }
                    desc.data.descriptor = Some(
                        unsafe {
                            allocator.allocate(
                                AshDescriptorDevice::wrap(self.device.raw()),
                                &desc.data.layout,
                                DescriptorSetLayoutCreateFlags::empty(),
                                &desc.data.count,
                                1,
                            )
                        }?
                        .remove(0),
                    );
                }
            }
        }
        let mut writes = Vec::with_capacity(4096);
        let mut buffers = TempList::new();
        let mut images = TempList::new();
        {
            puffin::profile_scope!("Prepare new descriptors");
            self.dirty.iter().for_each(|desc| {
                self.prepare_descriptor(&mut writes, &mut images, &mut buffers, *desc)
            });
        }
        {
            puffin::profile_scope!("Submit descriptors to device");
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        // Make dirty descriptors valid if they have eveything
        self.dirty.iter().for_each(|desc| {
            if let Some(desc) = self.container.get_mut(*desc) {
                if desc.data.is_valid() {
                    if let Some(descriptor) = &mut desc.data.descriptor {
                        desc.raw = *descriptor.raw();
                    }
                }
            }
        });
        // Discard old data
        self.dirty
            .retain(|x| !Self::is_valid_impl(&self.container, *x));
        self.retired_descriptors
            .drain(..)
            .filter_map(|x| x.data.descriptor)
            .for_each(|x| drop_list.free_descriptor_set(x));

        self.uniforms.flush()
    }

    fn prepare_descriptor(
        &self,
        writes: &mut Vec<vk::WriteDescriptorSet>,
        images: &mut TempList<vk::DescriptorImageInfo>,
        buffers: &mut TempList<vk::DescriptorBufferInfo>,
        handle: DescriptorHandle,
    ) {
        if let Some(desc) = self.container.get(handle) {
            if !desc.data.is_valid() {
                return;
            }
            if let Some(descriptor) = &desc.data.descriptor {
                desc.data
                    .images
                    .iter()
                    .map(|binding| {
                        let image = binding.data.unwrap();
                        let image = vk::DescriptorImageInfo::builder()
                            .image_layout(image.layout)
                            .image_view(image.view)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .image_info(slice::from_ref(images.add(image)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.data
                    .static_buffers
                    .iter()
                    .map(|binding| {
                        let buffer = binding.data.unwrap();
                        let data = self.uniforms.raw();
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(data)
                            .offset(buffer.offset as _)
                            .range(buffer.size as _)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .buffer_info(slice::from_ref(buffers.add(buffer)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.data
                    .dynamic_buffers
                    .iter()
                    .map(|binding| {
                        let data = binding.data.unwrap().buffer;
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(data)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .buffer_info(slice::from_ref(buffers.add(buffer)))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
            }
        }
    }

    pub fn invalidate_uniform(&mut self, handle: DescriptorHandle, binding: u32) {
        if let Some(desc) = self.container.get_mut(handle) {
            let point = desc
                .data
                .static_buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = point {
                if let Some(old) = point.data.take() {
                    self.retired_uniforms.push(old.offset);
                    desc.raw = vk::DescriptorSet::null();
                }
                self.dirty.insert(handle);
            }
        }
    }

    pub fn invalidate_image(&mut self, handle: DescriptorHandle, binding: u32) {
        if let Some(desc) = self.container.get_mut(handle) {
            let point = desc
                .data
                .images
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = point {
                point.data.take();
                desc.raw = vk::DescriptorSet::null();
                self.dirty.insert(handle);
            }
        }
    }

    pub fn invalidate_dynamic_uniform(&mut self, handle: DescriptorHandle, binding: u32) {
        if let Some(desc) = self.container.get_mut(handle) {
            let point = desc
                .data
                .dynamic_buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(point) = point {
                point.data.take();
                desc.raw = vk::DescriptorSet::null();
                self.dirty.insert(handle);
            }
        }
    }

    fn is_valid_impl(container: &HandleContainer<Descriptor>, handle: DescriptorHandle) -> bool {
        if let Some(desc) = container.get(handle) {
            desc.is_valid()
        } else {
            false
        }
    }

    pub fn is_valid(&self, handle: DescriptorHandle) -> bool {
        !handle.is_valid() && Self::is_valid_impl(&self.container, handle)
    }
}

impl Drop for DescriptorCache {
    fn drop(&mut self) {
        let mut drop_list = self.device.drop_list();
        self.retired_descriptors
            .drain(..)
            .filter_map(|x| x.data.descriptor)
            .for_each(|x| drop_list.free_descriptor_set(x));
    }
}
