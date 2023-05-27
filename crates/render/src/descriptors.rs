use std::{collections::HashSet, mem::replace, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk;
use dess_common::{Handle, HandleContainer};
use dess_render_backend::{Buffer, DescriptorSetInfo, Device, Image, ImageViewDesc};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    uniforms::UniformBuffer, DescriptorAllocator, DescriptorSet, DropList, RenderResult,
    UpdateContext,
};

#[derive(Debug, Clone, Copy)]
pub struct BindedImage {
    layout: vk::ImageLayout,
    view: vk::ImageView,
}

#[derive(Debug, Clone, Copy)]
pub struct BindingPoint<T> {
    pub binding: u32,
    pub data: Option<T>,
}

#[derive(Debug)]
pub struct Descriptor {
    pub descriptor: Option<DescriptorSet>,
    pub buffers: Vec<BindingPoint<UniformBuffer>>,
    pub images: Vec<BindingPoint<BindedImage>>,
    pub count: DescriptorTotalCount,
    pub layout: vk::DescriptorSetLayout,
}

impl Descriptor {
    pub fn is_valid(&self) -> bool {
        self.buffers.iter().all(|x| x.data.is_some())
            && self.images.iter().all(|x| x.data.is_some())
    }
}

pub type DescriptorHandle = Handle<Descriptor>;

pub struct DescriptorCache {
    device: Arc<Device>,
    uniform_buffer: Arc<Buffer>,
    container: HandleContainer<Descriptor>,
    dirty: HashSet<DescriptorHandle>,
    retired_descriptors: Vec<Descriptor>,
    retired_uniforms: Vec<UniformBuffer>,
}

impl DescriptorCache {
    pub fn new(device: &Arc<Device>, buffer: &Arc<Buffer>) -> Self {
        Self {
            device: device.clone(),
            uniform_buffer: buffer.clone(),
            container: HandleContainer::new(),
            dirty: HashSet::new(),
            retired_descriptors: Vec::new(),
            retired_uniforms: Vec::new(),
        }
    }

    pub fn create(&mut self, set: &DescriptorSetInfo) -> RenderResult<DescriptorHandle> {
        let buffers = set
            .types
            .iter()
            .filter_map(|(index, ty)| {
                if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                    Some(BindingPoint::<UniformBuffer> {
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
                    Some(BindingPoint::<BindedImage> {
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
            uniform_buffer_dynamic: buffers.len() as _,
            ..Default::default()
        };
        let handle = self.container.push(Descriptor {
            descriptor: None,
            buffers,
            images,
            count,
            layout: set.layout,
        });

        self.dirty.insert(handle);
        Ok(handle)
    }

    pub fn get(&self, handle: DescriptorHandle) -> Option<&Descriptor> {
        if let Some(value) = self.container.get(handle) {
            if value.descriptor.is_some() && value.is_valid() {
                Some(value)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn remove(&mut self, handle: DescriptorHandle) {
        if let Some(value) = self.container.remove(handle) {
            value
                .buffers
                .iter()
                .filter_map(|x| x.data)
                .for_each(|x| self.retired_uniforms.push(x));
            self.retired_descriptors.push(value);
        }
    }

    pub fn set_image(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        value: &Image,
        aspect: vk::ImageAspectFlags,
        layout: vk::ImageLayout,
    ) -> RenderResult<()> {
        if let Some(desc) = self.container.get_mut(handle) {
            let image = desc
                .images
                .iter_mut()
                .find(|point| point.binding == binding);
            let view = value.get_or_create_view(ImageViewDesc::default().aspect_mask(aspect))?;
            if let Some(image) = image {
                image.data = Some(BindedImage { layout, view });
                self.dirty.insert(handle);
            }
        }

        Ok(())
    }

    pub(crate) fn set_buffer(
        &mut self,
        drop_list: &mut DropList,
        handle: DescriptorHandle,
        binding: u32,
        value: UniformBuffer,
    ) {
        if let Some(desc) = self.container.get_mut(handle) {
            let buffer = desc
                .buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(buffer) = buffer {
                if let Some(old) = buffer.data.replace(value) {
                    drop_list.drop_uniform_buffer(old);
                }
            }
        }
    }

    pub(crate) fn update_descriptors(
        &mut self,
        allocator: &mut DescriptorAllocator,
        drop_list: &mut DropList,
    ) -> RenderResult<()> {
        puffin::profile_scope!("update descriptors");
        let device = &self.device.raw;
        for handle in self.dirty.iter() {
            if let Some(desc) = self.container.get_mut(*handle) {
                if let Some(old) = desc.descriptor.take() {
                    drop_list.drop_descriptor_set(old);
                }
                desc.descriptor = Some(
                    unsafe {
                        allocator.allocate(
                            AshDescriptorDevice::wrap(&self.device.raw),
                            &desc.layout,
                            DescriptorSetLayoutCreateFlags::empty(),
                            &desc.count,
                            1,
                        )
                    }?
                    .remove(0),
                );
            }
        }
        self.dirty
            .par_iter()
            .for_each(|desc| self.update_descriptor(device, *desc));
        self.dirty.retain(|x| !Self::is_valid(&self.container, *x));
        self.retired_uniforms
            .drain(..)
            .for_each(|x| drop_list.drop_uniform_buffer(x));
        self.retired_descriptors
            .drain(..)
            .filter_map(|x| x.descriptor)
            .for_each(|x| drop_list.drop_descriptor_set(x));

        Ok(())
    }

    fn update_descriptor(&self, device: &ash::Device, handle: DescriptorHandle) {
        if let Some(desc) = self.container.get(handle) {
            if !desc.is_valid() {
                return;
            }
            if let Some(descriptor) = &desc.descriptor {
                let mut writes = ArrayVec::<vk::WriteDescriptorSet, 256>::new();
                desc.images
                    .iter()
                    .map(|binding| {
                        let image = binding.data.unwrap();
                        let image = vk::DescriptorImageInfo::builder()
                            .image_layout(image.layout)
                            .image_view(image.view)
                            .build();

                        vk::WriteDescriptorSet::builder()
                            .image_info(slice::from_ref(&image))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .build()
                    })
                    .for_each(|x| writes.push(x));
                desc.buffers
                    .iter()
                    .map(|binding| {
                        let buffer = binding.data.unwrap();
                        let buffer = vk::DescriptorBufferInfo::builder()
                            .buffer(self.uniform_buffer.raw)
                            .offset(buffer.offset as _)
                            .range(buffer.size as _);

                        vk::WriteDescriptorSet::builder()
                            .buffer_info(slice::from_ref(&buffer))
                            .dst_binding(binding.binding)
                            .dst_set(*descriptor.raw())
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .build()
                    })
                    .for_each(|x| writes.push(x));

                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
        }
    }

    fn is_valid(container: &HandleContainer<Descriptor>, handle: DescriptorHandle) -> bool {
        if let Some(desc) = container.get(handle) {
            desc.is_valid()
        } else {
            false
        }
    }
}
