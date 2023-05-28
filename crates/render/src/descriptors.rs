use std::{collections::HashSet, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk;
use dess_common::{Handle, HandleContainer};
use dess_render_backend::{BufferView, DescriptorSetInfo, Device, Image, ImageViewDesc};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    megabuffer::AllocatedBuffer, DescriptorAllocator, DescriptorSet, DropList, RenderResult,
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
    pub buffers: Vec<BindingPoint<AllocatedBuffer>>,
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
    container: HandleContainer<Descriptor>,
    dirty: HashSet<DescriptorHandle>,
    retired_descriptors: Vec<Descriptor>,
    retired_uniforms: Vec<AllocatedBuffer>,
}

impl DescriptorCache {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: device.clone(),
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
                    Some(BindingPoint::<AllocatedBuffer> {
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
        self.container
            .get(handle)
            .filter(|&value| value.descriptor.is_some() && value.is_valid())
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
        value: AllocatedBuffer,
    ) {
        if let Some(desc) = self.container.get_mut(handle) {
            let buffer = desc
                .buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(buffer) = buffer {
                if let Some(old) = buffer.data.replace(value) {
                    drop_list.drop_buffer(old);
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
            .for_each(|x| drop_list.drop_buffer(x));
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
                            .buffer(buffer.buffer())
                            .offset(buffer.offset())
                            .range(buffer.size());

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
