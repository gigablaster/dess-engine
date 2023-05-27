use std::{collections::HashSet, slice, sync::Arc};

use arrayvec::ArrayVec;
use ash::vk;
use dess_common::{Handle, HandleContainer};
use dess_render_backend::{
    Buffer, DescriptorAllocator, DescriptorSet, DescriptorSetInfo, Device, DropList, Image,
    ImageViewDesc,
};
use gpu_descriptor::{DescriptorSetLayoutCreateFlags, DescriptorTotalCount};
use gpu_descriptor_ash::AshDescriptorDevice;
use rayon::prelude::{IntoParallelRefIterator, ParallelDrainFull, ParallelIterator};

use crate::{uniforms::UniformBuffer, RenderResult};

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
    descriptor: DescriptorSet,
    buffers: Vec<BindingPoint<UniformBuffer>>,
    images: Vec<BindingPoint<BindedImage>>,
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
    retired: Vec<Descriptor>
}

impl DescriptorCache {
    pub fn new(device: &Arc<Device>, buffer: &Arc<Buffer>) -> Self {
        Self {
            device: device.clone(),
            uniform_buffer: buffer.clone(),
            container: HandleContainer::new(),
            dirty: HashSet::new(),
            retired: Vec::new()
        }
    }

    pub fn create(
        &mut self,
        allocator: &mut DescriptorAllocator,
        set: &DescriptorSetInfo,
    ) -> RenderResult<DescriptorHandle> {
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
        let descriptor = unsafe {
            allocator.allocate(
                AshDescriptorDevice::wrap(&self.device.raw),
                &set.layout,
                DescriptorSetLayoutCreateFlags::empty(),
                &count,
                1,
            )
        }?
        .remove(0);
        let handle = self.container.push(Descriptor {
            descriptor,
            buffers,
            images,
        });

        self.dirty.insert(handle);
        Ok(handle)
    }

    pub fn get(&self, handle: DescriptorHandle) -> Option<vk::DescriptorSet> {
        if let Some(value) = self.container.get(handle) {
            if value.is_valid() {
                Some(*value.descriptor.raw())
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn remove(&mut self, handle: DescriptorHandle) {
        if let Some(value) = self.container.remove(handle) {
            self.retired.push(value);
        }
    }

    pub fn set_image(
        &mut self,
        handle: DescriptorHandle,
        binding: u32,
        value: &Arc<Image>,
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

    pub fn set_buffer(&mut self, handle: DescriptorHandle, binding: u32, value: UniformBuffer) {
        if let Some(desc) = self.container.get_mut(handle) {
            let buffer = desc
                .buffers
                .iter_mut()
                .find(|point| point.binding == binding);
            if let Some(buffer) = buffer {
                buffer.data = Some(value);
            }
        }
    }

    pub(crate) fn update_descriptors(&mut self) {
        puffin::profile_scope!("update descriptors");
        let device = &self.device.raw;
        self.dirty
            .par_iter()
            .for_each(|desc| self.update_descriptor(device, *desc));
        self.dirty.retain(|x| !Self::is_valid(&self.container, *x));
    }

    fn update_descriptor(&self, device: &ash::Device, handle: DescriptorHandle) {
        if let Some(desc) = self.container.get(handle) {
            if !desc.is_valid() {
                return;
            }
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
                        .dst_set(*desc.descriptor.raw())
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
                        .dst_set(*desc.descriptor.raw())
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .build()
                })
                .for_each(|x| writes.push(x));

            unsafe { device.update_descriptor_sets(&writes, &[]) };
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
