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
    collections::HashMap,
    sync::{Arc, Mutex},
};

use arrayvec::ArrayVec;
use ash::vk;

use crate::BackendResult;

use super::{Device, Image, ImageViewDesc};

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Default)]
pub struct RenderPassLayout<'a> {
    pub color_attachments: &'a [RenderPassAttachmentDesc],
    pub depth_attachment: Option<RenderPassAttachmentDesc>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RenderPassAttachmentDesc {
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    samples: vk::SampleCountFlags,
}

impl RenderPassAttachmentDesc {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    pub fn garbage_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    pub fn clear_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    pub fn discard_output(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    pub fn build(
        self,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::AttachmentDescription {
        vk::AttachmentDescription {
            format: self.format,
            samples: self.samples,
            load_op: self.load_op,
            store_op: self.store_op,
            initial_layout,
            final_layout,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FboCacheKey {
    dims: [u32; 2],
    attachments: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
}

impl FboCacheKey {
    pub fn new(color_attachments: &[&Image], depth_attachment: Option<&Image>) -> Self {
        let dims = color_attachments
            .iter()
            .chain(depth_attachment.iter())
            .find_map(|image| Some(image.desc.extent));
        if let Some(dims) = dims {
            let attachments = color_attachments
                .iter()
                .map(|image| {
                    let desc = ImageViewDesc::default()
                        .format(image.desc.format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .level_count(1)
                        .base_mip_level(0)
                        .aspect_mask(vk::ImageAspectFlags::COLOR);
                    image.get_or_create_view(desc).unwrap()
                })
                .chain(depth_attachment.iter().map(|image| {
                    let desc = ImageViewDesc::default()
                        .format(image.desc.format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .level_count(1)
                        .base_mip_level(0)
                        .aspect_mask(vk::ImageAspectFlags::DEPTH);
                    image.get_or_create_view(desc).unwrap()
                }))
                .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

            Self { dims, attachments }
        } else {
            panic!("Can't create FboCacheKey without attachments");
        }
    }
}

pub(crate) struct FboCache {
    device: Arc<Device>,
    entries: Mutex<HashMap<FboCacheKey, vk::Framebuffer>>,
    render_pass: vk::RenderPass,
}

impl FboCache {
    pub fn new(device: &Arc<Device>, render_pass: vk::RenderPass) -> Self {
        Self {
            device: device.clone(),
            entries: Default::default(),
            render_pass,
        }
    }

    pub fn get_or_create(&self, key: FboCacheKey) -> BackendResult<vk::Framebuffer> {
        let mut entries = self.entries.lock().unwrap();
        if let Some(fbo) = entries.get(&key) {
            Ok(*fbo)
        } else {
            let fbo = self.create_fbo(&key)?;
            entries.insert(key, fbo);
            Ok(fbo)
        }
    }

    fn create_fbo(&self, key: &FboCacheKey) -> BackendResult<vk::Framebuffer> {
        let attachments = key
            .attachments
            .iter()
            .copied()
            .map(|attachment| attachment)
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let fbo_desc = vk::FramebufferCreateInfo::builder()
            .render_pass(self.render_pass)
            .width(key.dims[0] as _)
            .height(key.dims[1] as _)
            .layers(1)
            .attachments(&attachments)
            .build();

        Ok(unsafe { self.device.raw.create_framebuffer(&fbo_desc, None) }?)
    }

    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries
            .drain()
            .for_each(|(_, fbo)| unsafe { self.device.raw.destroy_framebuffer(fbo, None) });
    }
}

impl Drop for FboCache {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct RenderPass {
    device: Arc<Device>,
    pub(crate) raw: vk::RenderPass,
    pub(crate) fbo_cache: FboCache,
}

impl RenderPass {
    pub fn new(device: &Arc<Device>, layout: RenderPassLayout) -> BackendResult<Self> {
        let attachments = layout
            .color_attachments
            .iter()
            .map(|desc| {
                desc.build(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
            })
            .chain(layout.depth_attachment.iter().map(|desc| {
                desc.build(
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                )
            }))
            .collect::<Vec<_>>();

        let color_attacmnet_refs = (0..layout.color_attachments.len())
            .map(|index| vk::AttachmentReference {
                attachment: index as _,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();

        let mut subpass_desc = vk::SubpassDescription::builder()
            .color_attachments(&color_attacmnet_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let depth_attachment_ref = if layout.depth_attachment.is_some() {
            Some(vk::AttachmentReference {
                attachment: color_attacmnet_refs.len() as _,
                layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            })
        } else {
            None
        };

        if let Some(depth_attachment_ref) = depth_attachment_ref.as_ref() {
            subpass_desc = subpass_desc.depth_stencil_attachment(depth_attachment_ref);
        }

        let subpasses = [subpass_desc.build(); 1];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .build();

        let render_pass = unsafe { device.raw.create_render_pass(&render_pass_info, None) }?;

        Ok(Self {
            device: device.clone(),
            raw: render_pass,
            fbo_cache: FboCache::new(device, render_pass),
        })
    }

    pub fn clear_fbos(&self) {
        self.fbo_cache.clear();
    }
}
