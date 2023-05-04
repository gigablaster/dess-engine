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
    slice,
    sync::{Arc, Mutex},
};

use arrayvec::ArrayVec;
use ash::vk;
use log::info;

use crate::BackendResult;

use super::{Device, ImageDesc};

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Debug, Clone, Copy)]
pub struct RenderPassAttachmentDesc {
    pub format: vk::Format,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

impl RenderPassAttachmentDesc {
    pub fn new(format: vk::Format) -> Self {
        Self {
            format,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            samples: vk::SampleCountFlags::TYPE_1,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::GENERAL,
        }
    }

    pub fn initial_layout(mut self, value: vk::ImageLayout) -> Self {
        self.initial_layout = value;
        self
    }

    pub fn final_layout(mut self, value: vk::ImageLayout) -> Self {
        self.final_layout = value;
        self
    }

    pub fn garbage_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    pub fn clear_input(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
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

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct AttachmentDesc {
    pub usage_flags: vk::ImageUsageFlags,
    pub create_flags: vk::ImageCreateFlags,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct FramebufferCacheKey {
    pub dims: [u32; 2],
    pub attachments: ArrayVec<AttachmentDesc, MAX_ATTACHMENTS>,
}

impl FramebufferCacheKey {
    pub fn new<'a>(
        dims: [u32; 2],
        color_attachments: &[&'a ImageDesc],
        depth_stencil_attachment: Option<&'a ImageDesc>,
    ) -> Self {
        let attachments = color_attachments
            .iter()
            .chain(depth_stencil_attachment.as_ref().into_iter())
            .map(|attachment| AttachmentDesc {
                usage_flags: attachment.usage,
                create_flags: attachment.flags,
            })
            .collect();

        Self { dims, attachments }
    }
}

pub struct FramebufferCache {
    render_pass: vk::RenderPass,
    entries: Mutex<HashMap<FramebufferCacheKey, vk::Framebuffer>>,
    attachment_descs: ArrayVec<RenderPassAttachmentDesc, MAX_ATTACHMENTS>,
}

impl FramebufferCache {
    pub fn new(
        render_pass: vk::RenderPass,
        color_attachments: &[RenderPassAttachmentDesc],
        depth_stencil_attachment: Option<RenderPassAttachmentDesc>,
    ) -> Self {
        let mut attachment_descs = ArrayVec::new();
        attachment_descs
            .try_extend_from_slice(color_attachments)
            .unwrap();
        if let Some(depth_stencil) = depth_stencil_attachment {
            attachment_descs.push(depth_stencil);
        }

        Self {
            render_pass,
            entries: Default::default(),
            attachment_descs,
        }
    }

    pub fn get_or_create(
        &self,
        device: &ash::Device,
        key: FramebufferCacheKey,
    ) -> BackendResult<vk::Framebuffer> {
        let mut entries = self.entries.lock().unwrap();

        if let Some(fbo) = entries.get(&key) {
            Ok(*fbo)
        } else {
            let fbo = self.create_fbo(device, &key)?;
            entries.insert(key, fbo);
            Ok(fbo)
        }
    }

    pub fn clear(&self, device: &ash::Device) {
        let mut entries = self.entries.lock().unwrap();
        entries.iter().for_each(|(_, fbo)| {
            unsafe { device.destroy_framebuffer(*fbo, None) };
        });
        entries.clear();
    }

    fn create_fbo(
        &self,
        device: &ash::Device,
        key: &FramebufferCacheKey,
    ) -> BackendResult<vk::Framebuffer> {
        let [width, height] = key.dims;
        let mut formats = ArrayVec::<_, MAX_ATTACHMENTS>::new();
        let attachments = self
            .attachment_descs
            .iter()
            .zip(key.attachments.iter())
            .map(|(attachment_desc, image_desc)| {
                let index = formats.len();
                formats.push(attachment_desc.format);
                vk::FramebufferAttachmentImageInfo::builder()
                    .width(width as _)
                    .height(height as _)
                    .flags(image_desc.create_flags)
                    .layer_count(1)
                    .view_formats(slice::from_ref(&formats[index]))
                    .usage(image_desc.usage_flags)
                    .build()
            })
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();
        let mut imagess_desc = vk::FramebufferAttachmentsCreateInfo::builder()
            .attachment_image_infos(&attachments)
            .build();

        let mut fbo_create_info = vk::FramebufferCreateInfo::builder()
            .flags(vk::FramebufferCreateFlags::IMAGELESS)
            .render_pass(self.render_pass)
            .width(width as _)
            .height(height as _)
            .layers(1)
            .push_next(&mut imagess_desc)
            .build();

        fbo_create_info.attachment_count = attachments.len() as _;

        let fbo = unsafe { device.create_framebuffer(&fbo_create_info, None) }?;

        info!("DONE!");

        Ok(fbo)
    }
}
#[derive(Debug)]
pub struct RenderPassDesc<'a> {
    pub color_attachments: &'a [RenderPassAttachmentDesc],
    pub depth_attachment: Option<RenderPassAttachmentDesc>,
}

pub struct RenderPass {
    device: Arc<Device>,
    pub(crate) raw: vk::RenderPass,
    pub framebuffer_cache: FramebufferCache,
}

impl RenderPass {
    pub fn new(device: &Arc<Device>, desc: &RenderPassDesc) -> BackendResult<Arc<RenderPass>> {
        let attachment_refs = desc
            .color_attachments
            .iter()
            .map(|a| a.build(a.initial_layout, a.final_layout))
            .chain(
                desc.depth_attachment
                    .as_ref()
                    .map(|a| a.build(a.initial_layout, a.final_layout)),
            )
            .collect::<Vec<_>>();

        let color_attachment_refs = (0..desc.color_attachments.len() as u32)
            .map(|index| vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect::<Vec<_>>();

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: desc.color_attachments.len() as u32,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        };

        let mut subpass_description = vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        if desc.depth_attachment.is_some() {
            subpass_description =
                subpass_description.depth_stencil_attachment(&depth_attachment_ref);
        }

        let subpass_description = subpass_description.build();

        let subpasses = [subpass_description];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_refs)
            .subpasses(&subpasses)
            .build();

        let render_pass = unsafe {
            device
                .raw
                .create_render_pass(&render_pass_create_info, None)
        }?;

        let framebuffer_cache =
            FramebufferCache::new(render_pass, desc.color_attachments, desc.depth_attachment);

        Ok(Arc::new(Self {
            device: device.clone(),
            raw: render_pass,
            framebuffer_cache,
        }))
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        self.framebuffer_cache.clear(&self.device.raw);
        unsafe { self.device.raw.destroy_render_pass(self.raw, None) };
    }
}
