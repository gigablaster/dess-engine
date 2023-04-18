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

use std::sync::Arc;

use ash::vk;

use crate::BackendResult;

use super::Device;

#[derive(Debug, Clone, Copy)]
pub struct RenderPassAttachmentDesc {
    pub format: vk::Format,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub samples: vk::SampleCountFlags,
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

#[derive(Debug)]
pub struct RenderPassDesc<'a> {
    pub color_attachments: &'a [RenderPassAttachmentDesc],
    pub depth_attachment: Option<RenderPassAttachmentDesc>,
}

pub struct RenderPass {
    device: Arc<Device>,
    pub raw: vk::RenderPass,
    pub color_attachments: Vec<vk::AttachmentReference>,
    pub depth_attachment: Option<vk::AttachmentReference>,
}

impl RenderPass {
    pub fn new(device: &Arc<Device>, desc: &RenderPassDesc) -> BackendResult<Arc<RenderPass>> {
        let attachment_refs = desc
            .color_attachments
            .iter()
            .map(|a| {
                a.build(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
            })
            .chain(desc.depth_attachment.as_ref().map(|a| {
                a.build(
                    vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                )
            }))
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

        Ok(Arc::new(Self {
            device: device.clone(),
            raw: render_pass,
            color_attachments: color_attachment_refs,
            depth_attachment: if desc.depth_attachment.is_some() {
                Some(depth_attachment_ref)
            } else {
                None
            },
        }))
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.raw.destroy_render_pass(self.raw, None) };
    }
}
