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

use arrayvec::ArrayVec;
use ash::vk;

use crate::BackendResult;

use super::{Device, Image, ImageViewDesc, RenderPass};

pub(crate) const MAX_COLOR_ATTACHMENTS: usize = 8;
pub(crate) const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

pub struct Framebuffer {
    device: Arc<Device>,
    pub raw: vk::Framebuffer,
    pub views: ArrayVec<vk::ImageView, MAX_ATTACHMENTS>,
    pub extent: vk::Extent2D,
}

impl Framebuffer {
    pub fn new(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPass>,
        color_attachments: &[&Arc<Image>],
        depth_attachment: Option<&Arc<Image>>,
    ) -> BackendResult<Self> {
        let mut attachments = color_attachments
            .iter()
            .map(|image| Self::create_attachment_image_info(image))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        if let Some(depth_attachment) = depth_attachment {
            attachments.push(Self::create_attachment_image_info(depth_attachment));
        }

        let views = color_attachments
            .iter()
            .map(|image| {
                let desc = ImageViewDesc::default()
                    .format(image.desc.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .aspect_mask(vk::ImageAspectFlags::COLOR);
                device.create_image_view(image, &desc).unwrap()
            })
            .chain(depth_attachment.as_ref().map(|depth| {
                let desc = ImageViewDesc::default()
                    .format(depth.desc.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .aspect_mask(vk::ImageAspectFlags::DEPTH);
                device.create_image_view(depth, &desc).unwrap()
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let fbo_desc = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.raw)
            .width(attachments[0].width)
            .height(attachments[0].height)
            .attachments(&views)
            .layers(1)
            .build();

        let fbo = unsafe { device.raw.create_framebuffer(&fbo_desc, None) }?;

        let extent = vk::Extent2D {
            width: attachments[0].width,
            height: attachments[0].height,
        };

        Ok(Self {
            device: device.clone(),
            raw: fbo,
            views,
            extent,
        })
    }

    fn create_attachment_image_info(image: &Image) -> vk::FramebufferAttachmentImageInfoKHR {
        vk::FramebufferAttachmentImageInfoKHR::builder()
            .width(image.desc.extent[0])
            .height(image.desc.extent[1])
            .flags(image.desc.flags)
            .layer_count(1)
            .view_formats(&[image.desc.format])
            .usage(image.desc.usage)
            .build()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.raw.destroy_framebuffer(self.raw, None) };
    }
}
