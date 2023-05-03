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

use arrayvec::ArrayVec;
use ash::vk::{self, CommandBufferUsageFlags, FenceCreateFlags};

use crate::BackendResult;

use super::{
    Device, FramebufferCacheKey, Image, ImageViewDesc, QueueFamily, RenderPass, MAX_ATTACHMENTS,
    MAX_COLOR_ATTACHMENTS,
};

pub struct CommandBuffer {
    device: ash::Device,
    pub raw: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub pool: vk::CommandPool,
}

impl CommandBuffer {
    pub(crate) fn new(device: &ash::Device, queue_family: &QueueFamily) -> BackendResult<Self> {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family.index)
            .build();

        let pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

        let command_buffer_allocation_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();

        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocation_info)? }[0];

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED)
            .build();

        let fence = unsafe { device.create_fence(&fence_create_info, None)? };

        Ok(Self {
            device: device.clone(),
            raw: command_buffer,
            fence,
            pool,
        })
    }

    pub fn begin(&self) -> BackendResult<CommandBufferGenerator> {
        unsafe {
            self.device
                .reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())
        }?;
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe { self.device.begin_command_buffer(self.raw, &begin_info) }?;

        Ok(CommandBufferGenerator {
            device: &self.device,
            cb: self.raw,
        })
    }

    pub(crate) fn free(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

pub struct CommandBufferGenerator<'a> {
    device: &'a ash::Device,
    cb: vk::CommandBuffer,
}

impl<'a> Drop for CommandBufferGenerator<'a> {
    fn drop(&mut self) {
        unsafe { self.device.end_command_buffer(self.cb) }.unwrap();
    }
}

impl<'a> CommandBufferGenerator<'a> {
    pub fn begin_pass(
        &self,
        dims: [u32; 2],
        render_pass: &RenderPass,
        color_attachments: &[&Image],
        depth_attachment: Option<Image>,
    ) {
        let color_attachment_descs = color_attachments
            .iter()
            .map(|image| &image.desc)
            .collect::<ArrayVec<_, MAX_COLOR_ATTACHMENTS>>();
        let image_attachments = color_attachments
            .iter()
            .map(|image| {
                let desc = ImageViewDesc::default()
                    .level_count(1)
                    .base_mip_level(0)
                    .format(image.desc.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .aspect_mask(vk::ImageAspectFlags::COLOR);
                image.get_or_create_view(self.device, desc).unwrap()
            })
            .chain(depth_attachment.as_ref().map(|image| {
                let desc = ImageViewDesc::default()
                    .level_count(1)
                    .base_mip_level(0)
                    .format(image.desc.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .aspect_mask(vk::ImageAspectFlags::DEPTH);
                image.get_or_create_view(self.device, desc).unwrap()
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let key = FramebufferCacheKey::new(
            dims,
            &color_attachment_descs,
            depth_attachment.map(|image| image.desc).as_ref(),
        );

        let mut pass_attachments_info = vk::RenderPassAttachmentBeginInfoKHR::builder()
            .attachments(&image_attachments)
            .build();
        let [width, height] = dims;
        let framebuffer = render_pass
            .framebuffer_cache
            .get_or_create(&self.device, key)
            .unwrap();

        let begin_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: width as _,
                    height: height as _,
                },
            })
            .push_next(&mut pass_attachments_info)
            .build();

        unsafe {
            self.device.cmd_begin_render_pass(
                self.cb,
                &begin_pass_info,
                vk::SubpassContents::INLINE,
            )
        };
    }

    pub fn clear(
        &self,
        dims: [u32; 2],
        colors: &[vk::ClearColorValue],
        depth: Option<vk::ClearDepthStencilValue>,
    ) {
        let attachments = colors
            .iter()
            .enumerate()
            .map(|(index, value)| {
                vk::ClearAttachment::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(index as _)
                    .clear_value(vk::ClearValue { color: *value })
                    .build()
            })
            .chain(depth.as_ref().map(|depth| {
                vk::ClearAttachment::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .clear_value(vk::ClearValue {
                        depth_stencil: *depth,
                    })
                    .build()
            }))
            .collect::<ArrayVec<_, MAX_ATTACHMENTS>>();

        let rect = vk::ClearRect::builder()
            .rect(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: dims[0],
                    height: dims[1],
                },
            })
            .base_array_layer(0)
            .layer_count(1)
            .build();
        unsafe {
            self.device
                .cmd_clear_attachments(self.cb, &attachments, &[rect])
        };
    }

    pub fn end_pass(&self) {
        unsafe { self.device.cmd_end_render_pass(self.cb) };
    }
}

impl Device {
    pub fn create_command_buffer(&self) -> BackendResult<CommandBuffer> {
        CommandBuffer::new(&self.raw, &self.graphics_queue.family)
    }
}
