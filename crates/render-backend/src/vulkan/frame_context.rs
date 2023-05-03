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

use crate::BackendResult;

use super::{CommandBuffer, QueueFamily};

pub struct FrameContext {
    pub device: ash::Device,
    pub command_buffer: CommandBuffer,
    pub queue_family: QueueFamily,
}

impl FrameContext {
    pub fn new(device: &ash::Device, queue_family: &QueueFamily) -> BackendResult<Self> {
        Ok(Self {
            device: device.clone(),
            command_buffer: CommandBuffer::new(device, queue_family)?,
            queue_family: *queue_family,
        })
    }

    pub(crate) fn free(&mut self, device: &ash::Device) {
        self.command_buffer.free(device);
    }
}

