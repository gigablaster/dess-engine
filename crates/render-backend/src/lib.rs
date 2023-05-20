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

mod buffer;
mod command_buffer;
mod device;
mod droplist;
mod error;
mod frame_context;
mod image;
mod instance;
mod memory;
mod physical_device;
mod pipeline;
mod render_pass;
mod staging;
mod swapchain;

pub use self::image::*;
pub use buffer::*;
pub use command_buffer::*;
pub use device::*;
pub use error::*;
pub use frame_context::*;
pub use instance::*;
pub use physical_device::*;
pub use pipeline::*;
pub use render_pass::*;
pub use swapchain::*;

pub trait FreeGpuResource {
    fn free(&self, device: &ash::Device);
}
