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

mod descriptor_cache;
mod error;
mod staging;
mod uniforms;
pub mod vulkan;

use std::marker::PhantomData;

pub use descriptor_cache::*;
pub use error::*;
pub use staging::*;

pub type IndexType = u16;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index<T> {
    value: u32,
    _phantom: PhantomData<T>,
}

impl<T> Index<T> {
    pub fn new(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData::<T>,
        }
    }

    pub fn value(&self) -> u32 {
        self.value
    }
}

pub(crate) trait GpuResource {
    fn free(&self, device: &ash::Device);
}
