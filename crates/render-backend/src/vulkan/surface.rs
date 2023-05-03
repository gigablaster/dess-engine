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

use ash::{
    extensions::khr,
    vk::{self, Handle},
};
use sdl2::video::Window;

use crate::BackendResult;

use super::Instance;

pub struct Surface<'a> {
    pub(crate) window: &'a Window,
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) loader: khr::Surface,
}

impl<'a> Surface<'a> {
    pub fn create(instance: &Instance, window: &'a Window) -> BackendResult<Self> {
        let surface = window.vulkan_create_surface(instance.raw.handle().as_raw() as usize)?;
        let surface = vk::SurfaceKHR::from_raw(surface);
        let loader = khr::Surface::new(&instance.entry, &instance.raw);

        Ok(Self {
            window,
            raw: surface,
            loader,
        })
    }
}

impl<'a> Drop for Surface<'a> {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
