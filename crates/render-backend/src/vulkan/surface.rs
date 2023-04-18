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

use ash::{
    extensions::khr,
    vk::{self, Handle},
};
use sdl2::video::Window;

use crate::BackendResult;

use super::Instance;

pub struct Surface {
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) loader: khr::Surface,
}

impl Surface {
    pub fn create(instance: &Instance, window: &Window) -> BackendResult<Arc<Self>> {
        let surface = window.vulkan_create_surface(instance.raw.handle().as_raw() as usize)?;
        let surface = vk::SurfaceKHR::from_raw(surface);
        let loader = khr::Surface::new(&instance.entry, &instance.raw);

        Ok(Arc::new(Self {
            raw: surface,
            loader,
        }))
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.raw, None) };
    }
}
