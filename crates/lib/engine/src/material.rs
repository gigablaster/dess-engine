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
    fmt::Display,
    hash::{Hash, Hasher},
    mem,
    ptr::copy_nonoverlapping,
    slice,
};

use arrayvec::ArrayVec;
use ash::vk;
use dess_assets::ImageSource;
use dess_backend::vulkan::{
    DescriptorHandle, ImageHandle, ProgramHandle, PER_MATERIAL_BINDING_SLOT,
};
use smol_str::SmolStr;

use crate::{AssetHandle, EngineAsset, EngineAssetKey, Error};

#[derive(Debug, Default, Clone)]
pub struct MaterialSource {
    images: HashMap<SmolStr, ImageSource>,
    uniform_data: Vec<u8>,
}

impl Display for MaterialSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Material(")?;
        for (image, source) in self.images.iter() {
            write!(f, "{}=>{}, ", image, source)?;
        }
        write!(f, ")")
    }
}

impl EngineAssetKey for MaterialSource {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        for (slot, source) in self.images.iter() {
            slot.hash(&mut hasher);
            source.hash(&mut hasher);
        }
        self.uniform_data.hash(&mut hasher);
        hasher.finish()
    }
}

impl MaterialSource {
    pub fn add_image(&mut self, name: &str, source: ImageSource) {
        self.images.insert(name.into(), source);
    }

    pub fn set_uniform<T: Sized + Copy>(&mut self, uniform: &T) {
        let size = mem::size_of::<T>();
        let ptr = slice::from_ref(uniform).as_ptr() as *const u8;
        self.uniform_data.resize(size, 0u8);
        unsafe { copy_nonoverlapping(ptr, self.uniform_data.as_mut_ptr(), size) }
    }
}

#[derive(Debug)]
pub struct MeshRenderMaterial {
    ds: DescriptorHandle,
    program: AssetHandle<ProgramHandle>,
    images: HashMap<SmolStr, AssetHandle<ImageHandle>>,
    uniform_data: Vec<u8>,
}

const MAX_TEXTURES: usize = 16;

impl EngineAsset for MeshRenderMaterial {
    fn is_ready(&self, asset_cache: &crate::AssetCache) -> bool {
        let mut images = ArrayVec::<AssetHandle<ImageHandle>, MAX_TEXTURES>::new();
        for (_, handle) in self.images.iter() {
            images.push(*handle);
        }
        asset_cache.are_images_loaded(&images) && asset_cache.is_program_loaded(self.program)
    }

    fn resolve(&mut self, asset_cache: &crate::AssetCache) -> Result<(), Error> {
        let mut images = ArrayVec::<_, MAX_TEXTURES>::new();
        for (name, handle) in &self.images {
            images.push((name.clone(), asset_cache.resolve_image(*handle)?))
        }
        let program = asset_cache.resolve_program(self.program)?;
        asset_cache.device().with_descriptors(|mut ctx| {
            let ds = ctx.create(program.as_ref().clone(), PER_MATERIAL_BINDING_SLOT)?;
            for (name, image) in images.iter() {
                ctx.bind_image_by_name(
                    ds,
                    name,
                    *image.as_ref(),
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )?;
            }
            ctx.bind_uniform_raw_by_name(ds, "material", &self.uniform_data)?;

            self.ds = ds;
            Ok(())
        })?;
        Ok(())
    }
}
