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

use std::collections::HashMap;

use ash::vk;
use dess_backend::{
    vulkan::{
        DescriptorHandle, ImageHandle, PipelineCreateDesc, ProgramHandle, PER_MATERIAL_BINDING_SLOT,
    },
    BackendResultExt,
};
use smol_str::SmolStr;

use crate::{AssetCacheFns, AssetHandle, EngineAsset, Error};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EffectSource {
    pub path: String,
}

#[derive(Debug, Clone, Hash)]
pub struct RenderEffectTechinque {
    pub program: ProgramHandle,
    pub pipeline_desc: PipelineCreateDesc,
}

/// Effect
///
/// Set of named pipelines, created from program and pipeline descriptor.
/// Used by materials to get pipelines for different passes.
///
/// Each pipeline might or might not use different program.
#[derive(Debug, Default)]
pub struct RenderEffect {
    pub techinques: HashMap<SmolStr, RenderEffectTechinque>,
}

impl EngineAsset for RenderEffect {
    fn is_ready<T: AssetCacheFns>(&self, _asset_cache: &T) -> bool {
        true
    }

    fn resolve<T: AssetCacheFns>(&mut self, _asset_cache: &T) -> Result<(), Error> {
        Ok(())
    }
}

/// Material contains effect and a per-material descriptor set
/// for every effect technique.
///
/// Pipelines aren't created at this stage, they belong to render pass.
#[derive(Debug, Default)]
pub struct RenderMaterial {
    pub effect: AssetHandle<RenderEffect>,
    pub uniform: Vec<u8>,
    pub images: HashMap<SmolStr, AssetHandle<ImageHandle>>,
    pub descriptors: HashMap<SmolStr, DescriptorHandle>,
}

impl EngineAsset for RenderMaterial {
    fn is_ready<T: AssetCacheFns>(&self, asset_cache: &T) -> bool {
        asset_cache.is_effect_loaded(self.effect)
            && self
                .images
                .iter()
                .all(|(_, image)| asset_cache.is_image_loaded(*image))
    }

    fn resolve<T: AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), Error> {
        let effect = asset_cache.resolve_effect(self.effect)?;
        let mut images = HashMap::new();
        for (name, image) in self.images.iter() {
            let image = asset_cache.resolve_image(*image)?;
            images.insert(name, image);
        }
        self.descriptors.clear();
        asset_cache.render_device().with_descriptors(|mut ctx| {
            for (name, tech) in effect.techinques.iter() {
                let ds = ctx.from_program(tech.program, PER_MATERIAL_BINDING_SLOT)?;
                if !self.uniform.is_empty() {
                    ctx.bind_uniform_raw_by_name(ds, "material", &self.uniform)
                        .ignore_missing()?;
                }
                for (name, image) in images.iter() {
                    ctx.bind_image_by_name(
                        ds,
                        name,
                        *image.as_ref(),
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )
                    .ignore_missing()?;
                }
                self.descriptors.insert(name.to_owned(), ds);
            }
            Ok(())
        })?;
        Ok(())
    }
}
