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

use dess_backend::vulkan::{DescriptorHandle, ImageHandle, PipelineCreateDesc, ProgramHandle};
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
#[derive(Debug)]
pub struct RenderMaterial {
    effect: AssetHandle<RenderEffect>,
    images: HashMap<String, AssetHandle<ImageHandle>>,
    descriptors: HashMap<String, DescriptorHandle>,
}
