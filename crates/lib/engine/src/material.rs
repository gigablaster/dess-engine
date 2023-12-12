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
    hash::{Hash, Hasher},
};

use dess_backend::vulkan::{PipelineCreateDesc, ProgramHandle};
use smol_str::SmolStr;

use crate::{AssetCacheFns, EngineAsset, EngineAssetKey, Error};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EffectSource {
    pub path: String,
}

impl EngineAssetKey for EffectSource {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
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
