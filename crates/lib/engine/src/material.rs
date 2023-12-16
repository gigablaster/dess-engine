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

use dess_backend::vulkan::ImageHandle;
use smol_str::SmolStr;

use crate::{AssetCacheFns, AssetHandle, EngineAsset, Error};

/// Material contains effect and a per-material descriptor set
/// for every effect technique.
///
/// Pipelines aren't created at this stage, they belong to render pass.
#[derive(Debug, Default)]
pub struct RenderMaterial {
    pub images: HashMap<SmolStr, AssetHandle<ImageHandle>>,
}

impl EngineAsset for RenderMaterial {
    fn is_ready<T: AssetCacheFns>(&self, asset_cache: &T) -> bool {
        self.images
            .iter()
            .all(|(_, image)| asset_cache.is_image_loaded(*image))
    }

    fn resolve<T: AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), Error> {
        let mut images = HashMap::new();
        for (name, image) in self.images.iter() {
            let image = asset_cache.resolve_image(*image)?;
            images.insert(name, image);
        }
        Ok(())
    }
}
