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

use dess_assets::{ShaderStage, SpecializationConstant};
use serde::{Deserialize, Serialize};

use crate::ImagePurpose;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageDesc {
    pub source: String,
    #[serde(flatten)]
    pub purpose: ImagePurpose,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShaderSource {
    pub source: String,
    pub stage: ShaderStage,
    pub defines: Option<Vec<String>>,
    pub specializations: Option<HashMap<SpecializationConstant, usize>>,
}

/// Asset to put in bundle
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename = "asset")]
pub enum BundledAsset {
    #[serde(rename = "image")]
    Image(ImageDesc),
    #[serde(rename = "model")]
    Model(String),
    #[serde(rename = "shader")]
    Shader(ShaderSource),
}

/// Single bundle description
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "bundle")]
pub struct BundleDesc {
    /// Bundle name
    name: String,
    /// Asset name -> asset map
    assets: HashMap<String, BundledAsset>,
}

impl BundleDesc {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_ascii_lowercase(),
            assets: HashMap::default(),
        }
    }

    pub fn insert(&mut self, name: &str, asset: BundledAsset) {
        self.assets.insert(name.to_ascii_lowercase(), asset);
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn asset(&self, name: &str) -> Option<&BundledAsset> {
        self.assets.get(name)
    }

    pub fn assets(&self) -> Vec<(String, BundledAsset)> {
        self.assets
            .iter()
            .map(|(name, asset)| (name.clone(), asset.clone()))
            .collect()
    }
}
