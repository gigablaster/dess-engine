use std::collections::HashMap;

use dess_assets::GpuShaderStage;
use serde::{Deserialize, Serialize};

use crate::ImagePurpose;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShaderDesc {
    pub source: String,
    #[serde(flatten)]
    pub ty: GpuShaderStage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageDesc {
    pub source: String,
    #[serde(flatten)]
    pub purpose: ImagePurpose,
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
    Shader(ShaderDesc),
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
