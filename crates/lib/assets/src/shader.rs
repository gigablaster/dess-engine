// Copyright (C) 2023 Vladimir Kuskov

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

use std::hash::Hash;

use ash::vk;
use siphasher::sip128::Hasher128;

use crate::{Asset, AssetLoad, ContentSource};

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct ShaderSource {
    pub stage: ShaderStage,
    pub path: String,
}

impl ContentSource for ShaderSource {
    fn get_ref(&self) -> crate::AssetRef {
        let mut hasher = siphasher::sip128::SipHasher::default();
        self.hash(&mut hasher);
        hasher.finish128().as_u128().into()
    }
}

#[derive(Debug)]
pub struct ShaderAsset {
    pub code: Vec<u8>,
}

impl Asset for ShaderAsset {
    fn to_bytes(&self) -> std::io::Result<Vec<u8>> {
        Ok(self.code.clone())
    }
}

impl AssetLoad for ShaderAsset {
    fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self {
            code: data.to_vec(),
        })
    }
}
