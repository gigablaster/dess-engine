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

use std::{hash::Hash, path::Path};

use bytes::Bytes;
use dess_backend::ShaderStage;
use siphasher::sip128::Hasher128;

use crate::{Asset, AssetLoad, ContentSource};

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct ShaderSource {
    pub stage: ShaderStage,
    pub path: String,
}

impl ShaderSource {
    pub fn vertex<P: AsRef<Path>>(path: P) -> Self {
        Self {
            stage: ShaderStage::Vertex,
            path: path.as_ref().to_str().unwrap().to_owned(),
        }
    }

    pub fn fragment<P: AsRef<Path>>(path: P) -> Self {
        Self {
            stage: ShaderStage::Fragment,
            path: path.as_ref().to_str().unwrap().to_owned(),
        }
    }

    pub fn compute<P: AsRef<Path>>(path: P) -> Self {
        Self {
            stage: ShaderStage::Compute,
            path: path.as_ref().to_str().unwrap().to_owned(),
        }
    }
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
    pub code: Bytes,
}

impl Asset for ShaderAsset {
    fn to_bytes(&self) -> std::io::Result<Bytes> {
        Ok(Bytes::copy_from_slice(&self.code))
    }
}

impl AssetLoad for ShaderAsset {
    fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        Ok(Self {
            code: Bytes::copy_from_slice(data),
        })
    }
}
