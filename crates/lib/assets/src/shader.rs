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

use std::{collections::HashMap, hash::Hash};

use serde::{Deserialize, Serialize};
use siphasher::sip128::Hasher128;
use speedy::{Readable, Writable};

use crate::{Asset, AssetRef, AssetRefProvider};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum ShaderStage {
    #[serde(rename = "vertex")]
    Vertex,
    #[serde(rename = "fragment")]
    Fragment,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShaderSource {
    pub source: String,
    pub stage: ShaderStage,
    pub defines: Option<Vec<String>>,
    pub specializations: Option<HashMap<SpecializationConstant, usize>>,
}

impl AssetRefProvider for ShaderSource {
    fn asset_ref(&self) -> AssetRef {
        let mut hasher = siphasher::sip128::SipHasher::default();
        self.source.hash(&mut hasher);
        self.stage.hash(&mut hasher);
        self.defines.hash(&mut hasher);
        self.specializations
            .iter()
            .for_each(|x| x.iter().for_each(|x| x.hash(&mut hasher)));

        AssetRef::from_u128(hasher.finish128().as_u128())
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum SpecializationConstant {
    #[serde(rename = "local_light_count")]
    LocalLightCount,
}

#[derive(Debug, Clone, Readable, Writable)]
pub struct ShaderAsset {
    pub stage: ShaderStage,
    pub specializations: Vec<(SpecializationConstant, u32)>,
    pub code: Vec<u8>,
}

impl Asset for ShaderAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("0d35d32c-8b62-41b2-8bdc-d329f06a5564");
    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}
