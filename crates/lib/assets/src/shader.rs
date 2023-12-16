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

use std::path::{Path, PathBuf};

use ash::vk;
use serde::{Deserialize, Serialize};
use shader_prepper::SourceChunk;
use speedy::{Readable, Writable};
use uuid::{uuid, Uuid};

use crate::{get_absolute_asset_path, get_relative_asset_path, read_to_end, Asset, Error};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
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

#[derive(Debug, Writable, Hash, Readable)]
pub struct ShaderAsset {
    pub stage: ShaderStage,
    pub code: Vec<u8>,
    pub entry: String,
}

impl Asset for ShaderAsset {
    const TYPE_ID: Uuid = uuid!("8ef54231-2b31-4826-8b49-d3b2947764b3");

    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}

struct IncludeProvider {}

impl shader_prepper::IncludeProvider for IncludeProvider {
    type IncludeContext = PathBuf;

    fn get_include(
        &mut self,
        path: &str,
        context: &Self::IncludeContext,
    ) -> Result<(String, Self::IncludeContext), failure::Error> {
        let path = get_relative_asset_path(context.join(Path::new(path)))?;
        let parent = path.parent().unwrap().to_owned();
        let data = read_to_end(get_absolute_asset_path(path)?)?;

        Ok((String::from_utf8(data)?, parent))
    }
}

pub fn import_shader(path: &str) -> Result<Vec<SourceChunk>, Error> {
    let mut include_provider = IncludeProvider {};
    let chunks =
        shader_prepper::process_file(&path, &mut include_provider, Path::new("").to_path_buf())
            .map_err(|x| Error::ImportFailed(x.to_string()))?;
    Ok(chunks)
}

pub fn compile_shader(
    name: &str,
    chunks: &[SourceChunk],
    stage: ShaderStage,
) -> Result<ShaderAsset, Error> {
    let mut source = String::new();
    chunks.into_iter().for_each(|x| source += &x.source);
    let profile = match stage {
        ShaderStage::Vertex => "vs_6_4",
        ShaderStage::Fragment => "ps_6_4",
        ShaderStage::Compute => "cs_6_4",
    };
    let code = hassle_rs::compile_hlsl(
        &name,
        &source,
        "main",
        profile,
        &[
            "-spirv",
            "-fspv-target-env=vulkan1.3",
            "-WX",
            "-Ges",
            "-HV 2021",
        ],
        &[],
    )
    .map_err(|x| Error::ProcessingFailed(x.to_string()))?;
    Ok(ShaderAsset {
        stage,
        code,
        entry: "main".to_owned(),
    })
}
