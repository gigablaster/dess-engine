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

use std::{path::Path, process::Command};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};
use turbosloth::{LazyWorker, RunContext};

use crate::{get_absolute_asset_path, Asset, Error};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum ShaderStage {
    #[serde(rename = "vertex")]
    Vertex,
    #[serde(rename = "fragment")]
    Fragment,
}

#[derive(Debug, Serialize, Deserialize, Clone, Readable, Writable)]
pub struct ShaderSource {
    pub source: String,
    pub stage: ShaderStage,
}

#[derive(Debug, Clone, Readable, Writable)]
pub struct ShaderAsset {
    pub stage: ShaderStage,
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

#[async_trait]
impl LazyWorker for ShaderSource {
    type Output = Result<ShaderAsset, Error>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let stage = match self.stage {
            ShaderStage::Vertex => "-fshader-stage=vert",
            ShaderStage::Fragment => "-fshader-stage=frag",
        };
        let cmd = Command::new("glslc")
            .arg(stage)
            .arg("--target-env=vulkan1.3")
            .arg(get_absolute_asset_path(Path::new(&self.source))?)
            .arg("-o")
            .arg("-")
            .output()
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        if cmd.status.success() {
            Ok(ShaderAsset {
                stage: self.stage,
                code: cmd.stdout,
            })
        } else {
            Err(Error::ProcessingFailed(
                String::from_utf8(cmd.stderr).unwrap(),
            ))
        }
    }
}
