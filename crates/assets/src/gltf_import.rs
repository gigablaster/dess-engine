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

use std::path::PathBuf;

use async_trait::async_trait;
use turbosloth::{Lazy, LazyWorker, RunContext};

use crate::gpumodel::GpuModel;

pub struct LoadGltf {
    path: PathBuf,
}

impl LoadGltf {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

pub struct LoadedGltf {
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
}

#[async_trait]
impl LazyWorker for LoadGltf {
    type Output = anyhow::Result<LoadedGltf>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        let (document, buffers, images) = gltf::import(self.path)?;

        Ok(LoadedGltf {
            document,
            buffers,
            images,
        })
    }
}

pub struct CreateGpuModel {
    pub gltf: Lazy<LoadedGltf>,
}

impl CreateGpuModel {
    pub fn new(gltf: Lazy<LoadedGltf>) -> Self {
        Self { gltf }
    }
}

#[async_trait]
impl LazyWorker for CreateGpuModel {
    type Output = anyhow::Result<GpuModel>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let gltf = self.gltf.eval(&ctx).await?;

        todo!()
    }
}
