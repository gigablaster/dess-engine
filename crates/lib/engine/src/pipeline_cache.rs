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

use dess_backend::{Device, RasterPipelineCreateDesc, RasterPipelineHandle};
use parking_lot::Mutex;

use crate::Error;

pub struct PipelineCacheBuilder<'a> {
    device: &'a Device,
    raster_pipelines: Mutex<HashMap<RasterPipelineCreateDesc, RasterPipelineHandle>>,
}
pub struct PipelineCache {
    raster_pipelines: HashMap<RasterPipelineCreateDesc, RasterPipelineHandle>,
}

impl<'a> PipelineCacheBuilder<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            raster_pipelines: Mutex::default(),
        }
    }

    pub fn register_raster_pipeline(&self, desc: RasterPipelineCreateDesc) -> Result<(), Error> {
        self.raster_pipelines
            .lock()
            .insert(desc, self.device.register_raster_pipeline(desc)?);
        Ok(())
    }

    pub fn build(self) -> PipelineCache {
        PipelineCache {
            raster_pipelines: self.raster_pipelines.into_inner(),
        }
    }
}

impl PipelineCache {
    pub fn get_raster_pipeline(
        &self,
        desc: &RasterPipelineCreateDesc,
    ) -> Option<RasterPipelineHandle> {
        self.raster_pipelines.get(desc).copied()
    }
}
