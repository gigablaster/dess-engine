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

use std::{collections::HashMap, mem};

use dess_backend::{Device, RasterPipelineCreateDesc, RasterPipelineHandle};
use parking_lot::Mutex;

use crate::Error;

pub struct PipelineCache<'a> {
    device: &'a Device,
    raster_pipelines_builder: Mutex<HashMap<RasterPipelineCreateDesc, RasterPipelineHandle>>,
    raster_pipelines: HashMap<RasterPipelineCreateDesc, RasterPipelineHandle>,
}

impl<'a> PipelineCache<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            raster_pipelines_builder: Mutex::default(),
            raster_pipelines: HashMap::default(),
        }
    }

    pub fn register_raster_pipeline(&self, desc: RasterPipelineCreateDesc) -> Result<(), Error> {
        self.raster_pipelines_builder
            .lock()
            .insert(desc, self.device.register_raster_pipeline(desc)?);
        Ok(())
    }

    pub fn sync(&mut self) {
        self.raster_pipelines = mem::take(&mut self.raster_pipelines_builder.lock());
    }

    pub fn get_raster_pipeline(
        &self,
        desc: &RasterPipelineCreateDesc,
    ) -> Option<RasterPipelineHandle> {
        self.raster_pipelines.get(desc).copied()
    }
}
