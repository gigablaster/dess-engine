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

    pub fn get_or_register_raster_pipeline(
        &self,
        desc: RasterPipelineCreateDesc,
    ) -> Result<RasterPipelineHandle, Error> {
        if let Some(handle) = self.raster_pipelines.get(&desc) {
            Ok(*handle)
        } else {
            let mut handles = self.raster_pipelines_builder.lock();
            if let Some(handle) = handles.get(&desc) {
                Ok(*handle)
            } else {
                let handle = self.device.register_raster_pipeline(desc)?;
                handles.insert(desc, handle);
                Ok(handle)
            }
        }
    }

    pub fn sync(&mut self) {
        let mut pipelines = self.raster_pipelines_builder.lock();
        pipelines.drain().for_each(|(desc, handle)| {
            self.raster_pipelines.insert(desc, handle);
        });
    }

    pub fn get_raster_pipeline(
        &self,
        desc: &RasterPipelineCreateDesc,
    ) -> Option<RasterPipelineHandle> {
        if let Some(handle) = self.raster_pipelines.get(desc) {
            Some(*handle)
        } else {
            if let Some(handle) = self.raster_pipelines_builder.lock().get(desc) {
                Some(*handle)
            } else {
                None
            }
        }
    }
}
