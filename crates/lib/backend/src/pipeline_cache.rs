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

use std::{
    fs::{self, File},
    io,
    sync::Arc,
};

use ash::vk;
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};
use directories::ProjectDirs;
use four_cc::FourCC;
use log::{info, warn};
use uuid::Uuid;

use crate::{vulkan::Device, BackendError};

#[derive(Debug)]
pub struct PipelineCache {
    device: Arc<Device>,
    cache: vk::PipelineCache,
}

const MAGICK: FourCC = FourCC(*b"PLCH");
const VERSION: u32 = 1;
const CACHE_FILE_NAME: &str = "pipelines.bin";
const NEW_CACHE_FILE_NAME: &str = "pipelines.new.bin";

struct PipelineDiskCache {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: Uuid,

    data: Vec<u8>,
}

impl BinarySerialization for PipelineDiskCache {
    fn serialize(&self, w: &mut impl io::Write) -> std::io::Result<()> {
        MAGICK.serialize(w)?;
        w.write_u32::<LE>(VERSION)?;
        w.write_u64::<LE>(siphasher::sip::SipHasher::default().hash(&self.data))?;
        w.write_u32::<LE>(self.vendor_id)?;
        w.write_u32::<LE>(self.device_id)?;
        w.write_u32::<LE>(self.driver_version)?;
        self.uuid.serialize(w)?;
        self.data.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for PipelineDiskCache {
    fn deserialize(r: &mut impl io::prelude::Read) -> io::Result<Self> {
        let magick = FourCC::deserialize(r)?;
        if magick != MAGICK {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Wrong header"));
        }
        let version = r.read_u32::<LE>()?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Wrong version"));
        }
        let hash = r.read_u64::<LE>()?;
        let vendor_id = r.read_u32::<LE>()?;
        let device_id = r.read_u32::<LE>()?;
        let driver_version = r.read_u32::<LE>()?;
        let uuid = Uuid::deserialize(r)?;
        let data = Vec::deserialize(r)?;
        if siphasher::sip::SipHasher::default().hash(&data) != hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Data is corrupted",
            ));
        }

        Ok(Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data,
        })
    }
}

impl PipelineDiskCache {
    pub fn new(device: &Arc<Device>, data: &[u8]) -> Self {
        let vendor_id = device.physical_device().properties().vendor_id;
        let device_id = device.physical_device().properties().device_id;
        let driver_version = device.physical_device().properties().driver_version;
        let uuid = Uuid::from_bytes(device.physical_device().properties().pipeline_cache_uuid);

        Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data: data.to_vec(),
        }
    }

    pub fn load() -> io::Result<PipelineDiskCache> {
        if let Some(project_dirs) = ProjectDirs::from("com", "zlogaemz", "engine") {
            let cache_path = project_dirs.cache_dir().join(CACHE_FILE_NAME);
            info!("Loading pipeline cache from {:?}", cache_path);
            PipelineDiskCache::deserialize(&mut File::open(cache_path)?)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Can't get cache dir path",
            ))
        }
    }

    pub fn save(&self) -> io::Result<()> {
        if let Some(project_dirs) = ProjectDirs::from("com", "zlogaemz", "engine") {
            fs::create_dir_all(project_dirs.cache_dir())?;
            let cache_path = project_dirs.cache_dir().join(CACHE_FILE_NAME);
            let new_cache_path = project_dirs.cache_dir().join(NEW_CACHE_FILE_NAME);
            info!("Saving pipeline cache to {:?}", cache_path);
            self.serialize(&mut File::create(&new_cache_path)?)?;
            fs::rename(&new_cache_path, &cache_path)?;

            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Can't get cache dir path",
            ))
        }
    }
}

impl PipelineCache {
    pub fn new(device: &Arc<Device>) -> Result<Self, BackendError> {
        let data = if let Ok(cache) = PipelineDiskCache::load() {
            if cache.vendor_id == device.physical_device().properties().vendor_id
                && cache.device_id == device.physical_device().properties().device_id
                && cache.driver_version == device.physical_device().properties().driver_version
                && cache.uuid
                    == Uuid::from_bytes(device.physical_device().properties().pipeline_cache_uuid)
            {
                Some(cache.data)
            } else {
                None
            }
        } else {
            None
        };

        let create_info = if let Some(data) = &data {
            vk::PipelineCacheCreateInfo::builder().initial_data(data)
        } else {
            vk::PipelineCacheCreateInfo::builder()
        }
        .build();

        let cache = match unsafe { device.raw().create_pipeline_cache(&create_info, None) } {
            Ok(cache) => cache,
            Err(_) => {
                // Failed with initial data - so create empty cache.
                let create_info = vk::PipelineCacheCreateInfo::builder().build();
                unsafe { device.raw().create_pipeline_cache(&create_info, None) }?
            }
        };

        Ok(Self {
            device: device.clone(),
            cache,
        })
    }

    pub fn save(&self) -> io::Result<()> {
        let data =
            unsafe { self.device.raw().get_pipeline_cache_data(self.cache) }.map_err(|err| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to get pipeline cache data from device: {:?}", err),
                )
            })?;
        PipelineDiskCache::new(&self.device, &data).save()
    }

    pub fn raw(&self) -> vk::PipelineCache {
        self.cache
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if let Err(err) = self.save() {
            warn!("Failed to save pipeline cache: {}", err);
        }
        unsafe { self.device.raw().destroy_pipeline_cache(self.cache, None) };
    }
}
