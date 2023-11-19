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
};

use ash::vk;
use directories::ProjectDirs;
use log::info;
use speedy::{Context, Readable, Writable};
use uuid::Uuid;

use crate::BackendResult;

use super::PhysicalDevice;

#[derive(Debug)]
pub struct PipelineCache {
    pub raw: vk::PipelineCache,
}

const MAGICK: [u8; 4] = *b"PLCH";
const VERSION: u32 = 1;
const CACHE_FILE_NAME: &str = "pipelines.bin";
const NEW_CACHE_FILE_NAME: &str = "pipelines.new.bin";

#[derive(Debug, Readable, Writable, PartialEq, Eq)]
struct Header {
    pub magic: [u8; 4],
    pub version: u32,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            magic: MAGICK,
            version: VERSION,
        }
    }
}

#[derive(Debug)]
struct PipelineDiskCache {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: Uuid,
    data: Vec<u8>,
}

impl<'a, C: Context> Readable<'a, C> for PipelineDiskCache {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        reader
            .read_value::<Header>()
            .map(|x| x == Header::default())?;
        Ok(Self {
            vendor_id: reader.read_value()?,
            device_id: reader.read_value()?,
            driver_version: reader.read_value()?,
            uuid: reader.read_value()?,
            data: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for PipelineDiskCache {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_value(&Header::default())?;
        writer.write_value(&self.vendor_id)?;
        writer.write_value(&self.device_id)?;
        writer.write_value(&self.driver_version)?;
        writer.write_value(&self.uuid)?;
        writer.write_value(&self.data)?;

        Ok(())
    }
}

impl PipelineDiskCache {
    pub fn new(pdevice: &PhysicalDevice, data: &[u8]) -> Self {
        let vendor_id = pdevice.properties.vendor_id;
        let device_id = pdevice.properties.device_id;
        let driver_version = pdevice.properties.driver_version;
        let uuid = Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid);

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
            Ok(PipelineDiskCache::read_from_stream_buffered(File::open(
                cache_path,
            )?)?)
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
            self.write_to_stream(File::create(&new_cache_path)?)?;
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

pub fn load_or_create_pipeline_cache(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
) -> BackendResult<vk::PipelineCache> {
    let data = if let Ok(cache) = PipelineDiskCache::load() {
        if cache.vendor_id == pdevice.properties.vendor_id
            && cache.device_id == pdevice.properties.device_id
            && cache.driver_version == pdevice.properties.driver_version
            && cache.uuid == Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid)
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

    let cache = match unsafe { device.create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            let create_info = vk::PipelineCacheCreateInfo::builder().build();
            unsafe { device.create_pipeline_cache(&create_info, None) }?
        }
    };

    Ok(cache)
}

pub fn save_pipeline_cache(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    cache: vk::PipelineCache,
) -> io::Result<()> {
    let data = unsafe { device.get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    PipelineDiskCache::new(pdevice, &data).save()
}
