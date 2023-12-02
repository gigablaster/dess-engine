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
    collections::HashMap,
    fs::File,
    io::{self, Cursor},
    path::Path,
};

use dess_asset_pipeline::get_cached_asset_path;
use dess_assets::{Asset, AssetRef, MappedFile};
use dess_backend::vulkan::{Device, ImageHandle, ProgramHandle};
use parking_lot::RwLock;

pub struct AssetManager<'a> {
    device: &'a Device,
    images: RwLock<HashMap<AssetRef, ImageHandle>>,
    programs: RwLock<HashMap<(AssetRef, AssetRef), ProgramHandle>>,
}

impl<'a> AssetManager<'a> {
    fn load_cached<T: Asset>(&self, asset: AssetRef) -> io::Result<Option<T>> {
        let cached = get_cached_asset_path(asset);
        if cached.exists() {
            let data = MappedFile::open(&cached)?;
            Ok(Some(T::deserialize(&mut Cursor::new(data.data()))?))
        } else {
            Ok(None)
        }
    }

    fn save_cache<T: Asset>(&self, asset: AssetRef, data: &T) -> io::Result<()> {
        let cached = get_cached_asset_path(asset);
        data.serialize(&mut File::create(cached)?)
    }
}
