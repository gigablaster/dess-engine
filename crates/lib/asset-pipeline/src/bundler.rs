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
    fs::File,
    io::{self, Seek, Write},
    path::Path,
};

use dess_assets::{LocalBundleDesc, ROOT_ASSET_PATH};
use log::{error, info};
use speedy::Writable;

use crate::{get_cached_asset_path, read_to_end, OfflineAssetProcessingContext};

const LOCAL_BUNDLE_ALIGN: u64 = 4096;

/// Builds local asset bundle
///
/// Doesn't do processing work, just collect already processed files and put them
/// in bundle in sorted way.
pub fn build_bundle(context: OfflineAssetProcessingContext, name: &str) -> io::Result<()> {
    let mut target = File::create(Path::new(ROOT_ASSET_PATH).join(format!("{name}.bin")))?;
    let mut desc = LocalBundleDesc::default();
    let all_assets = context.all_assets();
    for info in all_assets {
        let src_path = get_cached_asset_path(info.asset);
        if src_path.exists() {
            let data = read_to_end(src_path)?;
            let size = data.len() as u32;
            let data = if size >= LOCAL_BUNDLE_ALIGN as u32 {
                info!("Compress {}", info.asset);
                let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
                encoder.write_all(&data)?;
                encoder.finish()?
            } else {
                info!("Write small asset {}", info.asset);
                data
            };
            let offset = try_align(&mut target)?;
            target.write_all(&data)?;
            desc.add_asset(info.asset, info.ty, offset, size, data.len() as _);
            if let Some(dependencies) = context.get_dependencies(info.asset) {
                desc.set_dependencies(info.asset, &dependencies);
            }
        } else {
            error!("Asset {} doesn't exist - skip", info.asset);
        }
    }
    context
        .all_names()
        .iter()
        .for_each(|(name, asset)| desc.set_name(*asset, name));
    desc.write_to_stream(File::create(
        Path::new(ROOT_ASSET_PATH).join(format!("{name}.idx")),
    )?)?;

    Ok(())
}

fn try_align<W: Seek>(w: &mut W) -> io::Result<u64> {
    let offset = w.stream_position()?;
    if offset == 0 {
        return Ok(0);
    }
    let offset_align = (offset & !(LOCAL_BUNDLE_ALIGN - 1)) + LOCAL_BUNDLE_ALIGN;

    // Align if possible, current position if not.
    Ok(w.seek(io::SeekFrom::Start(offset_align))
        .unwrap_or(w.stream_position()?))
}
