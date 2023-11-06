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
    io::{self, Cursor, Seek, Write},
    path::Path,
};

use byteorder::{LittleEndian, WriteBytesExt};
use dess_assets::{
    BundleDesc, LocalBundleDesc, LOCAL_BUNDLE_ALIGN, LOCAL_BUNDLE_DICT_SIZE,
    LOCAL_BUNDLE_DICT_USAGE_LIMIT, LOCAL_BUNDLE_FILE_VERSION, LOCAL_BUNDLE_MAGIC,
};
use dess_common::traits::BinarySerialization;
use log::{error, info};
use uuid::Uuid;

use crate::{cached_asset_path, read_to_end, AssetProcessingContext};

const TRAIN_SAMPLE_SIZE: usize = 256;

/// Builds local asset bundle
///
/// Doesn't do processing work, just collect already processed files and put them
/// in bundle in sorted way.
pub fn build_bundle(context: AssetProcessingContext, target: &Path) -> io::Result<()> {
    let mut target = File::create(target)?;
    let mut desc = LocalBundleDesc::default();
    let mut dicts = HashMap::default();
    LOCAL_BUNDLE_MAGIC.serialize(&mut target)?;
    LOCAL_BUNDLE_FILE_VERSION.serialize(&mut target)?;
    let all_assets = context.all_assets();
    for info in all_assets {
        let src_path = cached_asset_path(info.asset);
        if src_path.exists() {
            let data = read_to_end(src_path)?;
            let size = data.len() as u32;
            let data = if size >= LOCAL_BUNDLE_ALIGN as u32 {
                info!("Compress {}", info.asset);
                let mut result = Vec::new();
                let writer = Cursor::new(&mut result);
                let mut encoder = create_encoder(writer, &data, info.ty, &mut dicts)?;
                encoder.set_pledged_src_size(Some(size as _))?;
                encoder.write_all(&data)?;
                encoder.do_finish()?;

                result
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
    let offset = try_align(&mut target)?;
    dicts.serialize(&mut target)?;
    desc.serialize(&mut target)?;
    target.write_u64::<LittleEndian>(offset)?;

    Ok(())
}

fn create_encoder<'a, W: Write>(
    w: W,
    data: &'a [u8],
    ty: Uuid,
    dicts: &'a mut HashMap<Uuid, Vec<u8>>,
) -> io::Result<zstd::Encoder<'a, W>> {
    if data.len() <= LOCAL_BUNDLE_DICT_USAGE_LIMIT {
        zstd::Encoder::new(w, 22)
    } else {
        let dict = if let Some(dict) = dicts.get(&ty) {
            dict
        } else {
            info!("Generate zstd dictionary for asset type {}", ty);
            let dict = prepare_dict(data)?;
            dicts.insert(ty, dict);

            dicts.get(&ty).unwrap()
        };
        zstd::Encoder::with_dictionary(w, 22, dict)
    }
}

fn prepare_dict(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut samples = Vec::new();
    let mut size = data.len();
    while size > TRAIN_SAMPLE_SIZE {
        samples.push(TRAIN_SAMPLE_SIZE);
        size -= TRAIN_SAMPLE_SIZE;
    }
    samples.push(size);

    zstd::dict::from_continuous(data, &samples, LOCAL_BUNDLE_DICT_SIZE)
}

fn try_align<W: Seek>(w: &mut W) -> io::Result<u64> {
    let offset = w.stream_position()?;
    let offset_align = (offset & !(LOCAL_BUNDLE_ALIGN - 1)) + LOCAL_BUNDLE_ALIGN;

    // Align if possible, current position if not.
    Ok(w.seek(io::SeekFrom::Start(offset_align))
        .unwrap_or(w.stream_position()?))
}
