use std::{
    fs::File,
    io::{self, Seek, Write},
    path::Path,
};

use byteorder::{LittleEndian, WriteBytesExt};
use dess_assets::{
    Asset, BundleDesc, GpuModel, LocalBundleDesc, LOCAL_BUNDLE_ALIGN, LOCAL_BUNDLE_FILE_VERSION,
    LOCAL_BUNDLE_MAGIC,
};
use dess_common::traits::BinarySerialization;
use log::{error, info};

use crate::{cached_asset_name, read_to_end, AssetProcessingContext};

/// Builds local asset bundle
///
/// Doesn't do processing work, just collect already processed files and put them
/// in bundle in sorted way.
pub fn build_bundle(context: AssetProcessingContext, target: &Path) -> io::Result<()> {
    let mut target = File::create(target)?;
    let mut desc = LocalBundleDesc::default();
    LOCAL_BUNDLE_MAGIC.serialize(&mut target)?;
    LOCAL_BUNDLE_FILE_VERSION.serialize(&mut target)?;
    let all_assets = context.all_assets();
    for (asset, ty) in all_assets {
        let src_path = cached_asset_name(asset);
        if src_path.exists() {
            let data = read_to_end(src_path)?;
            let size = data.len() as u32;
            let data = if size > LOCAL_BUNDLE_ALIGN as u32 && !skip_compression(ty) {
                info!("Compress {}", asset);
                lz4_flex::compress(&data)
            } else {
                info!("Write {}", asset);
                data
            };
            let offset = try_align(&mut target)?;
            target.write_all(&data)?;
            desc.add_asset(asset, ty, offset, size, data.len() as _);
        } else {
            error!("Asset {} doesn't exist - skip", asset);
        }
    }
    context
        .all_names()
        .iter()
        .for_each(|(name, asset)| desc.set_name(*asset, name));
    let offset = try_align(&mut target)?;
    desc.serialize(&mut target)?;
    target.write_u64::<LittleEndian>(offset)?;

    Ok(())
}

fn skip_compression(ty: uuid::Uuid) -> bool {
    ty == GpuModel::TYPE_ID
}

fn try_align<W: Write + Seek>(w: &mut W) -> io::Result<u64> {
    let offset = w.stream_position()?;
    let offset_align = (offset & !(LOCAL_BUNDLE_ALIGN - 1)) + LOCAL_BUNDLE_ALIGN;

    // Align if possible, current position if not.
    Ok(w.seek(io::SeekFrom::Start(offset_align))
        .unwrap_or(w.stream_position()?))
}
