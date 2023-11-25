use std::{
    fs::{read_dir, read_to_string},
    io,
    path::Path,
};

use clap::{Arg, ArgAction};
use dess_asset_pipeline::{
    desc::{BundleDesc, BundledAsset},
    AssetPipeline, BUNDLE_DESC_PATH,
};
use log::info;
use serde_json::from_str;

fn process_assets(pipeline: &AssetPipeline, desc: &BundleDesc) -> Result<(), io::Error> {
    for (name, asset) in desc.assets() {
        let asset = match asset {
            BundledAsset::Shader(desc) => pipeline.import_effect(desc),
            BundledAsset::Model(path) => pipeline.import_model(Path::new(&path)),
            BundledAsset::Image(image) => {
                pipeline.import_image(Path::new(&image.source), image.purpose)
            }
        };
        pipeline.set_name(asset, &name);
    }
    Ok(())
}

fn ends_with(path: &Path, e: &str) -> bool {
    if let Some(ext) = path.to_str() {
        ext.to_ascii_lowercase().ends_with(e)
    } else {
        false
    }
}

fn build_bundle(path: &Path, pack: bool) -> io::Result<()> {
    let desc: BundleDesc = from_str(&read_to_string(path)?)?;

    info!("======= Build bundle \"{}\"", desc.name());

    let pipeline = AssetPipeline::new(desc.name());
    process_assets(&pipeline, &desc)?;
    pipeline.process_pending_assets()?;
    pipeline.save_db()?;

    if pack {
        info!("======= Pack bundle \"{}\"", desc.name());
        pipeline.bundle(desc.name())?
    }

    Ok(())
}

fn process_bundles(pack: bool) -> io::Result<()> {
    let dir = read_dir(Path::new(BUNDLE_DESC_PATH))?;
    for it in dir.into_iter() {
        let it = it?;
        if ends_with(&it.path(), ".json") {
            build_bundle(&it.path(), pack)?
        }
    }

    Ok(())
}

fn main() {
    let args = clap::Command::new("builder")
        .version("0.1.0")
        .author("gigablaster")
        .about("Asset builder for kiri engine")
        .arg(
            Arg::new("bundle")
                .long("bundle")
                .short('b')
                .help("Pack assets into bundles")
                .action(ArgAction::SetTrue),
        )
        .get_matches();
    simple_logger::init().unwrap();
    process_bundles(args.contains_id("bundle")).unwrap();
}
