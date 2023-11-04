use std::{env, fs, io, path::Path};

use dess_asset_pipeline::{AssetPipeline, ROOT_DATA_PATH};
use log::info;

fn process_assets(pipeline: &AssetPipeline, root: &Path) -> Result<(), io::Error> {
    let paths = fs::read_dir(root)?;
    for path in paths {
        let path = path?.path();
        if path.is_symlink() {
            continue;
        }
        if path.is_file() && have_extension(&path, "gltf") {
            let asset = pipeline.import_model(&path);
            info!("Found asset {} in {:?}", asset, path);
        }
        if path.is_dir() {
            process_assets(pipeline, &path)?;
        }
    }

    Ok(())
}

fn have_extension(path: &Path, e: &str) -> bool {
    if let Some(ext) = path.extension() {
        ext.to_ascii_lowercase().to_str().unwrap() == e
    } else {
        false
    }
}

fn main() {
    simple_logger::init().unwrap();
    let pipeline = AssetPipeline::default();
    let root = env::current_dir()
        .unwrap()
        .canonicalize()
        .unwrap()
        .join(ROOT_DATA_PATH);

    process_assets(&pipeline, &root).unwrap();
    pipeline.process_pending_assets();
}
