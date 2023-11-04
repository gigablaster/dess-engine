use std::{env, fs, io, path::Path};

use clap::{Arg, ArgAction};
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
    let args = clap::Command::new("builder")
        .version("0.1.0")
        .author("gigablaster")
        .about("Asset bundler for dess engine")
        .arg(
            Arg::new("bundle")
                .long("bundle")
                .short('b')
                .help("Asset bundle to build")
                .value_name("FILE")
                .action(ArgAction::Set),
        )
        .get_matches();
    simple_logger::init().unwrap();
    let pipeline = AssetPipeline::default();
    let root = env::current_dir()
        .unwrap()
        .canonicalize()
        .unwrap()
        .join(ROOT_DATA_PATH);

    process_assets(&pipeline, &root).unwrap();
    pipeline.process_pending_assets();
    pipeline.save_db().unwrap();
    if let Some(bundle) = args.get_one::<String>("bundle") {
        info!("Buidling asset bundle {}", bundle);
        pipeline.bundle(Path::new(bundle)).unwrap();
    }
}
