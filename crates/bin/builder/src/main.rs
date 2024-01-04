use std::{
    fs, io,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    thread,
    time::Duration,
};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use clap::{Arg, ArgAction};
use dess_asset_pipeline::{ContentProcessor, ImportContext};
use dess_assets::{GltfSource, ShaderSource, ROOT_DATA_PATH};
use log::info;
use notify::{RecursiveMode, Watcher};

fn collect(processor: &ContentProcessor, root: &Path) -> io::Result<()> {
    for path in fs::read_dir(root)? {
        let path = path?;
        if path.path().is_dir() {
            collect(processor, &path.path())?
        } else {
            let path = path.path().strip_prefix(ROOT_DATA_PATH).unwrap().to_owned();
            let path_str = path.to_str().unwrap().replace('\\', "/");
            if path_str.ends_with(".gltf") {
                processor.import(Box::new(GltfSource::new(path_str)));
            } else if path_str.ends_with("_ps.hlsl") {
                processor.import(Box::new(ShaderSource::fragment(path_str)));
            } else if path_str.ends_with("_vs.hlsl") {
                processor.import(Box::new(ShaderSource::vertex(path_str)));
            } else if path_str.ends_with("_cs.hlsl") {
                processor.import(Box::new(ShaderSource::compute(path_str)));
            }
        }
    }

    Ok(())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = clap::Command::new("builder")
        .version("0.1.0")
        .author("gigablaster <gigakek@protonmail.com>")
        .about("Asset builder for dess engine")
        .arg(
            Arg::new("watch")
                .long("watch")
                .required(false)
                .action(ArgAction::SetTrue),
        )
        .get_matches();
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    let processor = ContentProcessor::default();
    collect(&processor, Path::new(ROOT_DATA_PATH)).unwrap();
    processor.process();
    let need_reimport = Arc::new(AtomicBool::new(false));

    if args.get_flag("watch") {
        info!("Watching for changes...");
        let need_reimport2 = need_reimport.clone();
        let mut watcher = notify::recommended_watcher(move |_| {
            need_reimport2.store(true, std::sync::atomic::Ordering::Release);
        })
        .unwrap();
        loop {
            watcher
                .watch(Path::new(ROOT_DATA_PATH), RecursiveMode::Recursive)
                .unwrap();
            thread::sleep(Duration::from_secs(1));
            if need_reimport.load(std::sync::atomic::Ordering::Acquire) {
                let processor = ContentProcessor::default();
                collect(&processor, Path::new(ROOT_DATA_PATH)).unwrap();
                processor.process();
                need_reimport.store(false, std::sync::atomic::Ordering::Release);
            }
        }
    }
}
