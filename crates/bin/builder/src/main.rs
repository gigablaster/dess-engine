use std::{fs, io, path::Path};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use dess_asset_pipeline::{ContentProcessor, ImportContext};
use dess_assets::{GltfSource, ShaderSource, ROOT_DATA_PATH};

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
    simple_logger::init().unwrap();
    AsyncComputeTaskPool::get_or_init(TaskPool::new);
    let processor = ContentProcessor::default();
    collect(&processor, Path::new(ROOT_DATA_PATH)).unwrap();
    processor.process();
}
