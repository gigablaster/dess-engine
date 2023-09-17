use std::{
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use clap::{Arg, ArgAction};
use dess_content::{texture::TextureImporter, ContentImporter};

fn need_update(src: &Path, dst: &Path) -> bool {
    if !dst.exists() {
        return true;
    }
    src.metadata().unwrap().modified().unwrap() > dst.metadata().unwrap().created().unwrap()
}

fn collect_dirs(dir: &Path, dirs: &mut Vec<PathBuf>) {
    let paths = fs::read_dir(dir).unwrap();
    paths.for_each(|path| {
        let path = path.unwrap().path();
        if path.is_file() {
            dirs.push(path);
        } else if path.is_dir() {
            collect_dirs(&path, dirs);
        }
    })
}

fn transform_directory(
    importers: &[Box<dyn ContentImporter>],
    root: &Path,
    current: &Path,
    output: &Path,
) {
    let dir = root.join(current);
    let mut dirs = Vec::new();
    collect_dirs(&dir, &mut dirs);
    rayon::scope(|s| {
        dirs.iter().for_each(|path| {
            if let Some(importer) = importers.iter().find(|x| x.can_handle(path)) {
                let target_path = if let Some(parent) = path.strip_prefix(root).unwrap().parent() {
                    output.join(parent)
                } else {
                    output.into()
                };
                if !target_path.exists() {
                    create_dir_all(&target_path).unwrap();
                }
                let file_name: PathBuf = path.file_name().unwrap().into();
                let target_path = target_path.join(importer.target_name(&file_name));
                let path = path.clone();
                if need_update(&path, &target_path) {
                    s.spawn(move |_| {
                        let content = match importer.import(&path) {
                            Ok(content) => content,
                            Err(err) => {
                                eprintln!("Import failed: {:?}", err);
                                return;
                            }
                        };
                        match content.save(&target_path) {
                            Ok(_) => println!("Transformed {:?} into {:?}", path, target_path),
                            Err(err) => eprintln!("Save falied: {:?} - {:?}", path, err),
                        }
                    })
                }
            }
        })
    });
}

fn main() {
    let args = clap::Command::new("transformer")
        .version("0.1.0")
        .author("gigablaster")
        .about("Transform content for dess engine")
        .arg(
            Arg::new("destination")
                .long("dst")
                .help("Output path")
                .value_name("DIR")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("source")
                .long("src")
                .help("Source path")
                .value_name("DIR")
                .action(ArgAction::Set),
        )
        .get_matches();
    let root = args
        .get_one::<String>("source")
        .expect("Need source folder");
    let out = args
        .get_one::<String>("destination")
        .expect("Need output folder");
    let importers: Vec<Box<dyn ContentImporter>> = vec![Box::<TextureImporter>::default()];
    transform_directory(
        &importers,
        Path::new(&root),
        &PathBuf::from(""),
        Path::new(out),
    );
}
