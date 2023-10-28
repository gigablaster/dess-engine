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
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use clap::{Arg, ArgAction};
use dess_content::{ContentImporter, GltfModelImporter, ImportContext, TextureImporter};

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
                let target_path: PathBuf =
                    if let Some(parent) = path.strip_prefix(root).unwrap().parent() {
                        output.join(parent)
                    } else {
                        output.into()
                    }
                    .as_os_str()
                    .to_ascii_lowercase()
                    .into();
                if !target_path.exists() {
                    create_dir_all(&target_path).unwrap();
                }
                let file_name: PathBuf = path.file_name().unwrap().into();
                let target_file: PathBuf = target_path
                    .join(importer.target_name(&file_name))
                    .into_os_string()
                    .to_ascii_lowercase()
                    .into();
                let path = path.clone();
                if need_update(&path, &target_file) {
                    s.spawn(move |_| {
                        let context = ImportContext {
                            source_dir: path.parent().unwrap(),
                            destination_dir: target_path.strip_prefix(output).unwrap(),
                        };
                        let content = match importer.import(&path, &context) {
                            Ok(content) => content,
                            Err(err) => {
                                eprintln!("Import failed: {:?}", err);
                                return;
                            }
                        };
                        match content.save(&target_file) {
                            Ok(_) => println!("Transformed {:?} into {:?}", path, target_file),
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
    let importers: Vec<Box<dyn ContentImporter>> = vec![
        Box::<TextureImporter>::default(),
        Box::<GltfModelImporter>::default(),
    ];
    transform_directory(
        &importers,
        Path::new(&root),
        &PathBuf::from(""),
        Path::new(out),
    );
}
