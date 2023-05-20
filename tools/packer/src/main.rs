use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use clap::{Arg, ArgAction};
use dess_vfs::DirectoryBaker;
use glob::Pattern;

trait Excluded {
    fn is_excluded(&self, path: &Path) -> bool;
}

impl Excluded for Vec<Pattern> {
    fn is_excluded(&self, path: &Path) -> bool {
        self.iter()
            .any(|pattern| pattern.matches(path.to_str().unwrap()))
    }
}

fn bake_directory(
    writer: &mut DirectoryBaker<File>,
    root: &Path,
    current: &Path,
    exclude: &Vec<Pattern>,
) {
    let paths = fs::read_dir(current).unwrap();
    for path in paths {
        let path = path.unwrap().path();
        if exclude.is_excluded(&path) || path.is_symlink() {
            continue;
        }
        if path.is_file() {
            let mut data = Vec::new();
            let mut file = File::open(&path).unwrap();
            file.read_to_end(&mut data).unwrap();
            writer
                .write(path.strip_prefix(root).unwrap().to_str().unwrap(), &data)
                .unwrap();
        }
        if path.is_dir() {
            bake_directory(writer, root, &path, exclude)
        }
    }
}

fn main() {
    let args = clap::Command::new("packer")
        .version("0.1.0")
        .author("gigablaster")
        .about("Archive packer for dess engine")
        .arg(
            Arg::new("out")
                .long("out")
                .short('o')
                .help("Output archive")
                .value_name("FILE")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("root")
                .long("root")
                .short('r')
                .help("Root path")
                .value_name("DIR")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("exclude")
                .long("exclude")
                .short('x')
                .help("File patterns to exclude")
                .num_args(0..)
                .action(ArgAction::Set),
        )
        .get_matches();
    let root = if let Some(root) = args.get_one::<String>("root") {
        PathBuf::from(root)
    } else {
        eprintln!("Need root folder");
        return;
    };
    let out = if let Some(out) = args.get_one::<String>("out") {
        PathBuf::from(out)
    } else {
        eprintln!("Need output file");
        return;
    };
    let exclude = if let Some(exclude) = args.get_many("exclude") {
        exclude
            .into_iter()
            .cloned()
            .map(|pattern: &String| Pattern::new(pattern).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    if !root.exists() || !root.is_dir() {
        eprintln!("{:?} must exist and be a directory", root);
        return;
    }
    let file = File::create(out).unwrap();
    let mut builder = DirectoryBaker::new(file).unwrap();
    bake_directory(&mut builder, &root, &root, &exclude);
    builder.finish().unwrap();
}
