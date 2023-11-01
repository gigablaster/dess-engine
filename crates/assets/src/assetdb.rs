use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use dess_common::traits::XmlDeserialization;
use uuid::Uuid;
use xml::reader::XmlEvent;

use crate::Asset;

struct AssetData {
    /// Asset type ID
    ty: Uuid,
    /// Source file path. Might be anything, using to check if source asset
    /// should be rebuild.
    source: PathBuf,
}

/// Asset database
///
/// Used to keep track on all assets and their dependencies during development.
/// Then used to pack assets into final asset pack to ship with game.
struct AssetDatabase {
    // Many assets might point on same source file
    assets: HashMap<Uuid, AssetData>,
    // Map of named assets
    names: HashMap<String, Uuid>,
}

impl AssetData {
    pub fn new<T: Asset>(source: PathBuf) -> Self {
        Self {
            ty: T::TYPE_ID,
            source,
        }
    }
}
