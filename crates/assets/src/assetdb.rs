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
