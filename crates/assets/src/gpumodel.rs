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

use std::collections::HashMap;

use dess_common::traits::{BinaryDeserialization, BinarySerialization};

use crate::{
    gpumesh::{Bone, StaticGpuMesh},
    Asset,
};

#[derive(Debug, Default)]
pub struct GpuModel {
    pub static_meshes: Vec<StaticGpuMesh>,
    pub mesh_names: HashMap<String, u32>,
    pub bones: Vec<Bone>,
    pub names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

impl BinarySerialization for GpuModel {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.static_meshes.serialize(w)?;
        self.mesh_names.serialize(w)?;
        self.bones.serialize(w)?;
        self.names.serialize(w)?;
        self.node_to_mesh.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for GpuModel {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let static_meshes = Vec::deserialize(r)?;
        let mesh_names = HashMap::deserialize(r)?;
        let bones = Vec::deserialize(r)?;
        let names = HashMap::deserialize(r)?;
        let node_to_mesh = Vec::deserialize(r)?;

        Ok(Self {
            static_meshes,
            mesh_names,
            bones,
            names,
            node_to_mesh,
        })
    }
}

impl Asset for GpuModel {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("7b229650-8f34-4d5a-b140-8e5d9ce599aa");

    fn collect_dependencies(&self, deps: &mut Vec<crate::AssetRef>) {
        self.static_meshes
            .iter()
            .for_each(|x| x.collect_dependencies(deps))
    }
}
