// Copyright (C) 2023 Vladimir Kuskov

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

use std::collections::{HashMap, HashSet};

use speedy::{Context, Readable, Writable};

use crate::{AddressableAsset, Asset, AssetRef, Material};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct LightingAttributes {
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [i16; 2],
    _pad: [i16; 2],
}

impl<'a, C: Context> Readable<'a, C> for LightingAttributes {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(Self {
            normal: reader.read_value::<[i16; 2]>()?,
            tangent: reader.read_value::<[i16; 2]>()?,
            uv: reader.read_value::<[i16; 2]>()?,
            _pad: [0, 0],
        })
    }
}

impl<C: Context> Writable<C> for LightingAttributes {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_value(&self.normal)?;
        writer.write_value(&self.tangent)?;
        writer.write_value(&self.uv)?;

        Ok(())
    }
}

impl LightingAttributes {
    pub fn new(normal: [i16; 2], tangent: [i16; 2], uv: [i16; 2]) -> Self {
        Self {
            normal,
            tangent,
            uv,
            _pad: [0, 0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Readable, Writable)]
pub struct Bone {
    pub parent: u32,
    pub local_translation: [f32; 3],
    pub local_rotation: [f32; 4],
    pub local_scale: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Readable, Writable)]
pub struct Surface {
    pub first: u32,
    pub count: u32,
    pub bounds: ([f32; 3], [f32; 3]),
    pub max_position_value: f32,
    pub max_uv_value: f32,
    pub material: Material,
}

#[derive(Debug, Default, Readable, Writable)]
pub struct MeshData {
    pub geometry: u32,
    pub attributes: u32,
    pub indices: u32,
    pub surfaces: Vec<Surface>,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct StaticMeshGeometry {
    pub position: [i16; 3],
    _padding: u16,
}

impl<'a, C: Context> Readable<'a, C> for StaticMeshGeometry {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        Ok(Self {
            position: reader.read_value()?,
            _padding: 0,
        })
    }
}

impl<C: Context> Writable<C> for StaticMeshGeometry {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> Result<(), <C as Context>::Error> {
        writer.write_value(&self.position)
    }
}

impl StaticMeshGeometry {
    pub fn new(position: [i16; 3]) -> Self {
        Self {
            position,
            _padding: 0,
        }
    }
}

impl Surface {
    pub(crate) fn collect_dependencies(&self, deps: &mut HashSet<AssetRef>) {
        self.material.collect_dependencies(deps);
    }
}

#[derive(Debug, Default, Readable, Writable)]
pub struct ModelAsset {
    pub static_geometry: Vec<StaticMeshGeometry>,
    pub attributes: Vec<LightingAttributes>,
    pub indices: Vec<u16>,
    pub static_meshes: Vec<MeshData>,
    pub mesh_names: HashMap<String, u32>,
    pub bones: Vec<Bone>,
    pub names: HashMap<String, u32>,
    pub node_to_mesh: Vec<(u32, u32)>,
}

impl AddressableAsset for ModelAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("7b229650-8f34-4d5a-b140-8e5d9ce599aa");
}

impl Asset for ModelAsset {
    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn collect_depenencies(&self, dependencies: &mut std::collections::HashSet<AssetRef>) {
        self.static_meshes.iter().for_each(|x| {
            x.surfaces
                .iter()
                .for_each(|x| x.collect_dependencies(dependencies))
        });
    }
}
