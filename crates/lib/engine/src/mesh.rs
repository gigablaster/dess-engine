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

use std::{collections::HashMap, sync::Arc};

use dess_backend::vulkan::{BufferSlice, DescriptorHandle};
use smol_str::SmolStr;

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
struct ObjectScaleUniform {
    pub position_scale: f32,
    pub uv_scale: f32,
    _pad: [f32; 2],
}

/// Single primitive to draw
#[derive(Debug, Clone, Copy)]
pub struct SubMesh {
    pub first_index: u32,
    pub index_count: u32,
    pub bounds: (glam::Vec3, glam::Vec3),
    pub object_ds: DescriptorHandle,
    // pub material: MaterialHandle,
}

/// Reperesentation of single mesh
#[derive(Debug)]
pub struct RenderMesh {
    pub vertices: BufferSlice,
    pub indices: BufferSlice,
    pub submeshes: Vec<SubMesh>,
}

/// Representation of single gltf scene
#[derive(Debug)]
pub struct RenderModel {
    // bone transformations
    bones: Vec<glam::Affine3A>,
    // Bone parents, u32::MAX - no parent
    bone_parents: Vec<u32>,
    // Bone names
    bone_names: HashMap<String, u32>,
    // Actual meshes
    meshes: Vec<RenderMesh>,
    // Mesh names, name->index
    mesh_names: HashMap<String, u32>,
    // Which mesh located where, bone->mesh
    instances: Vec<(u32, u32)>,
}

/// Representation of single gltf file
///
/// Contains multiple models (scenes).
#[derive(Debug)]
pub struct RenderModelContainer {
    models: HashMap<String, RenderModel>,
}
