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

mod renderworld;

use std::mem;

use dess_assets::{MeshVertexAttributes, StaticMeshVertex};
use dess_backend::{Format, InputVertexAttributeDesc, InputVertexStreamDesc};

#[repr(C, packed)]
pub struct BasicMeshVertex {
    pub position: [f32; 3],
}

#[repr(C, packed)]
pub struct BasicMeshAttribute {
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
}

#[repr(C, packed)]
pub struct BasicVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl BasicVertex {
    pub fn new(position: glam::Vec3, uv: glam::Vec2) -> Self {
        Self {
            position: position.to_array(),
            uv: uv.to_array(),
        }
    }
}

pub const BASIC_MESH_LAYOUT: [InputVertexStreamDesc; 2] = [
    InputVertexStreamDesc {
        attributes: &[InputVertexAttributeDesc {
            format: Format::RGB32_SFLOAT,
            locaion: 0,
            binding: 0,
            offset: 0,
        }],
        stride: mem::size_of::<BasicMeshVertex>(),
    },
    InputVertexStreamDesc {
        attributes: &[
            InputVertexAttributeDesc {
                format: Format::RGB32_SFLOAT,
                locaion: 1,
                binding: 1,
                offset: 0,
            },
            InputVertexAttributeDesc {
                format: Format::RGB32_SFLOAT,
                locaion: 2,
                binding: 1,
                offset: 12,
            },
            InputVertexAttributeDesc {
                format: Format::RG32_SFLOAT,
                locaion: 3,
                binding: 1,
                offset: 24,
            },
            InputVertexAttributeDesc {
                format: Format::RG32_SFLOAT,
                locaion: 4,
                binding: 1,
                offset: 32,
            },
        ],
        stride: mem::size_of::<BasicMeshAttribute>(),
    },
];

pub const BASIC_VERTEX_LAYOUT: [InputVertexStreamDesc; 1] = [InputVertexStreamDesc {
    attributes: &[
        InputVertexAttributeDesc {
            format: Format::RGB32_SFLOAT,
            locaion: 0,
            binding: 0,
            offset: 0,
        },
        InputVertexAttributeDesc {
            format: Format::RG32_SFLOAT,
            locaion: 1,
            binding: 0,
            offset: 12,
        },
    ],
    stride: mem::size_of::<BasicVertex>(),
}];

impl From<StaticMeshVertex> for BasicMeshVertex {
    fn from(value: StaticMeshVertex) -> Self {
        Self {
            position: value.position,
        }
    }
}

impl From<MeshVertexAttributes> for BasicMeshAttribute {
    fn from(value: MeshVertexAttributes) -> Self {
        Self {
            normal: value.normal,
            tangent: value.tangent,
            uv1: value.uv1,
            uv2: value.uv2,
        }
    }
}
