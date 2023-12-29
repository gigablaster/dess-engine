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

use std::mem;

use dess_assets::StaticMeshVertex;
use dess_backend::{Format, InputVertexAttributeDesc, InputVertexStreamDesc, PipelineVertex};

pub struct PackedMeshVertex {
    pub position: [u16; 3],
    _pad: u16,
    pub normal: [u16; 2],
    pub tangent: [u16; 2],
    pub uv1: [u16; 2],
    pub uv2: [u16; 2],
}

pub struct BasicVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl PipelineVertex for PackedMeshVertex {
    fn vertex_streams() -> &'static [InputVertexStreamDesc] {
        &[InputVertexStreamDesc {
            attributes: &[
                InputVertexAttributeDesc {
                    format: Format::RGB16_UNORM,
                    locaion: 0,
                    binding: 0,
                    offset: 0,
                },
                InputVertexAttributeDesc {
                    format: Format::RG16_UNORM,
                    locaion: 1,
                    binding: 0,
                    offset: 8,
                },
                InputVertexAttributeDesc {
                    format: Format::RG16_UNORM,
                    locaion: 2,
                    binding: 0,
                    offset: 12,
                },
                InputVertexAttributeDesc {
                    format: Format::RG16_UNORM,
                    locaion: 3,
                    binding: 0,
                    offset: 16,
                },
                InputVertexAttributeDesc {
                    format: Format::RG16_UNORM,
                    locaion: 4,
                    binding: 0,
                    offset: 20,
                },
            ],
            stride: mem::size_of::<PackedMeshVertex>(),
        }]
    }
}

impl PipelineVertex for BasicVertex {
    fn vertex_streams() -> &'static [InputVertexStreamDesc] {
        &[InputVertexStreamDesc {
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
        }]
    }
}

impl From<StaticMeshVertex> for PackedMeshVertex {
    fn from(value: StaticMeshVertex) -> Self {
        Self {
            position: value.position,
            _pad: 0,
            normal: value.normal,
            tangent: value.tangent,
            uv1: value.uv1,
            uv2: value.uv2,
        }
    }
}
