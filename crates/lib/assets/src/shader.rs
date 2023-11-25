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

use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};

use crate::Asset;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum ShaderStage {
    #[serde(rename = "vertex")]
    Vertex,
    #[serde(rename = "fragment")]
    Fragment,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum BlendFactor {
    #[serde(rename = "zero")]
    Zero,
    #[serde(rename = "one")]
    One,
    #[serde(rename = "src_color")]
    SrcColor,
    #[serde(rename = "one_minus_src_color")]
    OneMinusSrcColor,
    #[serde(rename = "dst_color")]
    DstColor,
    #[serde(rename = "one_minus_dst_color")]
    OneMinusDstColor,
    #[serde(rename = "src_alpha")]
    SrcAlpha,
    #[serde(rename = "one_minus_src_alpha")]
    OneMinusSrcAlpha,
    #[serde(rename = "dst_alpha")]
    DstAlpha,
    #[serde(rename = "one_minus_dst_alpha")]
    OneMinusDstAlpha,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum BlendOp {
    #[serde(rename = "add")]
    Add,
    #[serde(rename = "subtract")]
    Subtract,
    #[serde(rename = "reverse_subtract")]
    ReverseSubtract,
    #[serde(rename = "min")]
    Min,
    #[serde(rename = "max")]
    Max,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum CompareOp {
    #[serde(rename = "never")]
    Never,
    #[serde(rename = "less")]
    Less,
    #[serde(rename = "equal")]
    Equal,
    #[serde(rename = "less_or_equal")]
    LessOrEqual,
    #[serde(rename = "greater")]
    Greater,
    #[serde(rename = "not_equal")]
    NotEqual,
    #[serde(rename = "greater_or_equal")]
    GreatedOrEqual,
    #[serde(rename = "always")]
    Always,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum CullMode {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "front")]
    Front,
    #[serde(rename = "back")]
    Back,
    #[serde(rename = "both")]
    FrontAndBack,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum FrontFace {
    #[serde(rename = "cw")]
    Clockwise,
    #[serde(rename = "ccw")]
    CounterClockwise,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub struct BlendDesc {
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub op: BlendOp,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum SpecializationConstant {
    #[serde(rename = "local_light_count")]
    LocalLightCount,
}

#[derive(Debug, Clone, Readable, Writable)]
pub struct ShaderAsset {
    pub stage: ShaderStage,
    pub specializations: Vec<(SpecializationConstant, u32)>,
    pub code: Vec<u8>,
}

impl Asset for ShaderAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("0d35d32c-8b62-41b2-8bdc-d329f06a5564");
    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}
