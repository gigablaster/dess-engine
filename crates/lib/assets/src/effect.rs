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

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use ash::vk;
use bevy_tasks::AsyncComputeTaskPool;
use dess_backend::vulkan::{BlendDesc, PipelineCreateDesc};
use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};
use uuid::{uuid, Uuid};

use crate::{get_absolute_asset_path, get_relative_asset_path, read_to_end, Asset, Error};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Readable, Writable, Serialize, Deserialize)]
pub enum ShaderStage {
    #[serde(rename = "vertex")]
    Vertex,
    #[serde(rename = "fragment")]
    Fragment,
    #[serde(rename = "compute")]
    Compute,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShaderConstant {
    #[serde(rename = "alpha_test")]
    AlphaTest,
}

/// Shader description
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffectShaderDesc {
    /// Path to shader
    pub shader: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Readable, Writable)]
pub enum EffectBlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

impl From<EffectBlendFactor> for vk::BlendFactor {
    fn from(value: EffectBlendFactor) -> Self {
        match value {
            EffectBlendFactor::Zero => vk::BlendFactor::ZERO,
            EffectBlendFactor::One => vk::BlendFactor::ONE,
            EffectBlendFactor::SrcColor => vk::BlendFactor::SRC1_COLOR,
            EffectBlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            EffectBlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            EffectBlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            EffectBlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            EffectBlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            EffectBlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            EffectBlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Readable, Writable)]
pub enum EffectBlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl From<EffectBlendOp> for vk::BlendOp {
    fn from(value: EffectBlendOp) -> Self {
        match value {
            EffectBlendOp::Add => vk::BlendOp::ADD,
            EffectBlendOp::Subtract => vk::BlendOp::SUBTRACT,
            EffectBlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            EffectBlendOp::Min => vk::BlendOp::MIN,
            EffectBlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Readable, Writable)]
pub enum EffectCompareOp {
    Never,
    Less,
    Equal,
    #[default]
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl From<EffectCompareOp> for vk::CompareOp {
    fn from(value: EffectCompareOp) -> Self {
        match value {
            EffectCompareOp::Never => vk::CompareOp::NEVER,
            EffectCompareOp::Less => vk::CompareOp::LESS,
            EffectCompareOp::Equal => vk::CompareOp::EQUAL,
            EffectCompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            EffectCompareOp::Greater => vk::CompareOp::GREATER,
            EffectCompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            EffectCompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            EffectCompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, Readable, Writable)]
pub enum EffectCullMode {
    None,
    #[default]
    Front,
    Back,
    Both,
}

impl From<EffectCullMode> for vk::CullModeFlags {
    fn from(value: EffectCullMode) -> Self {
        match value {
            EffectCullMode::None => vk::CullModeFlags::NONE,
            EffectCullMode::Front => vk::CullModeFlags::FRONT,
            EffectCullMode::Back => vk::CullModeFlags::BACK,
            EffectCullMode::Both => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, Readable, Writable)]
pub enum EffectFrontFace {
    #[default]
    ClockWise,
    CounterClockWise,
}

impl From<EffectFrontFace> for vk::FrontFace {
    fn from(value: EffectFrontFace) -> Self {
        match value {
            EffectFrontFace::ClockWise => vk::FrontFace::CLOCKWISE,
            EffectFrontFace::CounterClockWise => vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Writable, Readable)]
pub struct EffectBlend {
    pub src: EffectBlendFactor,
    pub dst: EffectBlendFactor,
    pub op: EffectBlendOp,
}

impl From<EffectBlend> for BlendDesc {
    fn from(value: EffectBlend) -> Self {
        BlendDesc {
            src_blend: value.src.into(),
            dst_blend: value.dst.into(),
            op: value.op.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Readable, Writable)]
pub struct EffectPipelineDesc {
    pub blend: Option<(EffectBlend, EffectBlend)>,
    pub depth: Option<EffectCompareOp>,
    pub depth_write: Option<bool>,
    pub cull: Option<(EffectCullMode, EffectFrontFace)>,
}

impl From<EffectPipelineDesc> for PipelineCreateDesc {
    fn from(value: EffectPipelineDesc) -> Self {
        PipelineCreateDesc {
            blend: value
                .blend
                .map(|(color, alpha)| (color.into(), alpha.into())),
            depth_test: value.depth.map(|x| x.into()),
            depth_write: value.depth_write.unwrap_or(true),
            cull: value.cull.map(|(mode, front)| (mode.into(), front.into())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectTechniqueSource {
    /// Vertex shader
    pub vertex: Option<EffectShaderDesc>,
    /// Fragment shader
    pub fragment: Option<EffectShaderDesc>,
    /// Pipeline description
    #[serde(rename = "pipeline")]
    pub pipeline_desc: EffectPipelineDesc,
}

/// Technique description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSource {
    techniques: HashMap<String, EffectTechniqueSource>,
}

#[derive(Debug, Writable, Hash, Readable)]
pub struct EffectShader {
    pub stage: ShaderStage,
    pub code: Vec<u8>,
    pub entry: String,
}

#[derive(Debug, Writable, Readable)]
pub struct EffectTechinque {
    pub shaders: Vec<EffectShader>,
    pub pipeline_desc: EffectPipelineDesc,
}

#[derive(Debug, Writable, Readable)]
pub struct EffectAsset {
    pub techniques: HashMap<String, EffectTechinque>,
}

impl Asset for EffectAsset {
    const TYPE_ID: Uuid = uuid!("8ef54231-2b31-4826-8b49-d3b2947764b3");

    fn serialize<W: std::io::prelude::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::prelude::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }
}

struct IncludeProvider {}

impl shader_prepper::IncludeProvider for IncludeProvider {
    type IncludeContext = PathBuf;

    fn get_include(
        &mut self,
        path: &str,
        context: &Self::IncludeContext,
    ) -> Result<(String, Self::IncludeContext), failure::Error> {
        let path = get_relative_asset_path(context.join(Path::new(path)))?;
        let parent = path.parent().unwrap().to_owned();
        let data = read_to_end(get_absolute_asset_path(path)?)?;

        Ok((String::from_utf8(data)?, parent))
    }
}

async fn compile_hlsl(path: String, stage: ShaderStage) -> Result<EffectShader, Error> {
    let mut source = String::new();
    let mut include_provider = IncludeProvider {};
    let chunks =
        shader_prepper::process_file(&path, &mut include_provider, Path::new("").to_path_buf())
            .map_err(|x| Error::ImportFailed(x.to_string()))?;
    chunks.into_iter().for_each(|x| source += &x.source);
    let profile = match stage {
        ShaderStage::Vertex => "vs_6_4",
        ShaderStage::Fragment => "ps_6_4",
        ShaderStage::Compute => "cs_6_4",
    };
    let code = hassle_rs::compile_hlsl(
        &path,
        &source,
        "main",
        profile,
        &[
            "-spirv",
            "-fspv-target-env=vulkan1.3",
            "-WX",
            "-Ges",
            "-HV 2021",
        ],
        &[],
    )
    .map_err(|x| Error::ProcessingFailed(x.to_string()))?;
    Ok(EffectShader {
        stage,
        code,
        entry: "main".to_owned(),
    })
}

pub fn import_effect(effect: EffectSource) -> Result<EffectAsset, Error> {
    let mut techniques = HashMap::new();
    for (name, tech) in effect.techniques.into_iter() {
        let compiled_shaders = AsyncComputeTaskPool::get().scope(|s| {
            if let Some(x) = tech.vertex {
                s.spawn(compile_hlsl(x.shader, ShaderStage::Vertex))
            }
            if let Some(x) = tech.fragment {
                s.spawn(compile_hlsl(x.shader, ShaderStage::Fragment))
            };
        });
        let mut shaders = Vec::new();
        for shader in compiled_shaders {
            shaders.push(shader?);
        }
        if shaders.is_empty() {
            return Err(Error::ImportFailed(
                format!("Techinique {} has no shaders", name).to_owned(),
            ));
        }
        techniques.insert(
            name,
            EffectTechinque {
                shaders,
                pipeline_desc: tech.pipeline_desc,
            },
        );
    }

    Ok(EffectAsset { techniques })
}
