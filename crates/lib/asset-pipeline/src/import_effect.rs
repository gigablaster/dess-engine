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
    fs::{self, File},
    path::{Path, PathBuf},
};

use byte_slice_cast::AsSliceOf;
use dess_assets::{EffectAsset, Pipeline, Shader, ShaderPass, ShaderStage, SpecializationConstant};
use log::{debug, error, info, warn};
use normalize_path::NormalizePath;
use serde::{Deserialize, Serialize};
use spirv_tools::{
    error::{MessageCallback, MessageLevel},
    opt::Optimizer,
    val::Validator,
};

use crate::{get_absolute_asset_path, Content, ContentImporter, ContentProcessor, Error};

#[derive(Debug, Clone)]
pub struct EffectSource {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderDesc {
    pub stage: ShaderStage,
    pub shader: PathBuf,
    pub defines: Option<HashSet<String>>,
    pub specializations: Option<HashMap<SpecializationConstant, u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectPassDesc {
    pub shaders: Vec<ShaderDesc>,
    pub pipeline: Option<Pipeline>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDesc {
    pub passes: HashMap<String, EffectPassDesc>,
}

#[derive(Debug)]
pub struct LoadedShader {
    pub stage: ShaderStage,
    pub code: String,
    pub sources: HashSet<String>,
    pub defines: Vec<String>,
    pub specializations: Vec<(SpecializationConstant, u32)>,
}

#[derive(Debug, Default)]
pub struct EffectPassContent {
    pub shaders: Vec<LoadedShader>,
    pub pipeline: Pipeline,
}

#[derive(Debug, Default)]
pub struct EffectContent(HashMap<String, EffectPassContent>);

impl Content for EffectContent {}

impl ContentImporter<EffectContent> for EffectSource {
    fn import(&self) -> Result<EffectContent, crate::Error> {
        let desc: EffectDesc = serde_json::from_reader(File::open(&self.path)?)?;
        let mut effect = EffectContent::default();
        for (name, pass_desc) in desc.passes.iter() {
            let mut pass = EffectPassContent::default();
            let pipeline = pass_desc.clone().pipeline.unwrap_or_default();
            for shader in pass_desc.shaders.iter() {
                let chunks = shader_prepper::process_file(
                    get_absolute_asset_path(&shader.shader)?
                        .to_str()
                        .ok_or(Error::Fail)?,
                    &mut FileIncludeProvider,
                    shader.shader.clone(),
                )
                .map_err(|err| Error::ImportFailed(err.to_string()))?;

                let mut code = String::new();
                chunks.iter().for_each(|chunk| code += &chunk.source);
                let shader = shader.clone();
                pass.shaders.push(LoadedShader {
                    stage: shader.stage,
                    code,
                    sources: chunks.iter().map(|x| x.file.to_owned()).collect(),
                    defines: shader.defines.unwrap_or_default().iter().cloned().collect(),
                    specializations: shader
                        .specializations
                        .unwrap_or_default()
                        .iter()
                        .map(|x| (*x.0, *x.1))
                        .collect(),
                });
            }
            pass.pipeline = pipeline;
            effect.0.insert(name.to_owned(), pass);
        }

        Ok(effect)
    }
}

struct FileIncludeProvider;

impl shader_prepper::IncludeProvider for FileIncludeProvider {
    type IncludeContext = PathBuf;

    fn get_include(
        &mut self,
        path: &str,
        context: &Self::IncludeContext,
    ) -> Result<(String, Self::IncludeContext), failure::Error> {
        let parent: PathBuf = get_absolute_asset_path(context)
            .map_err(|err| failure::err_msg(format!("{:?}", err)))?
            .parent()
            .unwrap()
            .into();
        let path = parent.join(Path::new(path)).normalize();
        let result =
            fs::read_to_string(path).map_err(|err| failure::err_msg(format!("{:?}", err)))?;

        Ok((result, context.clone()))
    }
}

#[derive(Debug, Default)]
pub struct CompileEffect;

impl CompileEffect {
    fn compile_shader(stage: ShaderStage, code: &str, flags: &[String]) -> Result<Vec<u8>, Error> {
        let profile = match stage {
            ShaderStage::Vertex => "vs_6_4",
            ShaderStage::Fragment => "ps_6_4",
        };
        let defines = flags
            .iter()
            .map(|x| (x.as_ref(), Some("1")))
            .collect::<Vec<_>>();
        let spirv = hassle_rs::compile_hlsl(
            "",
            code,
            "main",
            profile,
            &[
                "-spirv",
                "-fspv-target-env=vulkan1.1",
                "-WX",
                "-Ges",
                "-HV 2021",
            ],
            &defines,
        )
        .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        let mut optimizer = spirv_tools::opt::create(Some(spirv_tools::TargetEnv::Vulkan_1_1));
        optimizer.register_performance_passes();
        let data = spirv.as_slice_of::<u32>().unwrap();
        let spirv = optimizer
            .optimize(data, &mut OptCallbacks {}, None)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        let validator = spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1));
        validator
            .validate(spirv.as_words(), None)
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        Ok(spirv.as_bytes().to_vec())
    }
}

struct OptCallbacks;

impl MessageCallback for OptCallbacks {
    fn on_message(&mut self, msg: spirv_tools::error::Message) {
        match msg.level {
            MessageLevel::Info => info!("{} - {}", msg.line, msg.message),
            MessageLevel::Debug => debug!("{} - {}", msg.line, msg.message),
            MessageLevel::Error | MessageLevel::Fatal | MessageLevel::InternalError => {
                error!("{} - {}", msg.line, msg.message)
            }
            MessageLevel::Warning => warn!("{} - {}", msg.line, msg.message),
        }
    }
}

impl ContentProcessor<EffectContent, EffectAsset> for CompileEffect {
    fn process(
        &self,
        _asset: dess_assets::AssetRef,
        _context: &crate::AssetProcessingContext,
        content: EffectContent,
    ) -> Result<EffectAsset, Error> {
        let mut passes = HashMap::new();
        for (name, pass) in content.0.iter() {
            let mut shaders = Vec::new();
            for shader in &pass.shaders {
                let code = Self::compile_shader(shader.stage, &shader.code, &shader.defines)?;
                shaders.push(Shader {
                    stage: shader.stage,
                    code,
                    specializations: shader.specializations.clone(),
                });
            }
            let pass = ShaderPass {
                shaders,
                pipeline: pass.pipeline.clone(),
            };
            passes.insert(name.to_owned(), pass);
        }

        Ok(EffectAsset { passes })
    }
}
