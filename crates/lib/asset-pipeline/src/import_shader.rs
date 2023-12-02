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
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};

use crate::{
    get_absolute_asset_path, Content, ContentImporter, ContentProcessor, ContentSource, Error,
};
use dess_assets::{ShaderAsset, ShaderSource, ShaderStage, SpecializationConstant};
use normalize_path::NormalizePath;

impl ContentSource<ShaderContent> for ShaderSource {}

#[derive(Debug)]
pub struct ShaderContent {
    pub stage: ShaderStage,
    pub code: String,
    pub sources: HashSet<String>,
    pub defines: Vec<String>,
    pub specializations: Vec<(SpecializationConstant, u32)>,
}

impl Content for ShaderContent {}

#[derive(Debug, Default)]
pub struct ShaderImporter;

impl ContentImporter<ShaderContent, ShaderSource> for ShaderImporter {
    fn import(&self, source: ShaderSource) -> Result<ShaderContent, crate::Error> {
        let path = Path::new(&source.source).to_owned();
        let chunks = shader_prepper::process_file(
            get_absolute_asset_path(&path)?
                .to_str()
                .ok_or(Error::Fail)?,
            &mut FileIncludeProvider,
            path,
        )
        .map_err(|err| Error::ImportFailed(err.to_string()))?;

        let mut code = String::new();
        chunks.iter().for_each(|chunk| code += &chunk.source);
        let shader = ShaderContent {
            stage: source.stage,
            code,
            sources: chunks.iter().map(|x| x.file.to_owned()).collect(),
            defines: source.defines.clone().unwrap_or_default().to_vec(),
            specializations: source
                .specializations
                .clone()
                .unwrap_or_default()
                .iter()
                .map(|x| (*x.0, *x.1 as u32))
                .collect(),
        };

        Ok(shader)
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
pub struct ShaderContentProcessor;

impl ShaderContentProcessor {
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
                "-fspv-target-env=vulkan1.3",
                "-WX",
                "-Ges",
                "-HV 2021",
            ],
            &defines,
        )
        .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        Ok(spirv)
    }
}

impl ContentProcessor<ShaderContent, ShaderAsset> for ShaderContentProcessor {
    fn process(
        &self,
        _asset: dess_assets::AssetRef,
        _context: &crate::AssetProcessingContext,
        content: ShaderContent,
    ) -> Result<ShaderAsset, Error> {
        let code = Self::compile_shader(content.stage, &content.code, &content.defines)?;
        Ok(ShaderAsset {
            stage: content.stage,
            specializations: content.specializations.clone(),
            code,
        })
    }
}
