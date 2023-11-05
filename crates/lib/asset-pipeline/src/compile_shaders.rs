use std::{
    fs,
    path::{Path, PathBuf},
};

use byte_slice_cast::AsSliceOf;
use dess_assets::{GpuShader, GpuShaderStage};
use normalize_path::NormalizePath;
use spirv_tools::{error::MessageCallback, opt::Optimizer};

use crate::{get_absolute_asset_path, Content, ContentImporter, ContentProcessor, Error};

#[derive(Debug, Clone)]
pub struct ShaderSource {
    pub stage: GpuShaderStage,
    pub path: PathBuf,
}

pub struct LoadedShaderCode {
    pub stage: GpuShaderStage,
    pub code: String,
}

impl Content for LoadedShaderCode {}

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

impl ContentImporter<LoadedShaderCode> for ShaderSource {
    fn import(&self) -> Result<LoadedShaderCode, crate::Error> {
        let chunks = shader_prepper::process_file(
            self.path.to_str().unwrap(),
            &mut FileIncludeProvider,
            self.path.clone(),
        )
        .map_err(|err| Error::ImportFailed(err.to_string()))?;

        let mut code = String::new();
        chunks.iter().for_each(|chunk| code += &chunk.source);

        Ok(LoadedShaderCode {
            stage: self.stage,
            code,
        })
    }
}

#[derive(Debug, Default)]
pub struct CompileShader;

struct OptCallbacks;

impl MessageCallback for OptCallbacks {
    fn on_message(&mut self, _msg: spirv_tools::error::Message) {}
}

impl ContentProcessor<LoadedShaderCode, GpuShader> for CompileShader {
    fn process(
        &self,
        _asset: dess_assets::AssetRef,
        _context: &crate::AssetProcessingContext,
        content: LoadedShaderCode,
    ) -> Result<GpuShader, Error> {
        let profile = match content.stage {
            GpuShaderStage::Vertex => "vs_6_4",
            GpuShaderStage::Fragment => "ps_6_4",
        };
        let mut shader = GpuShader::new(content.stage, &[]);
        let spirv = hassle_rs::compile_hlsl(
            "",
            &content.code,
            "main",
            profile,
            &[
                "-spirv",
                "-fspv-target-env=vulkan1.1",
                "-WX",
                "-Ges",
                "-HV 2021",
            ],
            &[],
        )
        .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        let mut optimizer = spirv_tools::opt::create(Some(spirv_tools::TargetEnv::Vulkan_1_1));
        optimizer.register_performance_passes();
        let data = spirv.as_slice_of::<u32>().unwrap();
        let spirv = optimizer
            .optimize(data, &mut OptCallbacks {}, None)
            .unwrap();

        shader
            .add_shader_variant(&[], spirv.as_bytes())
            .map_err(|err| Error::ProcessingFailed(err.to_string()))?;

        Ok(shader)
    }
}