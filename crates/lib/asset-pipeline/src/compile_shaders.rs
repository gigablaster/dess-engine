use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use byte_slice_cast::AsSliceOf;
use dess_assets::{GpuShader, GpuShaderStage};
use log::{debug, error, info, warn};
use normalize_path::NormalizePath;
use spirv_tools::{
    error::{MessageCallback, MessageLevel},
    opt::Optimizer,
    val::Validator,
};

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
            get_absolute_asset_path(&self.path)
                .unwrap()
                .to_str()
                .unwrap(),
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

// regex shat itself.
struct Parser<'a> {
    tokens: Vec<&'a str>,
    current: usize,
}

impl<'a> Parser<'a> {
    pub fn new(code: &'a str) -> Self {
        let tokens = code.split_whitespace().collect::<Vec<_>>();
        Self { tokens, current: 0 }
    }

    pub fn find_one_of(&mut self, tokens: &[&str]) -> Option<&'a str> {
        while self.current < self.tokens.len() {
            let current_token = self.tokens[self.current];
            if tokens.iter().any(|x| *x == current_token) {
                return Some(current_token);
            }
            self.current += 1;
        }

        None
    }

    pub fn get_next(&self) -> Option<&'a str> {
        self.tokens.get(self.current + 1).copied()
    }

    pub fn move_forward(&mut self) {
        self.current += 1;
    }
}

impl CompileShader {
    fn compile_shader(
        stage: GpuShaderStage,
        code: &str,
        flags: &[String],
    ) -> Result<Vec<u8>, Error> {
        let profile = match stage {
            GpuShaderStage::Vertex => "vs_6_4",
            GpuShaderStage::Fragment => "ps_6_4",
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

    fn extract_flags(code: &str) -> Vec<String> {
        let mut defines = HashSet::new();
        let mut parser = Parser::new(code);
        while parser.find_one_of(&["#ifdef", "#if", "#ifndef"]).is_some() {
            if let Some(def) = parser.get_next() {
                parser.move_forward();
                if def.starts_with("HAVE_") || def.starts_with("USE_") {
                    defines.insert(def);
                }
            }
        }

        defines.iter().map(|x| x.to_string()).collect::<Vec<_>>()
    }
}

impl ContentProcessor<LoadedShaderCode, GpuShader> for CompileShader {
    fn process(
        &self,
        _asset: dess_assets::AssetRef,
        _context: &crate::AssetProcessingContext,
        content: LoadedShaderCode,
    ) -> Result<GpuShader, Error> {
        let defines = Self::extract_flags(&content.code);
        let mut shader = GpuShader::new(content.stage, &defines);
        let variation_count = (1 << defines.len()) as u32;

        for index in 0..variation_count {
            let mut current_defines = Vec::new();
            for (shift, def) in defines.iter().enumerate() {
                let mask = 1 << shift;
                if index & mask == mask {
                    current_defines.push(def.clone());
                }
            }

            shader
                .add_shader_variant(
                    index,
                    &Self::compile_shader(content.stage, &content.code, &current_defines)?,
                )
                .map_err(|err| Error::ProcessingFailed(err.to_string()))?;
        }

        Ok(shader)
    }
}
