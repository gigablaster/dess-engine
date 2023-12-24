use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use dess_assets::{get_absolute_asset_path, get_relative_asset_path, ShaderAsset, ShaderSource};
use dess_backend::ShaderStage;
use shader_prepper::SourceChunk;

use crate::{is_asset_changed, read_to_end, AssetImporter, Error};

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

fn import_shader(path: &str) -> Result<Vec<SourceChunk>, Error> {
    let mut include_provider = IncludeProvider {};
    let chunks =
        shader_prepper::process_file(path, &mut include_provider, Path::new("").to_path_buf())
            .map_err(|x| Error::ProcessingFailed(x.to_string()))?;
    Ok(chunks)
}

fn compile_shader(
    name: &str,
    chunks: &[SourceChunk],
    stage: ShaderStage,
) -> Result<ShaderAsset, Error> {
    let mut source = String::new();
    chunks.iter().for_each(|x| source += &x.source);
    let profile = match stage {
        _ if stage == ShaderStage::Vertex => "vs_6_4",
        _ if stage == ShaderStage::Fragment => "ps_6_4",
        _ if stage == ShaderStage::Compute => "cs_6_4",
        _ => return Err(Error::ProcessingFailed("Unknown shader type".to_owned())),
    };
    let code = hassle_rs::compile_hlsl(
        name,
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
    .map_err(|x| Error::ProcessingFailed(x.to_string()))?
    .into();
    Ok(ShaderAsset { code })
}

impl AssetImporter for ShaderSource {
    fn import(
        &self,
        _ctx: &dyn crate::ImportContext,
    ) -> Result<Arc<dyn dess_assets::Asset>, Error> {
        let chunks = import_shader(&self.path)?;
        Ok(Arc::new(compile_shader(&self.path, &chunks, self.stage)?))
    }

    fn is_changed(&self, timestamp: std::time::SystemTime) -> bool {
        match import_shader(&self.path) {
            Ok(chunks) => chunks
                .iter()
                .any(|chunk| is_asset_changed(&chunk.file, timestamp)),
            Err(_) => true,
        }
    }
}
