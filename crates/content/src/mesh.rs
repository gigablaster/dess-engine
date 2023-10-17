use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use dess_common::{
    mesh::{
        CpuMesh, CpuModel, EffectInfo, BASE_COLOR_TEXTURE, METALLIC_ROUGHNESS_TEXTURE,
        NORMAL_MAP_TEXTURE, OCCLUSION_TEXTURE,
    },
    traits::BinarySerialization,
};
use gltf::{
    image,
    material::{NormalTexture, OcclusionTexture},
    mesh::Mode,
    texture, Document,
};
use normalize_path::NormalizePath;

use crate::{Content, ContentImporter, ImportContext, ImportError};

impl Content for CpuModel {
    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.serialize(&mut File::create(path)?)
    }
}

const PBR_OPAQUE_EFFECT: &str = "bdrf";

#[derive(Debug, Default)]
pub struct GltfModelImporter {}

impl GltfModelImporter {
    fn set_texture(effect: &mut EffectInfo, root: &Path, name: &str, info: &Option<texture::Info>) {
        if let Some(texture) = info {
            Self::bind_texture(effect, root, name, texture.texture());
        }
    }

    fn set_normal_texture(effect: &mut EffectInfo, root: &Path, normal: &Option<NormalTexture>) {
        if let Some(texture) = normal {
            Self::bind_texture(effect, root, NORMAL_MAP_TEXTURE, texture.texture());
        }
    }

    fn set_occlusion_texture(
        effect: &mut EffectInfo,
        root: &Path,
        occlusion: &Option<OcclusionTexture>,
    ) {
        if let Some(texture) = occlusion {
            Self::bind_texture(effect, root, OCCLUSION_TEXTURE, texture.texture());
        }
    }

    fn bind_texture(effect: &mut EffectInfo, root: &Path, name: &str, texture: texture::Texture) {
        if let image::Source::Uri { uri, .. } = texture.source().source() {
            let mut path: PathBuf = root
                .join(Path::new(uri))
                .normalize()
                .as_os_str()
                .to_ascii_lowercase()
                .into();
            path.set_extension("dds");
            effect.add_texture(name, path.to_str().unwrap());
        }
    }

    fn collect_effects(
        document: &Document,
        context: &ImportContext,
    ) -> HashMap<String, EffectInfo> {
        let mut effects = HashMap::new();
        document.materials().for_each(|mat| {
            if let Some(name) = mat.name() {
                let mut effect = EffectInfo::new(PBR_OPAQUE_EFFECT);
                Self::set_texture(
                    &mut effect,
                    context.destination_dir,
                    BASE_COLOR_TEXTURE,
                    &mat.pbr_metallic_roughness().base_color_texture(),
                );
                Self::set_texture(
                    &mut effect,
                    context.destination_dir,
                    METALLIC_ROUGHNESS_TEXTURE,
                    &mat.pbr_metallic_roughness().metallic_roughness_texture(),
                );
                Self::set_normal_texture(
                    &mut effect,
                    context.destination_dir,
                    &mat.normal_texture(),
                );
                Self::set_occlusion_texture(
                    &mut effect,
                    context.destination_dir,
                    &mat.occlusion_texture(),
                );

                effects.insert(name.into(), effect);
            }
        });

        effects
    }

    fn generate_meshes(target: &mut CpuModel, mesh: &gltf::Mesh, buffer: &gltf::Buffer) {
        mesh.primitives().for_each(|surface| {
            if surface.mode() != Mode::Triangles {
                return;
            }
        });
    }
}

impl ContentImporter for GltfModelImporter {
    fn can_handle(&self, path: &std::path::Path) -> bool {
        path.extension()
            .map(|x| {
                x.to_ascii_lowercase()
                    .to_str()
                    .map(|x| x == "gltf")
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn target_name(&self, path: &std::path::Path) -> std::path::PathBuf {
        let mut path = path.to_path_buf();
        path.set_extension("mesh");
        path
    }

    fn import(
        &self,
        path: &std::path::Path,
        context: &ImportContext,
    ) -> Result<Box<dyn Content>, ImportError> {
        let (document, buffers, images) = gltf::import(path)?;

        let effects = Self::collect_effects(&document, context);
        Ok(Box::<CpuModel>::default())
    }
}
