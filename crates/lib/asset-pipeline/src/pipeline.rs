use std::{
    fmt::Debug,
    fs::{self, File},
    io,
    path::Path,
};

use dess_assets::{Asset, AssetRef, GpuImage, GpuModel, GpuShader, GpuShaderStage};
use dess_common::traits::BinarySerialization;
use log::{error, info};

use crate::{
    build_bundle, cached_asset_path, compile_shaders::ShaderSource, AssetDatabase,
    AssetProcessingContext, CompileShader, Content, ContentImporter, ContentProcessor,
    CreateGpuImage, CreateGpuModel, Error, GltfSource, ImagePurpose, ImageSource, LoadedGltf,
    LoadedShaderCode, RawImage, ASSET_CACHE_PATH, ROOT_DATA_PATH,
};

#[derive(Debug)]
pub struct AssetPipeline {
    context: AssetProcessingContext,
    name: String,
}

impl AssetPipeline {
    pub fn new(name: &str) -> Self {
        if let Some(db) = AssetDatabase::try_load(name) {
            Self {
                context: AssetProcessingContext::from_database(&db),
                name: name.to_owned(),
            }
        } else {
            Self {
                context: AssetProcessingContext::default(),
                name: name.to_owned(),
            }
        }
    }

    pub fn save_db(&self) -> io::Result<()> {
        self.context.to_database().save(&self.name)
    }

    pub fn import_model(&self, path: &Path) -> AssetRef {
        self.context.import_model(&GltfSource::new(path))
    }

    pub fn import_vertex_shader(&self, path: &Path) -> AssetRef {
        self.context.import_shader(&ShaderSource {
            stage: GpuShaderStage::Vertex,
            path: path.into(),
        })
    }

    pub fn import_fragment_shader(&self, path: &Path) -> AssetRef {
        self.context.import_shader(&ShaderSource {
            stage: GpuShaderStage::Fragment,
            path: path.into(),
        })
    }

    pub fn import_image(&self, path: &Path, purpose: ImagePurpose) -> AssetRef {
        self.context
            .import_image(&ImageSource::from_file(path, purpose), None)
    }

    pub fn set_name(&self, asset: AssetRef, name: &str) {
        self.context.set_name(asset, name);
    }

    fn need_update(&self, asset: AssetRef) -> bool {
        if let Some(path) = self.context.get_owner(asset) {
            let source_timestamp = Path::new(ROOT_DATA_PATH)
                .join(path)
                .metadata()
                .unwrap()
                .modified()
                .unwrap();

            let asset_path = cached_asset_path(asset);
            if asset_path.exists() {
                asset_path.metadata().unwrap().created().unwrap() < source_timestamp
            } else {
                true
            }
        } else {
            // Or has no products, so we need to create it
            true
        }
    }

    pub fn process_pending_assets(&self) -> io::Result<()> {
        let mut need_work = true;
        fs::create_dir_all(Path::new(ASSET_CACHE_PATH))?;
        while need_work {
            need_work = false;
            let models_to_process = self.context.drain_models_to_process();
            need_work |= !models_to_process.is_empty();
            self.process_assets::<GpuModel, LoadedGltf, CreateGpuModel, GltfSource>(
                models_to_process,
            );
            let images_to_process = self.context.drain_images_to_process();
            need_work |= !images_to_process.is_empty();
            self.process_assets::<GpuImage, RawImage, CreateGpuImage, ImageSource>(
                images_to_process,
            );
            let shades_to_process = self.context.drain_shaders_to_process();
            need_work |= !shades_to_process.is_empty();
            self.process_assets::<GpuShader, LoadedShaderCode, CompileShader, ShaderSource>(
                shades_to_process,
            );
        }

        Ok(())
    }

    fn process_single_asset<T, C, P>(
        &self,
        asset: AssetRef,
        importer: impl ContentImporter<C> + Debug,
    ) -> Result<(), Error>
    where
        T: Asset + BinarySerialization,
        C: Content,
        P: ContentProcessor<C, T> + Default,
    {
        info!("Processing content {:?} into asset {}", importer, asset);
        let data = P::default().process(asset, &self.context, importer.import()?)?;
        data.serialize(&mut File::create(cached_asset_path(asset))?)?;

        Ok(())
    }

    fn process_assets<T, C, P, I>(&self, data: Vec<(AssetRef, I)>)
    where
        T: Asset + BinarySerialization,
        C: Content,
        P: ContentProcessor<C, T> + Default,
        I: ContentImporter<C> + Send + Debug,
    {
        let mut data = data;
        rayon::scope(|s| {
            data.drain(..).for_each(|(asset, importer)| {
                if self.need_update(asset) {
                    s.spawn(move |_| {
                        if let Err(err) = self.process_single_asset::<T, C, P>(asset, importer) {
                            error!("Asset processing failed: {:?}", err);
                        };
                    })
                }
            });
        });
    }

    pub fn bundle(self, path: &Path) -> io::Result<()> {
        build_bundle(self.context, path)
    }
}
