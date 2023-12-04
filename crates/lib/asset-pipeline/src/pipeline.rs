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
    fmt::Debug,
    fs::{self, File},
    io,
    path::Path,
};

use dess_assets::{
    Asset, AssetRef, ImageAsset, ImagePurpose, SceneAsset, ShaderAsset, ShaderSource,
};
use log::{error, info};

use crate::{
    build_bundle, get_cached_asset_path, AssetDatabase, AssetProcessingContext, Content,
    ContentImporter, ContentProcessor, ContentSource, Error, GltfContent, GltfContentProcessor,
    GltfImporter, GltfSource, ImageContent, ImageContentProcessor, ImageImporter, ImageSource,
    OfflineAssetProcessingContext, ShaderContent, ShaderContentProcessor, ShaderImporter,
    ASSET_CACHE_PATH, ROOT_DATA_PATH,
};

#[derive(Debug)]
pub struct AssetPipeline {
    context: OfflineAssetProcessingContext,
    name: String,
}

impl AssetPipeline {
    pub fn new(name: &str) -> Self {
        if let Some(db) = AssetDatabase::try_load(name) {
            Self {
                context: OfflineAssetProcessingContext::from_database(&db),
                name: name.to_owned(),
            }
        } else {
            Self {
                context: OfflineAssetProcessingContext::default(),
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

    pub fn import_shader(&self, shader: ShaderSource) -> AssetRef {
        self.context.import_shader(&shader)
    }

    pub fn import_image(&self, path: &Path, purpose: ImagePurpose) -> AssetRef {
        self.context
            .import_image(&ImageSource::from_file(path, purpose), None)
    }

    pub fn set_name(&self, asset: AssetRef, name: &str) {
        self.context.set_name(asset, name);
    }

    fn need_update<T: ContentSource<U>, U: Content>(&self, source: &T) -> bool {
        let asset = source.asset_ref();
        if let Some(path) = self.context.get_owner(asset) {
            let source_timestamp = Path::new(ROOT_DATA_PATH)
                .join(path)
                .metadata()
                .unwrap()
                .modified()
                .unwrap();

            let asset_path = get_cached_asset_path(asset);
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
            self.process_assets::<SceneAsset, GltfContent, GltfSource, GltfContentProcessor, GltfImporter>(
                models_to_process,
            );
            let images_to_process = self.context.drain_images_to_process();
            need_work |= !images_to_process.is_empty();
            self.process_assets::<ImageAsset, ImageContent, ImageSource, ImageContentProcessor, ImageImporter>(
                images_to_process,
            );
            let shades_to_process = self.context.drain_effects_to_process();
            need_work |= !shades_to_process.is_empty();
            self.process_assets::<ShaderAsset, ShaderContent, ShaderSource, ShaderContentProcessor, ShaderImporter>(
                shades_to_process,
            );
        }

        Ok(())
    }

    fn process_single_asset<T, C, S, P, I>(&self, source: S, importer: I) -> Result<(), Error>
    where
        T: Asset,
        C: Content,
        S: ContentSource<C> + Debug,
        P: ContentProcessor<C, T> + Default,
        I: ContentImporter<C, S> + Send + Debug,
    {
        let asset = source.asset_ref();
        info!("Processing content {:?} into asset {}", source, asset);
        let data = P::default().process(asset, &self.context, importer.import(source)?)?;
        data.serialize(&mut File::create(get_cached_asset_path(asset))?)?;

        Ok(())
    }

    fn process_assets<T, C, S, P, I>(&self, data: Vec<S>)
    where
        T: Asset,
        C: Content,
        S: ContentSource<C> + Debug + Clone,
        P: ContentProcessor<C, T> + Default,
        I: ContentImporter<C, S> + Default + Send + Debug,
    {
        let mut data = data;
        rayon::scope(|s| {
            data.drain(..).for_each(|source| {
                if self.need_update(&source) {
                    let source = source.clone();
                    s.spawn(move |_| {
                        if let Err(err) =
                            self.process_single_asset::<T, C, S, P, I>(source, I::default())
                        {
                            error!("Asset processing failed: {:?}", err);
                        };
                    })
                }
            });
        });
    }

    pub fn bundle(self, name: &str) -> io::Result<()> {
        build_bundle(self.context, name)
    }
}
