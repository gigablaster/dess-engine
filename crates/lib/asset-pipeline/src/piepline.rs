use std::{fs::File, io, path::Path};

use dess_assets::{Asset, AssetRef, GpuImage, GpuModel};
use dess_common::traits::BinarySerialization;
use log::error;

use crate::{
    build_bundle, cached_asset_path, AssetProcessingContext, Content, ContentImporter,
    ContentProcessor, CreateGpuImage, CreateGpuModel, Error, GltfSource, ImageSource, LoadedGltf,
    RawImage,
};

#[derive(Debug, Default)]
pub struct AssetPipeline {
    context: AssetProcessingContext,
}

impl AssetPipeline {
    pub fn import_model(&self, path: &Path) -> Result<AssetRef, Error> {
        if self.need_update(path, |context, path| context.get_model_id(path)) {
            let source = GltfSource::new(path);
            let asset = self.context.import_model(&source);

            Ok(asset)
        } else {
            Ok(self.context.get_model_id(path).unwrap())
        }
    }

    fn need_update<F>(&self, path: &Path, asset_fn: F) -> bool
    where
        F: FnOnce(&AssetProcessingContext, &Path) -> Option<AssetRef>,
    {
        let source_changed = path.metadata().unwrap().modified().unwrap();
        if let Some(asset) = asset_fn(&self.context, path) {
            // If newer than any of it's products
            cached_asset_path(asset)
                .metadata()
                .unwrap()
                .created()
                .unwrap()
                < source_changed
        } else {
            // Or has no products, so we need to create it
            true
        }
    }

    pub fn process_pending_assets(&self) {
        let mut need_work = true;
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
        }
    }

    fn process_single_asset<T, C, P>(
        &self,
        asset: AssetRef,
        importer: impl ContentImporter<C>,
    ) -> Result<(), Error>
    where
        T: Asset + BinarySerialization,
        C: Content,
        P: ContentProcessor<C, T> + Default,
    {
        let data = P::default().process(&self.context, importer.import()?)?;
        data.serialize(&mut File::create(cached_asset_path(asset))?)?;

        Ok(())
    }

    fn process_assets<T, C, P, I>(&self, data: Vec<(AssetRef, I)>)
    where
        T: Asset + BinarySerialization,
        C: Content,
        P: ContentProcessor<C, T> + Default,
        I: ContentImporter<C> + Send + Sync,
    {
        let mut data = data;
        rayon::scope(|s| {
            data.drain(..).for_each(|(asset, importer)| {
                s.spawn(move |_| {
                    if let Err(err) = self.process_single_asset::<T, C, P>(asset, importer) {
                        error!("Asset processing failed: {:?}", err);
                    };
                })
            });
        });
    }

    pub fn bundle(self, path: &Path) -> io::Result<()> {
        build_bundle(self.context, path)
    }
}
