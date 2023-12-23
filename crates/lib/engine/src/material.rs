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

use std::{collections::HashMap, fmt::Debug};

use bytes::Bytes;
use dess_assets::AssetRef;
use dess_backend::{
    BackendResultExt, ImageLayout,
    {DescriptorHandle, ImageHandle, ProgramHandle, PER_MATERIAL_BINDING_SLOT},
};
use smol_str::SmolStr;

use crate::{
    Error, Resource, ResourceContext, ResourceDependencies, ResourceHandle, ResourceLoader,
};

const MATERIAL_UNIFORM_NAME: &str = "material";
/// Material contains effect and a per-material descriptor set
/// for every effect technique.
///
/// Pipelines aren't created at this stage, they belong to render pass.
#[derive(Debug, Default)]
pub struct Material {
    program: ProgramHandle,
    images: HashMap<SmolStr, ResourceHandle<ImageHandle>>,
    uniform: Bytes,
    ds: DescriptorHandle,
}

impl Resource for Material {}

impl ResourceDependencies for Material {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.images
            .iter()
            .all(|(_, image)| ctx.is_image_finished(*image))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), Error> {
        let mut images = HashMap::new();
        for (name, image) in self.images.iter() {
            images.insert(name, ctx.resolve_image(*image)?);
        }
        ctx.device.with_descriptors(|ctx| {
            let ds = ctx.from_program(self.program, PER_MATERIAL_BINDING_SLOT)?;
            for (name, image) in images {
                ctx.bind_image_by_name(ds, name, *image, ImageLayout::ShaderRead)
                    .ignore_missing()?;
            }
            if !self.uniform.is_empty() {
                ctx.bind_uniform_by_name(ds, MATERIAL_UNIFORM_NAME, &self.uniform)
                    .ignore_missing()?;
            }
            self.ds = ds;
            Ok(())
        })?;

        Ok(())
    }
}

impl Material {
    pub fn new<T: ResourceLoader>(
        loader: &T,
        program: ProgramHandle,
        images: &[(String, AssetRef)],
        uniform: Bytes,
    ) -> Self {
        let images = images
            .iter()
            .map(|(name, asset)| (name.into(), loader.request_image(*asset)))
            .collect::<HashMap<_, _>>();
        Self {
            program,
            images,
            uniform,
            ds: DescriptorHandle::default(),
        }
    }
}
