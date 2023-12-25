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

use dess_assets::MeshMaterial;
use dess_backend::{
    BackendError, ImageLayout, UpdateBindGroupsContext, {BindGroupHandle, ImageHandle},
};
use smol_str::SmolStr;

use crate::{
    Error, Resource, ResourceContext, ResourceDependencies, ResourceHandle, ResourceLoader,
};

/// Material contains effect and a per-material descriptor set
/// for every effect technique.
///
/// Pipelines aren't created at this stage, they belong to render pass.
#[derive(Debug, Default)]
pub struct Material {
    images: HashMap<SmolStr, ResourceHandle<ImageHandle>>,
    resolved_images: HashMap<SmolStr, ImageHandle>,
    parameters: HashMap<SmolStr, f32>,
}

impl Resource for Material {
    fn dispose(&self, _ctx: &ResourceContext) {}
}

impl ResourceDependencies for Material {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.images
            .iter()
            .all(|(_, image)| ctx.is_image_finished(*image))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), Error> {
        self.images.clear();
        for (name, image) in self.images.iter() {
            self.resolved_images
                .insert(name.clone(), *ctx.resolve_image(*image)?);
        }
        Ok(())
    }
}

impl Material {
    pub fn new<T: ResourceLoader>(loader: &T, mesh_material: &MeshMaterial) -> Self {
        let images = mesh_material
            .images
            .iter()
            .map(|(name, asset)| (name.into(), loader.request_image(*asset)))
            .collect::<HashMap<_, _>>();
        Self {
            images,
            resolved_images: HashMap::default(),
            parameters: mesh_material
                .values
                .iter()
                .map(|(name, value)| (name.into(), *value))
                .collect(),
        }
    }

    pub fn bind_images(
        &self,
        handle: BindGroupHandle,
        ctx: &mut UpdateBindGroupsContext,
    ) -> Result<(), BackendError> {
        for (name, image) in &self.resolved_images {
            ctx.bind_image_by_name(handle, name, *image, ImageLayout::ShaderRead)?;
        }

        Ok(())
    }

    pub fn paramters(&self) -> &HashMap<SmolStr, f32> {
        &self.parameters
    }
}
