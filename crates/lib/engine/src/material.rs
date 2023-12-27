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

use dess_assets::{MeshBlendMode, MeshMaterial};
use dess_backend::{
    BackendResultExt, BindGroupLayoutDesc, BindType, ImageLayout, ShaderStage,
    {BindGroupHandle, ImageHandle},
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
    main_bind_group: BindGroupHandle,
    shadow_bind_group: BindGroupHandle,
    emissive_power: f32,
    alpha_cut: f32,
}

impl Resource for Material {
    fn dispose(&self, _ctx: &ResourceContext) {}
}

struct MainMaterialUniform {
    emissive_power: f32,
    alpha_cut: f32,
    _pad: [f32; 2],
}

struct ShadowMaterialUniform {
    alpha_cut: f32,
    _pad: [f32; 3],
}

impl ResourceDependencies for Material {
    fn is_finished(&self, ctx: &ResourceContext) -> bool {
        self.images
            .iter()
            .all(|(_, image)| ctx.is_image_finished(*image))
    }

    fn resolve(&mut self, ctx: &ResourceContext) -> Result<(), Error> {
        let mut images = HashMap::new();
        for (name, image) in self.images.iter() {
            images.insert(name.clone(), *ctx.resolve_image(*image)?);
        }
        ctx.device.with_bind_groups(|ctx| {
            for (name, image) in &images {
                ctx.bind_image_by_name(self.main_bind_group, name, *image, ImageLayout::ShaderRead)
                    .ignore_missing()?;
                ctx.bind_image_by_name(
                    self.shadow_bind_group,
                    name,
                    *image,
                    ImageLayout::ShaderRead,
                )
                .ignore_missing()?;
            }
            ctx.bind_uniform_by_name(
                self.main_bind_group,
                "material",
                &MainMaterialUniform {
                    emissive_power: self.emissive_power,
                    alpha_cut: self.alpha_cut,
                    _pad: [0.0; 2],
                },
            )?;
            ctx.bind_uniform_by_name(
                self.shadow_bind_group,
                "material",
                &ShadowMaterialUniform {
                    alpha_cut: self.alpha_cut,
                    _pad: [0.0; 3],
                },
            )?;

            Ok(())
        });
        Ok(())
    }
}

impl Material {
    pub fn new<T: ResourceLoader>(loader: &T, mesh_material: &MeshMaterial) -> Result<Self, Error> {
        let mut images = HashMap::new();
        images.insert("base", mesh_material.base);
        images.insert("normals", mesh_material.normals);
        images.insert("metallic_roughness", mesh_material.metallic_roughness);
        images.insert("occlusion", mesh_material.occlusion);
        images.insert("emissive", mesh_material.emissive);
        let images = images
            .into_iter()
            .map(|(name, image)| (name.into(), loader.request_image(image)))
            .collect();
        let alpha_cut = if let MeshBlendMode::AlphaTest(cut) = mesh_material.blend {
            cut
        } else {
            0.0
        };
        let main_bind_group = loader.render_device().create_bind_group_from_desc(
            &BindGroupLayoutDesc::default()
                .stage(ShaderStage::Vertex | ShaderStage::Fragment)
                .bind(0, "base", BindType::SampledImage, 1)
                .bind(1, "normals", BindType::SampledImage, 1)
                .bind(2, "metallic_roughness", BindType::SampledImage, 1)
                .bind(3, "occlusion", BindType::SampledImage, 1)
                .bind(4, "emissive", BindType::SampledImage, 1)
                .bind(5, "material", BindType::Uniform, 1),
        )?;
        let shadow_bind_group = loader.render_device().create_bind_group_from_desc(
            &BindGroupLayoutDesc::default()
                .stage(ShaderStage::Vertex | ShaderStage::Fragment)
                .bind(0, "base", BindType::SampledImage, 1)
                .bind(1, "material", BindType::Uniform, 1),
        )?;
        Ok(Self {
            images,
            main_bind_group,
            shadow_bind_group,
            alpha_cut,
            emissive_power: mesh_material.emission_power,
        })
    }
}
