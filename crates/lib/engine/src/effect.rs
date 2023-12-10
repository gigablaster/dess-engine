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
    collections::HashMap,
    fmt::Display,
    hash::{Hash, Hasher},
    sync::Arc,
};

use ash::vk;
use bevy_tasks::AsyncComputeTaskPool;
use dess_assets::Asset;
use dess_backend::vulkan::{Device, PipelineCreateDesc, PipelineHandle, ProgramHandle};
use smol_str::SmolStr;

use crate::{
    AssetCache, AssetCacheFns, AssetHandle, EngineAsset, EngineAssetKey, Error, ProgramSource,
};

#[derive(Debug, Hash, Clone)]
pub struct TechniqueSource {
    program: ProgramSource,
    desc: PipelineCreateDesc,
    color_attachments: Vec<vk::Format>,
    depth_attachment: Option<vk::Format>,
}

#[derive(Debug, Clone)]
pub struct EffectSource {
    techniques: HashMap<SmolStr, TechniqueSource>,
}

impl Display for EffectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Effect(")?;
        for tech in self.techniques.iter() {
            write!(f, "({} -> {:?}), ", tech.0, tech.1)?;
        }
        write!(f, ")")
    }
}

impl EngineAssetKey for EffectSource {
    fn key(&self) -> u64 {
        let mut hasher = siphasher::sip::SipHasher::default();
        self.techniques.iter().for_each(|x| {
            x.0.hash(&mut hasher);
            x.1.hash(&mut hasher);
        });
        hasher.finish()
    }
}

#[derive(Debug)]
struct RenderEffectTechinque {
    pipeline: PipelineHandle,
    program: AssetHandle<ProgramHandle>,
    desc: PipelineCreateDesc,
    handle: Arc<ProgramHandle>,
    color_attachments: Vec<vk::Format>,
    depth_attachment: Option<vk::Format>,
}

impl RenderEffectTechinque {
    fn new<T: AssetCacheFns>(asset_cache: &T, source: TechniqueSource) -> Self {
        Self {
            pipeline: PipelineHandle::default(),
            program: asset_cache.request_program(&source.program),
            desc: source.desc,
            handle: Arc::default(),
            color_attachments: source.color_attachments,
            depth_attachment: source.depth_attachment,
        }
    }
}

impl EngineAsset for RenderEffectTechinque {
    fn is_ready<T: AssetCacheFns>(&self, asset_cache: &T) -> bool {
        asset_cache.is_program_loaded(self.program)
    }

    fn resolve<T: AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), crate::Error> {
        self.handle = asset_cache.resolve_program(self.program)?;

        Ok(())
    }
}

impl RenderEffectTechinque {
    async fn create_pipeline(&self, device: &Device) -> Result<PipelineHandle, Error> {
        Ok(device.create_pipeline(
            self.handle.as_ref().clone(),
            &self.desc,
            &self.color_attachments,
            self.depth_attachment,
        )?)
    }
}

/// Effect
///
/// Set of named pipelines, created from program and pipeline descriptor.
/// Used by materials to get pipelines for different passes.
///
/// Each pipeline might or might not use different program.
#[derive(Debug, Default)]
pub struct RenderEffect {
    techinques: HashMap<SmolStr, RenderEffectTechinque>,
}

impl EngineAsset for RenderEffect {
    fn is_ready<T: AssetCacheFns>(&self, asset_cache: &T) -> bool {
        self.techinques.iter().all(|(_, x)| x.is_ready(asset_cache))
    }

    fn resolve<T: AssetCacheFns>(&mut self, asset_cache: &T) -> Result<(), Error> {
        let mut programs = Vec::with_capacity(self.techinques.len());
        for (_, tech) in self.techinques.iter_mut() {
            tech.resolve(asset_cache)?;
            programs.push(tech);
        }

        for (index, pipeline) in AsyncComputeTaskPool::get()
            .scope(|s| {
                for tech in programs.iter() {
                    s.spawn(tech.create_pipeline(asset_cache.render_device()))
                }
            })
            .into_iter()
            .enumerate()
        {
            programs[index].pipeline = pipeline?;
        }
        Ok(())
    }
}

impl RenderEffect {
    pub fn new<T: AssetCacheFns>(asset_cache: &T, source: EffectSource) -> Self {
        Self {
            techinques: source
                .techniques
                .into_iter()
                .map(|(name, techinque)| (name, RenderEffectTechinque::new(asset_cache, techinque)))
                .collect::<HashMap<_, _>>(),
        }
    }
}
