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

use std::{collections::HashMap, ops::Deref, sync::Arc};

use ash::vk;

use crate::BackendResult;

use super::{Device, RenderPass};

pub struct PipelineShader {
    pub stage: vk::PipelineStageFlags,
    pub code: Vec<u8>,
}

pub struct Pipeline {
    device: Arc<Device>,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

pub struct PipelineDesc<'a> {
    pub render_pass: Arc<RenderPass>,
    pub depth_write: bool,
    pub face_cull: bool,
    pub shaders: Vec<&'a PipelineShader>,
}

impl<'a> PipelineDesc<'a> {
    pub fn new(render_pass: &Arc<RenderPass>) -> Self {
        Self {
            render_pass: render_pass.clone(),
            depth_write: true,
            face_cull: true,
            shaders: Vec::new(),
        }
    }

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;
        self
    }

    pub fn face_cull(mut self, value: bool) -> Self {
        self.face_cull = value;
        self
    }

    pub fn add_shader(mut self, shader: &'a PipelineShader) -> Self {
        self.shaders.push(shader);
        self
    }
}
