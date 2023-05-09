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

use std::ffi::CStr;

use ash::vk;

use crate::BackendResult;

use super::{RenderPass, Shader};

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

pub struct PipelineDesc<'a> {
    pub render_pass: &'a RenderPass,
    pub depth_write: bool,
    pub face_cull: bool,
    pub shaders: Vec<&'a Shader>
}

impl<'a> PipelineDesc<'a> {
    pub fn new(render_pass: &'a RenderPass) -> Self {
        Self {
            render_pass,
            depth_write: true,
            face_cull: true,
            shaders: Vec::new()
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

    pub fn add_shader(mut self, shader: &'a Shader) -> Self {
        self.shaders.push(shader);
        self
    }
}

impl Pipeline {
    fn new(device: &ash::Device, desc: PipelineDesc) -> BackendResult<Self> {
        let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let shader_create_info = desc.shaders.iter().map(|shader| {
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(shader.stage)
                .module(shader.raw)
                .name(entry)
                .build()
        }).collect::<Vec<_>>();

    }
}
