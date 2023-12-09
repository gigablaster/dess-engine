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

use std::collections::HashMap;

use dess_assets::{ImageSource, MeshMaterial};
use dess_backend::vulkan::{DescriptorHandle, PipelineHandle};
use dess_common::{Handle, Pool};

#[derive(Debug, Default)]
pub enum MeshRenderPass {
    #[default]
    Main,
    Shadow,
}

#[derive(Debug, Default)]
pub enum MeshBlending {
    #[default]
    Opaque,
    AlphaBlend,
}

const ALPHA_TEST_MASK: usize = 1 << 0;
const BLEND_MASK: usize = 1 << 1;
const PASS_MASK: usize = 1 << 2;
const MAX_PIPELINES: usize = 1 << 3;
