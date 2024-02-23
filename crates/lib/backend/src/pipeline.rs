// Copyright (C) 2023-2024 gigablaster

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
    ffi::CString,
    fs::File,
    io::{self},
    mem,
    path::Path,
    slice,
    sync::Arc,
};

use ash::vk::{self};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use uuid::Uuid;

use crate::{AsVulkan, Program, RenderPass, Result};

use super::{Device, PhysicalDevice};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PipelineBlendDesc {
    pub src: vk::BlendFactor,
    pub dst: vk::BlendFactor,
    pub op: vk::BlendOp,
}

impl PipelineBlendDesc {
    pub fn new(src: vk::BlendFactor, dst: vk::BlendFactor, op: vk::BlendOp) -> Self {
        Self { src, dst, op }
    }
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
pub struct InputVertexStreamDesc {
    attributes: Vec<(vk::Format, usize, usize)>,
}

pub trait VertexAttribute: Sized + Copy {
    fn vk_format() -> vk::Format;
}

impl VertexAttribute for glam::Vec2 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32_SFLOAT
    }
}

impl VertexAttribute for glam::Vec3 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32_SFLOAT
    }
}

impl VertexAttribute for glam::Vec4 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32A32_SFLOAT
    }
}

impl VertexAttribute for glam::UVec2 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32_UINT
    }
}

impl VertexAttribute for glam::UVec3 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32_UINT
    }
}

impl VertexAttribute for glam::UVec4 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32A32_UINT
    }
}

impl VertexAttribute for glam::IVec2 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32_SINT
    }
}

impl VertexAttribute for glam::IVec3 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32_SINT
    }
}

impl VertexAttribute for glam::IVec4 {
    fn vk_format() -> vk::Format {
        vk::Format::R32G32B32A32_SINT
    }
}

impl VertexAttribute for glam::U16Vec2 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16_UINT
    }
}

impl VertexAttribute for glam::U16Vec3 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16B16_UINT
    }
}

impl VertexAttribute for glam::U16Vec4 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16B16A16_UINT
    }
}

impl VertexAttribute for glam::I16Vec2 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16_SINT
    }
}

impl VertexAttribute for glam::I16Vec3 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16B16_SINT
    }
}

impl VertexAttribute for glam::I16Vec4 {
    fn vk_format() -> vk::Format {
        vk::Format::R16G16B16A16_SINT
    }
}

impl InputVertexStreamDesc {
    pub fn attrubute<T: VertexAttribute>(mut self, offset: usize) -> Self {
        self.attributes
            .push((T::vk_format(), offset, mem::size_of::<T>()));
        self
    }

    fn build(&self, binding: usize) -> (usize, Vec<vk::VertexInputAttributeDescription>) {
        let stride = self.attributes.iter().map(|x| x.1 + x.2).max().unwrap();
        let attributes = self
            .attributes
            .iter()
            .enumerate()
            .map(|(index, attr)| vk::VertexInputAttributeDescription {
                location: index as u32,
                binding: binding as u32,
                format: attr.0,
                offset: attr.1 as u32,
            })
            .collect();

        (stride, attributes)
    }
}

/// Data to create pipeline.
///
/// Contains all data to create new pipeline.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct RasterPipelineCreateDesc {
    /// Blend data, None if opaque. Order: color, alpha
    pub blend: Option<(PipelineBlendDesc, PipelineBlendDesc)>,
    /// Culling
    pub cull: Option<vk::CullModeFlags>,
    /// Depth testing
    pub depth_test: Option<vk::CompareOp>,
    /// Depth writing
    pub depth_write: bool,
    /// Vertex streams layout
    pub streams: Vec<InputVertexStreamDesc>,
    /// Layouts
    pub layouts: Vec<vk::DescriptorSetLayout>,
}

impl RasterPipelineCreateDesc {
    pub fn blending(mut self, color: PipelineBlendDesc, alpha: PipelineBlendDesc) -> Self {
        self.blend = Some((color, alpha));

        self
    }

    pub fn cull(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull = Some(mode);

        self
    }

    pub fn alpha_blend(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC1_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC1_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn premultiplied(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::ONE,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::ONE,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn additive(mut self) -> Self {
        self.blend = Some((
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE,
                vk::BlendOp::ADD,
            ),
            PipelineBlendDesc::new(
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE,
                vk::BlendOp::ADD,
            ),
        ));
        self
    }

    pub fn depth_write(mut self) -> Self {
        self.depth_write = true;

        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);

        self
    }

    pub fn vertex_stream(mut self, stream: InputVertexStreamDesc) -> Self {
        self.streams.push(stream);
        self
    }

    pub fn layout(mut self, layout: vk::DescriptorSetLayout) -> Self {
        self.layouts.push(layout);
        self
    }
}

pub fn compile_raster_pipeline(
    device: &Arc<Device>,
    program: &Program,
    render_pass: &RenderPass,
    subpass: usize,
    desc: &RasterPipelineCreateDesc,
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    let shader_create_info = program
        .shaders()
        .map(|shader| {
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(shader.stage())
                .module(shader.as_vk())
                .name(&CString::new(shader.entry()).unwrap())
                .build()
        })
        .collect::<Vec<_>>();

    let assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false)
        .build();

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&dynamic_states)
        .build();

    let streams = desc
        .streams
        .iter()
        .enumerate()
        .map(|(index, stream)| stream.build(index))
        .collect::<Vec<_>>();

    let strides = streams
        .iter()
        .map(|(stride, _)| stride)
        .copied()
        .collect::<Vec<_>>();
    let attributes = streams
        .iter()
        .flat_map(|(_, attributes)| attributes)
        .copied()
        .collect::<Vec<_>>();
    let vertex_binding_desc = strides
        .iter()
        .enumerate()
        .map(|(index, _)| {
            vk::VertexInputBindingDescription::builder()
                .stride(strides[index] as _)
                .binding(attributes[index].binding)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()
        })
        .collect::<Vec<_>>();

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&vertex_binding_desc)
        .vertex_attribute_descriptions(&attributes)
        .build();

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(slice::from_ref(&vk::Viewport::default()))
        .scissors(slice::from_ref(&vk::Rect2D::default()))
        .build();

    let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_bias_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
        .cull_mode(desc.cull.unwrap_or(vk::CullModeFlags::NONE))
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();

    let mut depthstencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .stencil_test_enable(false)
        .depth_write_enable(desc.depth_write);

    if let Some(depth_compare) = desc.depth_test {
        depthstencil_state = depthstencil_state
            .depth_test_enable(true)
            .depth_compare_op(depth_compare);
    }

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_attachment = if let Some((color, alpha)) = desc.blend {
        color_blend_attachment
            .blend_enable(true)
            .src_color_blend_factor(color.src)
            .dst_color_blend_factor(color.dst)
            .color_blend_op(color.op)
            .src_alpha_blend_factor(alpha.src)
            .dst_alpha_blend_factor(alpha.dst)
            .alpha_blend_op(alpha.op)
    } else {
        color_blend_attachment.blend_enable(false)
    }
    .build();
    let blending_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(slice::from_ref(&color_blend_attachment))
        .logic_op_enable(false)
        .build();

    let pipeline_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&desc.layouts)
        .build();
    let pipeline_layout = unsafe {
        device
            .get()
            .create_pipeline_layout(&pipeline_create_info, None)
    }?;
    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .layout(pipeline_layout)
        .render_pass(render_pass.as_vk())
        .subpass(subpass as _)
        .stages(&shader_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .viewport_state(&viewport_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&blending_state)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly_state_create_info)
        .rasterization_state(&rasterizer_state)
        .depth_stencil_state(&depthstencil_state)
        .build();

    let pipeline = unsafe {
        device.get().create_graphics_pipelines(
            vk::PipelineCache::null(),
            slice::from_ref(&pipeline_create_info),
            None,
        )
    }?[0];

    Ok((pipeline, pipeline_layout))
}

const MAGICK: [u8; 4] = *b"PLCH";
const VERSION: u32 = 1;

#[derive(Debug, PartialEq, Eq)]
struct Header {
    pub magic: [u8; 4],
    pub version: u32,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            magic: MAGICK,
            version: VERSION,
        }
    }
}

impl Header {
    pub fn write<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_u32::<LittleEndian>(self.version)?;

        Ok(())
    }

    pub fn read<R: io::Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        Ok(Self {
            magic,
            version: r.read_u32::<LittleEndian>()?,
        })
    }

    pub fn validate(&self) -> bool {
        self.magic == MAGICK && self.version >= VERSION
    }
}

#[derive(Debug)]
struct PipelineDiskCache {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: Uuid,
    data: Vec<u8>,
}

impl PipelineDiskCache {
    pub fn new(pdevice: &PhysicalDevice, data: &[u8]) -> Self {
        let vendor_id = pdevice.properties().vendor_id;
        let device_id = pdevice.properties().device_id;
        let driver_version = pdevice.properties().driver_version;
        let uuid = Uuid::from_bytes(pdevice.properties().pipeline_cache_uuid);

        Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data: data.to_vec(),
        }
    }

    pub fn read<R: io::Read>(mut r: R) -> io::Result<Self> {
        if !Header::read(&mut r)?.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Wrong pipeline cache header",
            ));
        }
        Ok(Self {
            vendor_id: r.read_u32::<LittleEndian>()?,
            device_id: r.read_u32::<LittleEndian>()?,
            driver_version: r.read_u32::<LittleEndian>()?,
            uuid: Uuid::from_u128(r.read_u128::<LittleEndian>()?),
            data: Self::read_data(&mut r)?,
        })
    }

    fn read_data<R: io::Read>(r: &mut R) -> io::Result<Vec<u8>> {
        let size = r.read_u32::<LittleEndian>()?;
        let mut bytes = vec![0u8; size as usize];
        r.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn save<W: io::Write>(&self, mut w: W) -> io::Result<()> {
        Header::default().write(&mut w)?;
        w.write_u32::<LittleEndian>(self.vendor_id)?;
        w.write_u32::<LittleEndian>(self.device_id)?;
        w.write_u32::<LittleEndian>(self.driver_version)?;
        w.write_u128::<LittleEndian>(self.uuid.as_u128())?;
        w.write_u32::<LittleEndian>(self.data.len() as _)?;
        w.write_all(&self.data)?;
        Ok(())
    }
}

pub fn load_or_create_pipeline_cache<P: AsRef<Path>>(
    device: &Device,
    path: P,
) -> Result<vk::PipelineCache> {
    let data = if let Ok(cache) = PipelineDiskCache::read(File::open(path)?) {
        if cache.vendor_id == device.physical_device().properties().vendor_id
            && cache.device_id == device.physical_device().properties().device_id
            && cache.driver_version == device.physical_device().properties().driver_version
            && cache.uuid
                == Uuid::from_bytes(device.physical_device().properties().pipeline_cache_uuid)
        {
            Some(cache.data)
        } else {
            None
        }
    } else {
        None
    };

    let create_info = if let Some(data) = &data {
        vk::PipelineCacheCreateInfo::builder().initial_data(data)
    } else {
        vk::PipelineCacheCreateInfo::builder()
    }
    .build();

    let cache = match unsafe { device.get().create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            let create_info = vk::PipelineCacheCreateInfo::builder().build();
            unsafe { device.get().create_pipeline_cache(&create_info, None) }?
        }
    };

    Ok(cache)
}

pub fn save_pipeline_cache<P: AsRef<Path>>(
    device: &Device,
    cache: vk::PipelineCache,
    path: P,
) -> io::Result<()> {
    let data = unsafe { device.get().get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    PipelineDiskCache::new(device.physical_device(), &data).save(File::create(path)?)
}
