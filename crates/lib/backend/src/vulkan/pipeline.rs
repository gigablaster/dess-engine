use std::{
    fs::{self, File},
    hash::Hash,
    io, slice,
};

use arrayvec::ArrayVec;
use ash::vk::{self};
use bevy_tasks::AsyncComputeTaskPool;
use directories::ProjectDirs;
use log::{debug, error, info};
use speedy::{Context, Readable, Writable};
use uuid::Uuid;

use crate::{BackendError, BackendResult};

use super::{
    Device, PhysicalDevice, ProgramHandle, RasterPipelineHandle, Shader, MAX_COLOR_ATTACHMENTS,
    MAX_SHADERS,
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BlendDesc {
    pub src_blend: vk::BlendFactor,
    pub dst_blend: vk::BlendFactor,
    pub op: vk::BlendOp,
}

#[derive(Debug)]
pub struct VertexAttributeDesc {
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub stride: usize,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct InputVertexAttributeDesc {
    pub format: vk::Format,
    pub locaion: u32,
    pub binding: u32,
    pub offset: u32,
}

impl From<InputVertexAttributeDesc> for vk::VertexInputAttributeDescription {
    fn from(value: InputVertexAttributeDesc) -> Self {
        Self {
            location: value.locaion,
            binding: value.binding,
            format: value.format,
            offset: value.offset,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct InputVertexStreamDesc {
    pub attributes: &'static [InputVertexAttributeDesc],
    pub stride: u32,
}

/// Data to create pipeline.
///
/// Contains all data to create new pipeline.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RasterPipelineCreateDesc {
    /// Associated program
    pub program: ProgramHandle,
    /// Render pass layout
    pub pass_layout: RenderPassLayout,
    /// Blend data, None if opaque. Order: color, alpha
    pub blend: Option<(BlendDesc, BlendDesc)>,
    /// Depth comparison op, None if no depth test is happening
    pub depth_test: Option<vk::CompareOp>,
    /// true if we write into depth
    pub depth_write: bool,
    /// Culling information, None if we don't do culling
    pub cull: Option<(vk::CullModeFlags, vk::FrontFace)>,
    /// Vertex streams layout
    pub streams: &'static [InputVertexStreamDesc],
}

pub trait PipelineVertex: Sized {
    fn vertex_streams() -> &'static [InputVertexStreamDesc];
}

impl RasterPipelineCreateDesc {
    pub fn new<T: PipelineVertex>(program: ProgramHandle, pass_layout: RenderPassLayout) -> Self {
        Self {
            program,
            pass_layout,
            blend: None,
            depth_test: Some(vk::CompareOp::LESS),
            depth_write: true,
            cull: Some((vk::CullModeFlags::BACK, vk::FrontFace::CLOCKWISE)),
            streams: T::vertex_streams(),
        }
    }

    pub fn blend(mut self, color: BlendDesc, alpha: BlendDesc) -> Self {
        self.blend = Some((color, alpha));

        self
    }

    pub fn depth_test(mut self, value: vk::CompareOp) -> Self {
        self.depth_test = Some(value);

        self
    }

    pub fn depth_write(mut self, value: bool) -> Self {
        self.depth_write = value;

        self
    }

    pub fn cull(mut self, mode: vk::CullModeFlags, front: vk::FrontFace) -> Self {
        self.cull = Some((mode, front));

        self
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RenderPassLayout {
    pub color_attachments: ArrayVec<vk::Format, MAX_COLOR_ATTACHMENTS>,
    pub depth_attachment: Option<vk::Format>,
}

impl Device {
    async fn compile_raster_pipeline(
        &self,
        handle: RasterPipelineHandle,
        shaders: ArrayVec<Shader, MAX_SHADERS>,
        layout: vk::PipelineLayout,
        desc: RasterPipelineCreateDesc,
    ) -> BackendResult<(RasterPipelineHandle, vk::Pipeline, vk::PipelineLayout)> {
        debug!("Compile pipeline {:?}", desc);

        let shader_create_info = shaders
            .iter()
            .map(|shader| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(shader.stage)
                    .module(shader.raw)
                    .name(&shader.entry)
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

        let strides = desc.streams.iter().map(|x| x.stride).collect::<Vec<_>>();
        let attributes = desc
            .streams
            .iter()
            .flat_map(|x| x.attributes)
            .copied()
            .map(|x| x.into())
            .collect::<Vec<vk::VertexInputAttributeDescription>>();
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
            .depth_bias_slope_factor(0.0);

        let rasterizer_state = if let Some(desc) = desc.cull {
            rasterizer_state.cull_mode(desc.0).front_face(desc.1)
        } else {
            rasterizer_state.cull_mode(vk::CullModeFlags::NONE)
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let depthstencil_state = if let Some(op) = desc.depth_test {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(op)
                .stencil_test_enable(false)
                .depth_test_enable(true)
                .depth_write_enable(desc.depth_write)
                .build()
        } else {
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::NEVER)
                .stencil_test_enable(false)
                .depth_test_enable(false)
                .depth_write_enable(desc.depth_write)
                .build()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachment = if let Some((color, alpha)) = desc.blend {
            color_blend_attachment
                .blend_enable(true)
                .src_color_blend_factor(color.src_blend)
                .dst_color_blend_factor(color.dst_blend)
                .color_blend_op(color.op)
                .src_alpha_blend_factor(alpha.src_blend)
                .dst_alpha_blend_factor(alpha.dst_blend)
                .alpha_blend_op(alpha.op)
        } else {
            color_blend_attachment.blend_enable(false)
        }
        .build();
        let blending_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(slice::from_ref(&color_blend_attachment))
            .logic_op_enable(false)
            .build();

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&desc.pass_layout.color_attachments);
        if let Some(depth) = desc.pass_layout.depth_attachment {
            rendering_info = rendering_info.depth_attachment_format(depth)
        }

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(layout)
            .stages(&shader_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .viewport_state(&viewport_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&blending_state)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly_state_create_info)
            .rasterization_state(&rasterizer_state)
            .depth_stencil_state(&depthstencil_state)
            .push_next(&mut rendering_info)
            .build();

        let pipeline = unsafe {
            self.raw.create_graphics_pipelines(
                self.pipeline_cache,
                slice::from_ref(&pipeline_create_info),
                None,
            )
        };

        Ok((handle, pipeline?[0], layout))
    }

    pub async fn compile_raster_pipelines(&self) {
        let compiled = AsyncComputeTaskPool::get().scope(|s| {
            let programs = self.program_storage.read();
            self.pipelines
                .read()
                .enumerate()
                .for_each(|(handle, (pipeline, _), desc)| {
                    if *pipeline != vk::Pipeline::null() {
                        let program = programs
                            .get(desc.program.index())
                            .ok_or(BackendError::InvalidHandle)
                            .unwrap();
                        let shaders = program
                            .shaders
                            .iter()
                            .cloned()
                            .collect::<ArrayVec<_, MAX_SHADERS>>();
                        s.spawn(self.compile_raster_pipeline(
                            handle,
                            shaders,
                            program.pipeline_layout,
                            desc.clone(),
                        ));
                    }
                })
        });
        let mut pipelines = self.pipelines.write();
        for it in compiled {
            match it {
                Ok((handle, pipeline, layout)) => {
                    pipelines.replace(handle, (pipeline, layout));
                }
                Err(err) => error!("Failed to compiled pipeline: {:?}", err),
            }
        }
    }

    pub fn register_raster_pipeline(
        &self,
        desc: RasterPipelineCreateDesc,
    ) -> BackendResult<RasterPipelineHandle> {
        self.program_storage
            .read()
            .get(desc.program.index())
            .ok_or(BackendError::InvalidHandle)?;

        let mut pipelines = self.pipelines.write();
        let handle = pipelines.push((vk::Pipeline::null(), vk::PipelineLayout::null()), desc);

        Ok(handle)
    }
}

const MAGICK: [u8; 4] = *b"PLCH";
const VERSION: u32 = 1;
const CACHE_FILE_NAME: &str = "pipelines.bin";
const NEW_CACHE_FILE_NAME: &str = "pipelines.new.bin";

#[derive(Debug, Readable, Writable, PartialEq, Eq)]
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

#[derive(Debug)]
struct PipelineDiskCache {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: Uuid,
    data: Vec<u8>,
}

impl<'a, C: Context> Readable<'a, C> for PipelineDiskCache {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        reader
            .read_value::<Header>()
            .map(|x| x == Header::default())?;
        Ok(Self {
            vendor_id: reader.read_value()?,
            device_id: reader.read_value()?,
            driver_version: reader.read_value()?,
            uuid: reader.read_value()?,
            data: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for PipelineDiskCache {
    fn write_to<T: ?Sized + speedy::Writer<C>>(&self, writer: &mut T) -> Result<(), C::Error> {
        writer.write_value(&Header::default())?;
        writer.write_value(&self.vendor_id)?;
        writer.write_value(&self.device_id)?;
        writer.write_value(&self.driver_version)?;
        writer.write_value(&self.uuid)?;
        writer.write_value(&self.data)?;

        Ok(())
    }
}

impl PipelineDiskCache {
    pub fn new(pdevice: &PhysicalDevice, data: &[u8]) -> Self {
        let vendor_id = pdevice.properties.vendor_id;
        let device_id = pdevice.properties.device_id;
        let driver_version = pdevice.properties.driver_version;
        let uuid = Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid);

        Self {
            vendor_id,
            device_id,
            driver_version,
            uuid,
            data: data.to_vec(),
        }
    }

    pub fn load() -> io::Result<PipelineDiskCache> {
        if let Some(project_dirs) = ProjectDirs::from("com", "zlogaemz", "engine") {
            let cache_path = project_dirs.cache_dir().join(CACHE_FILE_NAME);
            info!("Loading pipeline cache from {:?}", cache_path);
            Ok(PipelineDiskCache::read_from_stream_buffered(File::open(
                cache_path,
            )?)?)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Can't get cache dir path",
            ))
        }
    }

    pub fn save(&self) -> io::Result<()> {
        if let Some(project_dirs) = ProjectDirs::from("com", "zlogaemz", "engine") {
            fs::create_dir_all(project_dirs.cache_dir())?;
            let cache_path = project_dirs.cache_dir().join(CACHE_FILE_NAME);
            let new_cache_path = project_dirs.cache_dir().join(NEW_CACHE_FILE_NAME);
            info!("Saving pipeline cache to {:?}", cache_path);
            self.write_to_stream(File::create(&new_cache_path)?)?;
            fs::rename(&new_cache_path, &cache_path)?;

            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "Can't get cache dir path",
            ))
        }
    }
}

pub fn load_or_create_pipeline_cache(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
) -> BackendResult<vk::PipelineCache> {
    let data = if let Ok(cache) = PipelineDiskCache::load() {
        if cache.vendor_id == pdevice.properties.vendor_id
            && cache.device_id == pdevice.properties.device_id
            && cache.driver_version == pdevice.properties.driver_version
            && cache.uuid == Uuid::from_bytes(pdevice.properties.pipeline_cache_uuid)
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

    let cache = match unsafe { device.create_pipeline_cache(&create_info, None) } {
        Ok(cache) => cache,
        Err(_) => {
            // Failed with initial data - so create empty cache.
            let create_info = vk::PipelineCacheCreateInfo::builder().build();
            unsafe { device.create_pipeline_cache(&create_info, None) }?
        }
    };

    Ok(cache)
}

pub fn save_pipeline_cache(
    device: &ash::Device,
    pdevice: &PhysicalDevice,
    cache: vk::PipelineCache,
) -> io::Result<()> {
    let data = unsafe { device.get_pipeline_cache_data(cache) }.map_err(|err| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to get pipeline cache data from device: {:?}", err),
        )
    })?;
    PipelineDiskCache::new(pdevice, &data).save()
}
