use super::{MeshLayout, VertexAttribute};

pub const BASE_COLOR_TEXTURE: &str = "base";
pub const NORMAL_MAP_TEXTURE: &str = "nomrmal";
pub const METALLIC_ROUGHNESS_TEXTURE: &str = "metallic_roughness";
pub const OCCLUSION_TEXTURE: &str = "occlusion";

pub const VERTEX_POSITION_CHANNEL: &str = "position";
pub const VERTEX_NORNAL_CHANNEL: &str = "normal";
pub const VERTEX_TANGENT_CHANNEL: &str = "tangent";
pub const VERTEX_BINORMAL_CHANNEL: &str = "binormal";
pub const VERTEX_UV_CHANNEL: &str = "uv";

#[derive(Debug)]
pub(crate) struct VertexChannel {
    pub count: usize,
    pub values: Vec<f32>,
}

pub trait VertexData {
    const COUNT: usize;
    fn push(&self, out: &mut Vec<f32>);
    fn from_data(values: &[f32]) -> Self;
}

impl VertexData for glam::Vec2 {
    const COUNT: usize = 2;
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.x);
        out.push(self.y);
    }

    fn from_data(values: &[f32]) -> Self {
        Self {
            x: values[0],
            y: values[1],
        }
    }
}

impl VertexData for glam::Vec3 {
    const COUNT: usize = 3;
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.x);
        out.push(self.y);
        out.push(self.z);
    }

    fn from_data(values: &[f32]) -> Self {
        Self {
            x: values[0],
            y: values[1],
            z: values[2],
        }
    }
}

impl VertexData for glam::Vec4 {
    const COUNT: usize = 4;
    fn push(&self, out: &mut Vec<f32>) {
        out.push(self.x);
        out.push(self.y);
        out.push(self.z);
        out.push(self.w);
    }

    fn from_data(values: &[f32]) -> Self {
        Self::new(values[0], values[1], values[2], values[3])
    }
}

impl VertexChannel {
    pub fn new(count: usize) -> Self {
        Self {
            count,
            values: Vec::new(),
        }
    }

    pub fn push<T: VertexData>(&mut self, value: T) {
        assert_eq!(self.count, T::COUNT);
        value.push(&mut self.values);
    }

    pub fn data(&self) -> &[f32] {
        &self.values
    }

    pub fn to_vertices<T: VertexData>(&self) -> Vec<T> {
        assert_eq!(self.count, T::COUNT);
        let count = self.values.len() / T::COUNT;
        let mut result = Vec::with_capacity(count);
        for index in 0..count {
            let vertex = T::from_data(&self.values[index * T::COUNT..index * T::COUNT + T::COUNT]);
            result.push(vertex);
        }

        result
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

#[derive(Debug, Default)]
pub struct MeshLayoutBuilder<'a> {
    channels: Vec<(&'a str, VertexAttribute)>,
}

impl<'a> MeshLayoutBuilder<'a> {
    pub fn channel(mut self, name: &'a str, attribute: VertexAttribute) -> Self {
        assert!(self.channels.iter().all(|(n, _)| &name != n));
        self.channels.push((name, attribute));
        self
    }
}

#[derive(Debug)]
pub struct MeshBuilder {
    names: Vec<String>,
    channels: Vec<VertexChannel>,
    layout: Vec<VertexAttribute>,
    indices: Vec<usize>,
    current_vertex: Option<usize>,
    current_channel: usize,
}

impl MeshBuilder {
    pub fn new(builder: MeshLayoutBuilder) -> Self {
        let mut layout = Vec::new();
        let mut channels = Vec::new();
        let mut names = Vec::new();
        let indices = Vec::new();
        builder.channels.into_iter().for_each(|(name, attr)| {
            layout.push(attr);
            names.push(name.into());
            channels.push(VertexChannel::new(attr.count()));
        });

        Self {
            names,
            channels,
            layout,
            indices,
            current_vertex: None,
            current_channel: 0,
        }
    }

    pub fn vertex(&mut self) -> usize {
        let current = self.current_vertex.unwrap_or(0);
        self.current_channel = 0;

        assert!(
            self.channels
                .iter()
                .all(|x| x.values.len() / x.count == current),
            "All vertex channels must be filled before moving to next vertex"
        );
        self.current_vertex = Some(current + 1);
        current
    }

    pub fn push<T: VertexData>(&mut self, value: T) {
        assert_ne!(None, self.current_vertex, "First vertex should be started");
        assert!(
            self.current_channel < self.layout.len(),
            "There are only that many channels in single vertex."
        );
        self.channels[self.current_channel].push(value);
        self.current_channel += 1;
    }

    pub fn triangle(&mut self, indices: [usize; 3]) {
        assert!(
            indices
                .iter()
                .all(|x| { self.channels.iter().all(|y| *x < y.values.len()) }),
            "Index out of bounds"
        );
        self.indices.push(indices[0]);
        self.indices.push(indices[1]);
        self.indices.push(indices[2]);
    }

    pub fn clear(&mut self) {
        self.channels.iter_mut().for_each(|x| x.clear());
    }

    pub fn vertices<T: VertexData>(&self, name: &str) -> Option<Vec<T>> {
        self.names
            .iter()
            .enumerate()
            .find_map(|(index, n)| if name == n { Some(index) } else { None })
            .map(|index| self.channels[index].to_vertices())
    }

    pub fn values(&self, name: &str) -> Option<&[f32]> {
        self.names
            .iter()
            .enumerate()
            .find_map(|(index, n)| if name == n { Some(index) } else { None })
            .map(|index| self.channels[index].data())
    }

    pub fn attribute(&self, name: &str) -> Option<VertexAttribute> {
        self.names
            .iter()
            .enumerate()
            .find_map(|(index, n)| if name == n { Some(index) } else { None })
            .map(|index| self.layout[index])
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn layout(&self) -> &[VertexAttribute] {
        &self.layout
    }

    pub(crate) fn current_vertex(&self) -> Option<usize> {
        self.current_vertex
    }

    pub(crate) fn channels(&self) -> &[VertexChannel] {
        &self.channels
    }

    pub(crate) fn create_mesh_layout(&self) -> MeshLayout {
        let mut layout = MeshLayout::with_capacity(self.layout.len());
        let mut offset = 0;
        for index in 0..self.layout.len() {
            let name = &self.names[index];
            let attribute = self.layout[index];
            layout.push((name.clone(), attribute, offset));
            offset += attribute.count() as u32;
        }

        layout
    }
}

#[cfg(test)]
mod test {
    use super::{MeshBuilder, MeshLayoutBuilder, VERTEX_POSITION_CHANNEL, VERTEX_UV_CHANNEL};
    use glam::{vec2, vec3, Vec2, Vec3};

    #[test]
    fn push_vertex() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(vec3(1.0, 1.0, 1.0));
    }

    #[test]
    #[should_panic]
    fn panic_push_unfilled_vertex() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);

        assert_eq!(0, builder.vertex());
        builder.vertex();
    }

    #[test]
    #[should_panic]
    fn panic_push_too_many_attributes() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        builder.push(Vec3::ZERO);
        builder.push(Vec3::ZERO);
    }

    #[test]
    #[should_panic]
    fn panic_push_no_first_vertex() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        builder.push(Vec3::ZERO);
    }

    #[test]
    fn push_many_vertices() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(vec3(1.0, 1.0, 1.0));
        assert_eq!(1, builder.vertex());
        builder.push(vec3(2.0, 2.0, 2.0));
    }

    #[test]
    fn multiple_attributes() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3)
            .channel(VERTEX_UV_CHANNEL, super::VertexAttribute::Vec2);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(Vec3::ZERO);
        builder.push(Vec2::ZERO);
        assert_eq!(1, builder.vertex());
    }

    #[test]
    fn get_vertices() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3)
            .channel(VERTEX_UV_CHANNEL, super::VertexAttribute::Vec2);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(vec3(1.0, 1.0, 1.0));
        builder.push(vec2(2.0, 2.0));
        assert_eq!(1, builder.vertex());
        builder.push(vec3(3.0, 3.0, 3.0));
        builder.push(vec2(4.0, 4.0));
        let positions = builder
            .vertices::<glam::Vec3>(VERTEX_POSITION_CHANNEL)
            .unwrap();
        assert_eq!(2, positions.len());
        assert_eq!(vec3(1.0, 1.0, 1.0), positions[0]);
        assert_eq!(vec3(3.0, 3.0, 3.0), positions[1]);
        let uvs = builder.vertices::<glam::Vec2>(VERTEX_UV_CHANNEL).unwrap();
        assert_eq!(2, uvs.len());
        assert_eq!(vec2(2.0, 2.0), uvs[0]);
        assert_eq!(vec2(4.0, 4.0), uvs[1]);
    }

    #[test]
    fn none_for_not_existing_channels() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let builder = MeshBuilder::new(layout);
        assert_eq!(None, builder.vertices::<glam::Vec2>(VERTEX_UV_CHANNEL));
    }

    #[test]
    #[should_panic]
    fn panic_if_wrong_vertex_type() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(vec3(1.0, 1.0, 1.0));
        builder.vertices::<glam::Vec2>(VERTEX_POSITION_CHANNEL);
    }

    #[test]
    #[should_panic]
    fn panic_if_push_wrong_type() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec3);
        let mut builder = MeshBuilder::new(layout);
        assert_eq!(0, builder.vertex());
        builder.push(vec2(1.0, 1.0));
    }

    #[test]
    fn add_triangle() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec2);
        let mut builder = MeshBuilder::new(layout);
        let v1 = builder.vertex();
        builder.push(vec2(1.0, 1.0));
        let v2 = builder.vertex();
        builder.push(vec2(2.0, 2.0));
        let v3 = builder.vertex();
        builder.push(vec2(1.0, 2.0));
        builder.triangle([v1, v2, v3]);
        let indices = builder.indices();
        assert_eq!(3, indices.len());
        assert_eq!(v1, indices[0]);
        assert_eq!(v2, indices[1]);
        assert_eq!(v3, indices[2]);
    }

    #[test]
    #[should_panic]
    fn panic_add_triangle_with_wrong_index() {
        let layout = MeshLayoutBuilder::default()
            .channel(VERTEX_POSITION_CHANNEL, super::VertexAttribute::Vec2);
        let mut builder = MeshBuilder::new(layout);
        let v1 = builder.vertex();
        builder.push(vec2(1.0, 1.0));
        let v2 = builder.vertex();
        builder.push(vec2(2.0, 2.0));
        builder.triangle([v1, v2, 99]);
    }
}
