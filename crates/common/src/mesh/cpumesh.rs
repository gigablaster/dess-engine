use crate::traits::{BinaryDeserialization, BinarySerialization};

use super::{EffectInfo, MeshBuilder, MeshLayout};

#[derive(Debug)]
pub struct CpuMesh {
    pub layout: MeshLayout,
    pub vertex_data: Vec<f32>,
    pub index_data: Vec<u16>,
    pub effect: EffectInfo,
}

impl BinarySerialization for CpuMesh {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.layout.serialize(w)?;
        self.effect.serialize(w)?;
        self.vertex_data.serialize(w)?;
        self.index_data.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for CpuMesh {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let layout = MeshLayout::deserialize(r)?;
        let effect = EffectInfo::deserialize(r)?;
        let vertex_data = Vec::deserialize(r)?;
        let index_data = Vec::deserialize(r)?;

        Ok(Self {
            layout,
            vertex_data,
            index_data,
            effect,
        })
    }
}

impl CpuMesh {
    pub fn build(builder: MeshBuilder, effect: EffectInfo) -> Self {
        let vertex_count = builder
            .current_vertex()
            .expect("Build must contain at least one vertex");
        let stride: usize = builder.layout().iter().map(|x| x.count()).sum();
        let mut vertices = Vec::with_capacity(vertex_count * stride);
        for index in 0..vertex_count {
            for (channel_index, channel) in builder.channels().iter().enumerate() {
                let count = builder.layout()[channel_index].count();
                let data = channel.data();
                for attr_index in 0..count {
                    vertices.push(data[index * count + attr_index]);
                }
            }
        }
        let indices = builder.indices().iter().map(|x| *x as u16).collect();
        Self {
            layout: builder.create_mesh_layout(),
            effect,
            vertex_data: vertices,
            index_data: indices,
        }
    }
}

#[cfg(test)]
mod test {
    use glam::{vec2, vec3};

    use crate::mesh::{
        builder::{VERTEX_POSITION_CHANNEL, VERTEX_UV_CHANNEL},
        EffectInfo, MeshBuilder, MeshLayoutBuilder, VertexAttribute,
    };

    use super::CpuMesh;

    #[test]
    fn interleave_vertex() {
        let mut builder = MeshBuilder::new(
            MeshLayoutBuilder::default()
                .channel(VERTEX_POSITION_CHANNEL, VertexAttribute::Vec3)
                .channel(VERTEX_UV_CHANNEL, VertexAttribute::Vec2),
        );
        let v1 = builder.vertex();
        builder.push(vec3(0.0, 1.0, 2.0));
        builder.push(vec2(3.0, 4.0));
        let v2 = builder.vertex();
        builder.push(vec3(5.0, 6.0, 7.0));
        builder.push(vec2(8.0, 9.0));
        let v3 = builder.vertex();
        builder.push(vec3(10.0, 11.0, 12.0));
        builder.push(vec2(13.0, 14.0));
        builder.triangle([v1, v2, v3]);
        let mesh = CpuMesh::build(builder, EffectInfo::new("test"));
        mesh.vertex_data
            .iter()
            .enumerate()
            .for_each(|(index, value)| assert_eq!(index as f32, *value));
    }
}
