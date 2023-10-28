use crate::traits::{BinaryDeserialization, BinarySerialization};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub center: glam::Vec3,
    pub extent: glam::Vec3,
}

impl BinarySerialization for AABB {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.center.serialize(w)?;
        self.extent.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for AABB {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let center = glam::Vec3::deserialize(r)?;
        let extent = glam::Vec3::deserialize(r)?;

        Ok(Self { center, extent })
    }
}

impl AABB {
    pub fn from_min_max(min: glam::Vec3, max: glam::Vec3) -> Self {
        let extent = max - min;
        let center = min + extent / 2.0;

        Self { center, extent }
    }
}
