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

use crate::traits::{BinaryDeserialization, BinarySerialization};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Transform {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

impl BinarySerialization for Transform {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.translation.serialize(w)?;
        self.rotation.serialize(w)?;
        self.scale.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for Transform {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let translation = glam::Vec3::deserialize(r)?;
        let rotation = glam::Quat::deserialize(r)?;
        let scale = glam::Vec3::deserialize(r)?;

        Ok(Self {
            translation,
            rotation,
            scale,
        })
    }
}

impl From<Transform> for glam::Affine3A {
    fn from(value: Transform) -> Self {
        glam::Affine3A::from_scale_rotation_translation(
            value.scale,
            value.rotation,
            value.translation,
        )
    }
}

impl From<Transform> for glam::Mat4 {
    fn from(value: Transform) -> Self {
        glam::Mat4::from_scale_rotation_translation(value.scale, value.rotation, value.translation)
    }
}
