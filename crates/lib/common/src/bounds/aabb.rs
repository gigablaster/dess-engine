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
