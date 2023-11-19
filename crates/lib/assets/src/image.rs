// Copyright (C) 2023 Vladimir Kuskov

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

use ash::vk;
use speedy::{Context, Readable, Writable};

use crate::{AddressableAsset, Asset};

#[derive(Debug)]
pub struct ImageAsset {
    pub format: vk::Format,
    pub dimensions: [u32; 2],
    pub mips: Vec<Vec<u8>>,
}

impl<'a, C: Context> Readable<'a, C> for ImageAsset {
    fn read_from<R: speedy::Reader<'a, C>>(reader: &mut R) -> Result<Self, <C as Context>::Error> {
        Ok(Self {
            format: vk::Format::from_raw(reader.read_i32()?),
            dimensions: reader.read_value()?,
            mips: reader.read_value()?,
        })
    }
}

impl<C: Context> Writable<C> for ImageAsset {
    fn write_to<T: ?Sized + speedy::Writer<C>>(
        &self,
        writer: &mut T,
    ) -> Result<(), <C as Context>::Error> {
        writer.write_i32(self.format.as_raw())?;
        writer.write_value(&self.dimensions)?;
        writer.write_value(&self.mips)?;

        Ok(())
    }
}

impl AddressableAsset for ImageAsset {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("c2871b90-6b51-427f-b1d8-4cedbedc8993");
}

impl Asset for ImageAsset {
    fn serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        Ok(self.write_to_stream(w)?)
    }

    fn deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self::read_from_stream_unbuffered(r)?)
    }

    fn collect_depenencies(&self, _dependencies: &mut std::collections::HashSet<crate::AssetRef>) {}
}
