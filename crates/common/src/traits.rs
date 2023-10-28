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

use std::io::{self, Read, Write};

pub trait BinaryDeserialization: Sized {
    fn deserialize(r: &mut impl Read) -> io::Result<Self>;
}

pub trait BinarySerialization {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()>;
}
