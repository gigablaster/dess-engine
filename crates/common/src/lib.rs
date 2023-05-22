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

mod client;
pub mod memory;
mod serialization;
mod time;

pub mod traits;

use std::marker::PhantomData;

pub use client::*;
pub use serialization::*;
pub use time::*;

pub trait Align<T> {
    fn align(self, align: T) -> Self;
}

impl Align<u32> for u32 {
    fn align(self, align: u32) -> Self {
        if self == 0 || self % align == 0 {
            self
        } else {
            (self & !(align - 1)) + align
        }
    }
}

impl Align<u64> for u64 {
    fn align(self, align: u64) -> Self {
        if self == 0 || self % align == 0 {
            self
        } else {
            (self & !(align - 1)) + align
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Handle<T> {
    pub index: u16,
    pub generation: u16,
    _phantom: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(index: u16, generation: u16) -> Self {
        Self {
            index,
            generation,
            _phantom: PhantomData::<T>,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::Align;

    #[test]
    fn align() {
        assert_eq!(0, 0u32.align(64));
        assert_eq!(64, 50u32.align(64));
        assert_eq!(64, 64u32.align(64));
        assert_eq!(128, 100u32.align(64));
        assert_eq!(128, 128u32.align(64));
    }
}
