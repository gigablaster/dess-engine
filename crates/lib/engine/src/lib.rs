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

mod asset_cache;
mod buffer_pool;
mod effect;
mod material;
mod mesh;
mod pool;

pub use asset_cache::*;
pub use buffer_pool::*;
use dess_backend::BackendError;
pub use effect::*;
pub use material::*;
pub use mesh::*;
pub use pool::*;

#[derive(Debug, Clone)]
pub enum Error {
    BackendError(BackendError),
    InvalidHandle,
    ImportFailed(dess_assets::Error),
    LoadingFailed,
}

impl From<BackendError> for Error {
    fn from(value: BackendError) -> Self {
        Self::BackendError(value)
    }
}

impl From<dess_assets::Error> for Error {
    fn from(value: dess_assets::Error) -> Self {
        Self::ImportFailed(value)
    }
}
