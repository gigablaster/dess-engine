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

pub mod barrier;
mod draw_stream;
mod error;
mod vulkan;

pub use draw_stream::*;
pub use error::*;
use vulkan::ExecutionContext;
pub use vulkan::*;

pub type BackendResult<T> = Result<T, BackendError>;

pub(crate) trait DeferedPass: Send + Sync {
    fn execute(&self, contex: &ExecutionContext) -> BackendResult<()>;
}

pub trait BackendResultExt {
    fn ignore_invalid_handle(self) -> BackendResult<()>;
    fn ignore_missing(self) -> BackendResult<()>;
}

impl BackendResultExt for BackendResult<()> {
    fn ignore_invalid_handle(self) -> BackendResult<()> {
        match self {
            Ok(_) => Ok(()),
            Err(err) => match err {
                BackendError::InvalidHandle => Ok(()),
                other => Err(other),
            },
        }
    }

    fn ignore_missing(self) -> BackendResult<()> {
        match self {
            Ok(_) => Ok(()),
            Err(err) => match err {
                BackendError::NotFound => Ok(()),
                other => Err(other),
            },
        }
    }
}
