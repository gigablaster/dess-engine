use crate::vulkan::{CreateError, MapError, ResetError, ResourceCreateError, WaitError};

#[derive(Debug)]
pub enum StagingError {
    ResourceCreateError(ResourceCreateError),
    CreateError(CreateError),
    WaitError(WaitError),
    ResetError(ResetError),
    MapError(MapError),
}

impl From<ResourceCreateError> for StagingError {
    fn from(value: ResourceCreateError) -> Self {
        Self::ResourceCreateError(value)
    }
}

impl From<CreateError> for StagingError {
    fn from(value: CreateError) -> Self {
        Self::CreateError(value)
    }
}

impl From<WaitError> for StagingError {
    fn from(value: WaitError) -> Self {
        Self::WaitError(value)
    }
}

impl From<ResetError> for StagingError {
    fn from(value: ResetError) -> Self {
        Self::ResetError(value)
    }
}

impl From<MapError> for StagingError {
    fn from(value: MapError) -> Self {
        Self::MapError(value)
    }
}
