use crate::vulkan::{CreateError, MapError, ResetError, ResourceCreateError, WaitError};

#[derive(Debug)]
pub enum StagingError {
    ResourceCreate(ResourceCreateError),
    Create(CreateError),
    Wait(WaitError),
    Reset(ResetError),
    Map(MapError),
}

impl From<ResourceCreateError> for StagingError {
    fn from(value: ResourceCreateError) -> Self {
        Self::ResourceCreate(value)
    }
}

impl From<CreateError> for StagingError {
    fn from(value: CreateError) -> Self {
        Self::Create(value)
    }
}

impl From<WaitError> for StagingError {
    fn from(value: WaitError) -> Self {
        Self::Wait(value)
    }
}

impl From<ResetError> for StagingError {
    fn from(value: ResetError) -> Self {
        Self::Reset(value)
    }
}

impl From<MapError> for StagingError {
    fn from(value: MapError) -> Self {
        Self::Map(value)
    }
}
