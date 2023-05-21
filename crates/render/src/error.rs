use dess_render_backend::BackendError;

pub enum RenderError {
    Backend(BackendError),
}

impl From<BackendError> for RenderError {
    fn from(value: BackendError) -> Self {
        RenderError::Backend(value)
    }
}

pub type RenderResult<T> = Result<T, RenderError>;
