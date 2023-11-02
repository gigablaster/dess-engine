use std::{io, path::Path};

use crate::AssetProcessingContext;

/// Builds local asset bundle
///
/// Doesn't do processing work, just collect already processed files and put them
/// in bundle in sorted way.
pub struct BundleBuilder {
    context: AssetProcessingContext,
}

impl BundleBuilder {
    pub fn new(context: AssetProcessingContext) -> Self {
        Self { context }
    }

    pub fn generate(&self, path: &Path) -> io::Result<()> {
        todo!()
    }
}
