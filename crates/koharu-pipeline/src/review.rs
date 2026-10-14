use std::sync::Arc;

use async_trait::async_trait;
use image::DynamicImage;
use koharu_scene::EntityId;

use crate::{InpaintingProvider, StopToken};

/// The exact prepared image and final prompt at the remote image-edit boundary.
#[derive(Clone, Debug)]
pub struct ImageEditReview {
    pub page: EntityId,
    pub page_label: String,
    pub provider: InpaintingProvider,
    pub model: String,
    pub prompt: String,
    pub image: Arc<DynamicImage>,
    pub restoration: bool,
}

#[async_trait]
pub trait ImageEditReviewer: Send + Sync {
    /// Returns an approved prompt verbatim, or `None` to stop this pipeline run.
    async fn review(
        &self,
        request: ImageEditReview,
        stop: &StopToken,
    ) -> anyhow::Result<Option<String>>;
}
