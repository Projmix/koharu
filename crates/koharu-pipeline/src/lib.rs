//! In-process, scene-native model orchestration for Koharu.

mod accelerator;
mod config;
mod error;
mod execution;
mod images;
mod model_cell;
mod pipeline;
mod progress;
mod report;
mod request;
mod resources;
mod review;
mod scheduler;
mod scope;
mod stage;
mod stage_runner;
mod stages;

pub use config::{
    ApiInpaintingConfig, ApiOcrConfig, DetectionModel, EnhancementConfig, InpaintingConfig,
    InpaintingMethod, InpaintingModelChoice, InpaintingProvider, LocalInpaintingModel, OcrConfig,
    OcrMethod, OcrModel, PipelineConfig, ProcessorConfig, RestorationConfig, SceneMarkupConfig,
    TranslationConfig, TranslationProfile, TranslationUnitPolicy,
};
pub use error::{ErrorKind, PipelineError};
pub use pipeline::Pipeline;
pub use progress::{Progress, ProgressSink};
pub use report::{Committer, Report, RunStatus, StageOutput};
pub use request::{InpaintingIntent, InpaintingMask, Operation, Request, StopToken};
pub use resources::{DeviceResources, ResourceSnapshot};
pub use review::{ImageEditReview, ImageEditReviewer};
pub use scope::{Bounds, Scope};
pub use stage::{Stage, StageProgress, StageTarget};
pub use stages::{
    ChapterTranslation, Flux2KleinConfig, ImageEditConfig, ImportScope, InpaintingApplyMode,
    KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig, Typeset, cleaning_unverified, ocr_unverified,
    open_unboxed_text, text_layer_placement,
};

use images::ImageCache;
use model_cell::ModelCell;

pub async fn inpainting_models() -> anyhow::Result<Vec<InpaintingModelChoice>> {
    stages::inpainting_models().await
}

#[cfg(test)]
mod tests;
