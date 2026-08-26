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
mod scheduler;
mod scope;
mod stage;
mod stage_runner;
mod stages;

pub use config::{
    ApiInpaintingConfig, ApiOcrConfig, DetectionModel, InpaintingConfig, InpaintingMethod,
    InpaintingModelChoice, InpaintingProvider, LocalInpaintingModel, OcrConfig, OcrMethod,
    OcrModel, PipelineConfig, ProcessorConfig, TranslationConfig, TranslationProfile,
    TranslationUnitPolicy,
};
pub use error::{ErrorKind, PipelineError};
pub use pipeline::Pipeline;
pub use progress::{Progress, ProgressSink};
pub use report::{Committer, Report, RunStatus, StageOutput};
pub use request::{InpaintingMask, Operation, Request, StopToken};
pub use resources::{DeviceResources, ResourceSnapshot};
pub use scope::{Bounds, Scope};
pub use stage::{Stage, StageTarget};
pub use stages::{
    ChapterTranslation, Flux2KleinConfig, ImageEditConfig, InpaintingApplyMode,
    KoharuLayoutRFDetrSeg2XLConfig, RoremMixedConfig, text_layer_placement,
};

use images::ImageCache;
use model_cell::ModelCell;

pub async fn inpainting_models() -> anyhow::Result<Vec<InpaintingModelChoice>> {
    stages::inpainting_models().await
}

#[cfg(test)]
mod tests;
