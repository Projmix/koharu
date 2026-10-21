use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{ImageEditReviewer, ProgressSink, Scope, Stage};
use koharu_scene::EntityId;

#[derive(Clone, Debug)]
pub struct InpaintingMask {
    pub page: EntityId,
    pub png: Arc<[u8]>,
    pub intent: InpaintingIntent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InpaintingIntent {
    Remove,
    Restore,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    #[default]
    Full,
    Through {
        stage: Stage,
    },
    Only {
        stage: Stage,
    },
    Stages {
        stages: Vec<Stage>,
    },
    /// Rewrite the generated translations of the selected texts shorter.
    Shorten,
}

impl Operation {
    pub(crate) fn stages(&self) -> Result<Vec<Stage>> {
        let stages = match self {
            // Scan enhancement changes the page's pixels, so it runs only when
            // the user selects it.
            Self::Full => Stage::ALL
                .into_iter()
                .filter(|stage| *stage != Stage::Enhancement)
                .collect(),
            Self::Through {
                stage: Stage::Enhancement,
            } => vec![Stage::Enhancement],
            Self::Through {
                stage: Stage::Detection,
            } => vec![Stage::Detection],
            Self::Through { stage: Stage::Ocr } => vec![Stage::Detection, Stage::Ocr],
            Self::Through {
                stage: Stage::Markup,
            } => vec![Stage::Detection, Stage::Ocr, Stage::Markup],
            Self::Through {
                stage: Stage::Translation,
            } => vec![
                Stage::Detection,
                Stage::Ocr,
                Stage::Markup,
                Stage::Translation,
            ],
            Self::Through {
                stage: Stage::Inpainting,
            } => vec![Stage::Detection, Stage::Inpainting],
            Self::Only { stage } => vec![*stage],
            Self::Shorten => vec![Stage::Translation],
            Self::Stages { stages } => Stage::ALL
                .into_iter()
                .filter(|stage| stages.contains(stage))
                .collect(),
        };
        if stages.is_empty() {
            bail!("at least one pipeline stage must be selected");
        }
        Ok(stages)
    }
}

#[derive(Clone)]
pub struct Request {
    pub operation: Operation,
    pub scope: Scope,
    pub stop: StopToken,
    pub progress: Option<ProgressSink>,
    pub inpainting_mask: Option<InpaintingMask>,
    pub image_edit_reviewer: Option<Arc<dyn ImageEditReviewer>>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            operation: Operation::Full,
            scope: Scope::Project,
            stop: StopToken::default(),
            progress: None,
            inpainting_mask: None,
            image_edit_reviewer: None,
        }
    }
}

struct StopState {
    stopped: AtomicBool,
    notify: tokio::sync::Notify,
}

#[derive(Clone)]
pub struct StopToken(Arc<StopState>);

impl Default for StopToken {
    fn default() -> Self {
        Self(Arc::new(StopState {
            stopped: AtomicBool::new(false),
            notify: tokio::sync::Notify::new(),
        }))
    }
}

impl StopToken {
    pub fn stop(&self) {
        if !self.0.stopped.swap(true, Ordering::AcqRel) {
            self.0.notify.notify_waiters();
        }
    }

    #[must_use]
    pub fn stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        if self.stopped() {
            return;
        }
        let notified = self.0.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.stopped() {
            return;
        }
        notified.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stop_wakes_all_async_waiters() {
        let stop = StopToken::default();
        let first = tokio::spawn({
            let stop = stop.clone();
            async move { stop.cancelled().await }
        });
        let second = tokio::spawn({
            let stop = stop.clone();
            async move { stop.cancelled().await }
        });

        tokio::task::yield_now().await;
        stop.stop();
        first.await.unwrap();
        second.await.unwrap();
        stop.cancelled().await;
    }
}
