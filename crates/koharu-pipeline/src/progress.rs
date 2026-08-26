use std::sync::Arc;

use crate::{Stage, StageTarget};
use koharu_scene::EntityId;

#[derive(Clone, Debug)]
pub enum Progress {
    Started {
        pages: Vec<EntityId>,
        stages: Vec<Stage>,
        total: usize,
    },
    Loading {
        target: StageTarget,
        stage: Stage,
        model: String,
    },
    Running {
        target: StageTarget,
        stage: Stage,
        model: String,
    },
    Finished {
        target: StageTarget,
        stage: Stage,
        model: String,
        elapsed: std::time::Duration,
    },
    Skipped {
        target: StageTarget,
        stage: Stage,
    },
    Warning {
        target: StageTarget,
        stage: Stage,
        model: String,
        error: String,
    },
}

pub type ProgressSink = Arc<dyn Fn(Progress) + Send + Sync>;

pub(crate) fn emit(sink: Option<&ProgressSink>, progress: Progress) {
    if let Some(sink) = sink
        && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sink(progress))).is_err()
    {
        tracing::warn!("pipeline progress callback panicked");
    }
}
