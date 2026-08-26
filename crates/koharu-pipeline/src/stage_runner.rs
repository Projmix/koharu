use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use koharu_scene::Patch;

use crate::{
    ErrorKind, PipelineConfig, PipelineError, Progress, ProgressSink, Stage, StopToken,
    accelerator::AcceleratorGate,
    progress,
    resources::ResourceMonitor,
    stages::{StageInput, Stages},
};

pub(crate) struct StageRunner {
    stages: Stages,
    accelerator: AcceleratorGate,
}

impl StageRunner {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        resources: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        Ok(Self {
            stages: Stages::new(config, translator, device)?,
            accelerator: AcceleratorGate::new(device, resources),
        })
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn run(&self, job: StageJob) -> StageCompletion {
        let started = Instant::now();
        let target = job.input.target();
        let model = self.stages.model(job.stage, &job.input);
        let outcome = self.run_with_recovery(&job, &model).await;
        StageCompletion {
            target,
            stage: job.stage,
            model,
            elapsed: started.elapsed(),
            outcome,
        }
    }

    async fn run_with_recovery(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<StageOutcome, PipelineError> {
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        let skip = self.stages.skip(job.stage, &job.input).map_err(|error| {
            self.stage_error(
                job.stage,
                model,
                AttemptFailure {
                    kind: ErrorKind::Processing,
                    error,
                },
            )
        })?;
        if skip {
            return Ok(StageOutcome::Skipped);
        }
        let permit = self.accelerator.acquire().await;
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        let first = self.load_and_process(job, model).await;
        let failure = match first {
            Ok(outcome) => return Ok(outcome),
            Err(failure)
                if should_retry_after_memory_pressure(
                    self.stages
                        .recovers_from_memory_pressure(job.stage, &job.input),
                    job.stop.stopped(),
                    &failure.error,
                ) =>
            {
                failure
            }
            Err(failure) => return Err(self.stage_error(job.stage, model, failure)),
        };

        drop(permit);
        tracing::warn!(stage = %job.stage, target = %job.input.target(), error = %failure.error, "retrying stage after memory pressure");
        let _metric =
            tracing::info_span!(target: "koharu_metrics", "stage_retry", stage = %job.stage, model);
        let _permit = self.accelerator.recover(job.stage, &self.stages).await;
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        match self.load_and_process(job, model).await {
            Ok(outcome) => Ok(outcome),
            Err(failure) => Err(self.stage_error(job.stage, model, failure)),
        }
    }

    async fn load_and_process(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<StageOutcome, AttemptFailure> {
        progress::emit(
            job.progress.as_ref(),
            Progress::Loading {
                target: job.input.target(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        self.stages
            .load(job.stage, &job.input)
            .await
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::ModelLoad,
                error,
            })?;
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        progress::emit(
            job.progress.as_ref(),
            Progress::Running {
                target: job.input.target(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        let patch = self
            .stages
            .process(job.stage, job.input.clone())
            .await
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::Processing,
                error,
            })?;
        if job.stop.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        Ok(if patch.is_empty() {
            StageOutcome::Skipped
        } else {
            StageOutcome::Patch(patch)
        })
    }

    fn stage_error(&self, stage: Stage, model: &str, failure: AttemptFailure) -> PipelineError {
        self.stages.unload(stage);
        let message = match failure.kind {
            ErrorKind::ModelLoad => format!("failed to load {model}"),
            _ => format!("{model} failed"),
        };
        PipelineError::new(failure.kind, Some(stage), failure.error.context(message))
    }
}

struct AttemptFailure {
    kind: ErrorKind,
    error: anyhow::Error,
}

fn is_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        let message = source.to_string().to_ascii_lowercase();
        message.contains("out of memory")
            || message.contains("cuda_error_out_of_memory")
            || message.contains("not enough memory")
    })
}

fn should_retry_after_memory_pressure(
    locally_managed: bool,
    stopped: bool,
    error: &anyhow::Error,
) -> bool {
    locally_managed && !stopped && is_out_of_memory(error)
}

pub(crate) struct StageJob {
    stage: Stage,
    input: StageInput,
    stop: StopToken,
    progress: Option<ProgressSink>,
}

impl StageJob {
    pub(crate) fn new(
        stage: Stage,
        input: StageInput,
        stop: StopToken,
        progress: Option<ProgressSink>,
    ) -> Self {
        Self {
            stage,
            input,
            stop,
            progress,
        }
    }
}

pub(crate) enum StageOutcome {
    Patch(Patch),
    Skipped,
    Stopped,
}

pub(crate) struct StageCompletion {
    pub(crate) target: crate::StageTarget,
    pub(crate) stage: Stage,
    pub(crate) model: String,
    pub(crate) elapsed: Duration,
    pub(crate) outcome: std::result::Result<StageOutcome, PipelineError>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_pressure_does_not_retry_remote_out_of_memory_wording() {
        let error = anyhow::anyhow!("Fal.ai provider failed: out of memory");

        assert!(!should_retry_after_memory_pressure(false, false, &error));
        assert!(should_retry_after_memory_pressure(true, false, &error));
        assert!(!should_retry_after_memory_pressure(true, true, &error));
    }
}
