use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use futures::{StreamExt as _, stream::FuturesUnordered};
use koharu_scene::{EntityId, Snapshot};

use crate::{
    Committer, ErrorKind, PipelineError, Progress, ProgressSink, Report, Request, RunStatus, Stage,
    StageOutput, StageTarget, StopToken,
    images::ImageCache,
    progress,
    resources::ResourceMonitor,
    scheduler::Scheduler,
    scope::NormalizedScope,
    stage_runner::{StageCompletion, StageJob, StageOutcome, StageRunner},
    stages::StageInput,
};

pub(crate) struct Execution<'a> {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
    committer: &'a mut dyn Committer,
    stop: StopToken,
    progress: Option<ProgressSink>,
    scope: NormalizedScope,
    scheduler: Scheduler,
    scene: Snapshot,
    images: BTreeMap<EntityId, Arc<ImageCache>>,
    busy_stages: BTreeSet<Stage>,
    completed: usize,
    failure: Option<PipelineError>,
    base: koharu_scene::Revision,
    started: Instant,
    inpainting_mask: Option<crate::InpaintingMask>,
}

impl<'a> Execution<'a> {
    pub(crate) fn new(
        runner: Arc<StageRunner>,
        resources: Arc<ResourceMonitor>,
        snapshot: Snapshot,
        request: Request,
        committer: &'a mut dyn Committer,
    ) -> std::result::Result<Self, PipelineError> {
        let started = Instant::now();
        let base = snapshot.revision();
        let stages = request
            .operation
            .stages()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let scope = NormalizedScope::new(&snapshot, &request.scope, &stages)
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let pages = scope.pages().to_vec();
        let scheduler = Scheduler::new(
            &pages,
            &stages,
            matches!(request.scope, crate::Scope::Project),
        );
        if let Some(mask) = request.inpainting_mask.as_ref()
            && (!pages.contains(&mask.page) || !stages.contains(&Stage::Inpainting))
        {
            return Err(PipelineError::new(
                ErrorKind::InvalidInput,
                Some(Stage::Inpainting),
                anyhow::anyhow!("the inpainting mask page is outside the inpainting scope"),
            ));
        }
        progress::emit(
            request.progress.as_ref(),
            Progress::Started {
                pages: pages.clone(),
                stages: stages.clone(),
                total: scheduler.total(),
            },
        );

        Ok(Self {
            runner,
            resources,
            committer,
            stop: request.stop,
            progress: request.progress,
            scope,
            scheduler,
            scene: snapshot,
            images: BTreeMap::new(),
            busy_stages: BTreeSet::new(),
            completed: 0,
            failure: None,
            base,
            started,
            inpainting_mask: request.inpainting_mask,
        })
    }

    pub(crate) async fn run(mut self) -> std::result::Result<Report, PipelineError> {
        if self.stopped() {
            return Ok(self.report(RunStatus::Stopped));
        }

        self.resources.start();
        self.resources.wait_for_sample().await;

        let runner = self.runner.clone();
        let mut running = FuturesUnordered::new();
        loop {
            while let Some(job) = self.take_ready_job() {
                running.push(runner.run(job));
            }

            let Some(completion) = running.next().await else {
                break;
            };
            self.handle_completion(completion).await;
        }

        self.finalize()
    }

    async fn handle_completion(&mut self, completion: StageCompletion) {
        self.busy_stages.remove(&completion.stage);
        if self.stopped() {
            return;
        }
        if let Err(error) = self.apply_completion(completion).await {
            self.failure.get_or_insert(error);
        }
    }

    fn take_ready_job(&mut self) -> Option<StageJob> {
        if self.stopped() || self.failure.is_some() {
            return None;
        }
        let (target, stage) = self.scheduler.start_next(&self.busy_stages)?;
        self.busy_stages.insert(stage);
        let input = match target {
            StageTarget::Page(page) => {
                let images = self
                    .images
                    .entry(page)
                    .or_insert_with(|| Arc::new(ImageCache::default()))
                    .clone();
                StageInput::new(
                    self.scene.clone(),
                    page,
                    self.scope.entities(),
                    self.scope.region(page),
                    images,
                    self.inpainting_mask
                        .as_ref()
                        .filter(|mask| stage == Stage::Inpainting && mask.page == page)
                        .cloned(),
                )
            }
            StageTarget::Chapter => {
                StageInput::chapter(self.scene.clone(), Arc::from(self.scope.pages()))
            }
        }
        .with_stop(self.stop.clone());
        Some(StageJob::new(
            stage,
            input,
            self.stop.clone(),
            self.progress.clone(),
        ))
    }

    async fn apply_completion(
        &mut self,
        completion: StageCompletion,
    ) -> std::result::Result<(), PipelineError> {
        let StageCompletion {
            target,
            stage,
            model,
            elapsed,
            outcome,
        } = completion;
        let outcome = match outcome {
            Err(error)
                if stage == Stage::Inpainting
                    && target.page().is_some()
                    && self.scheduler.project_scope() =>
            {
                self.mark_complete(target, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::Warning {
                        target,
                        stage,
                        model,
                        error: format!("{error:#}"),
                    },
                );
                return Ok(());
            }
            Err(error) => return Err(error),
            Ok(outcome) => outcome,
        };
        match outcome {
            StageOutcome::Stopped => {}
            StageOutcome::Skipped => {
                self.mark_complete(target, stage);
                progress::emit(self.progress.as_ref(), Progress::Skipped { target, stage });
            }
            StageOutcome::Patch(patch) => {
                if !self.commit_patch(target, stage, patch).await? {
                    return Ok(());
                }
                self.mark_complete(target, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::Finished {
                        target,
                        stage,
                        model,
                        elapsed,
                    },
                );
            }
        }
        Ok(())
    }

    async fn commit_patch(
        &mut self,
        target: StageTarget,
        stage: Stage,
        patch: koharu_scene::Patch,
    ) -> std::result::Result<bool, PipelineError> {
        let patch = patch
            .rebase_on(&self.scene)
            .and_then(|patch| {
                patch.validate_on(&self.scene)?;
                Ok(patch.with_label(format!("Pipeline {stage} for {target}")))
            })
            .context("failed to rebase stage output onto the latest scene")
            .map_err(|error| PipelineError::new(ErrorKind::InvalidOutput, Some(stage), error))?;
        if self.stopped() {
            return Ok(false);
        }

        let next = self
            .committer
            .commit(StageOutput {
                target,
                stage,
                patch,
            })
            .await
            .with_context(|| format!("failed to commit {stage} output for {target}"))
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        validate_commit(&self.scene, &next)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        self.scene = next;
        Ok(true)
    }

    fn mark_complete(&mut self, target: StageTarget, stage: Stage) {
        if self.scheduler.complete_stage(target, stage)
            && let Some(page) = target.page()
        {
            self.images.remove(&page);
        }
        self.completed += 1;
    }

    fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn finalize(mut self) -> std::result::Result<Report, PipelineError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if !self.stopped() && self.completed != self.scheduler.total() {
            return Err(PipelineError::new(
                ErrorKind::InvalidOutput,
                None,
                anyhow::anyhow!(
                    "pipeline scheduler stopped after {} of {} work items",
                    self.completed,
                    self.scheduler.total()
                ),
            ));
        }
        let status = if self.stopped() {
            RunStatus::Stopped
        } else {
            RunStatus::Completed
        };
        Ok(self.report(status))
    }

    fn report(&self, status: RunStatus) -> Report {
        Report {
            status,
            base: self.base,
            final_revision: self.scene.revision(),
            completed: self.completed,
            total: self.scheduler.total(),
            elapsed: self.started.elapsed(),
        }
    }
}

fn validate_commit(previous: &Snapshot, next: &Snapshot) -> Result<()> {
    ensure!(
        previous.project_id() == next.project_id(),
        "committer returned a snapshot from another project"
    );
    ensure!(
        next.revision() > previous.revision(),
        "committer did not advance the scene revision"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use koharu_scene::{At, Authored, PageDraft, SourceText, Translation};

    struct SessionCommitter {
        session: koharu_scene::Session,
    }

    #[async_trait]
    impl Committer for SessionCommitter {
        async fn commit(&mut self, output: StageOutput) -> anyhow::Result<Snapshot> {
            Ok(self.session.commit(output.patch).await?.snapshot)
        }
    }

    #[tokio::test]
    async fn late_translation_completion_is_committed_after_parallel_inpainting_failure() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let content = edit.add_text_content(page, At::End).unwrap();
        edit.set(
            content,
            &SourceText {
                text: Authored::user("source".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        let snapshot = session.snapshot();

        let mut translated = snapshot.edit();
        translated.observe::<SourceText>(content).unwrap();
        translated.observe::<Translation>(content).unwrap();
        translated
            .set(
                content,
                &Translation {
                    text: Authored::user("translated".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        let patch = translated.finish().unwrap();

        let device = koharu_ml::Device::cpu();
        let resources = ResourceMonitor::new(&device);
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let runner = Arc::new(
            StageRunner::new(
                &crate::PipelineConfig::default(),
                translator,
                &device,
                resources.clone(),
            )
            .unwrap(),
        );
        let mut committer = SessionCommitter { session };
        let mut execution = Execution::new(
            runner,
            resources,
            snapshot,
            Request {
                operation: crate::Operation::Only {
                    stage: Stage::Translation,
                },
                scope: crate::Scope::Pages(vec![page]),
                ..Request::default()
            },
            &mut committer,
        )
        .unwrap();
        execution.failure = Some(PipelineError::new(
            ErrorKind::Processing,
            Some(Stage::Inpainting),
            anyhow::anyhow!("inpainting failed"),
        ));

        execution
            .handle_completion(StageCompletion {
                target: StageTarget::Page(page),
                stage: Stage::Translation,
                model: "test-translator".to_owned(),
                elapsed: std::time::Duration::ZERO,
                outcome: Ok(StageOutcome::Patch(patch)),
            })
            .await;

        assert_eq!(
            execution
                .scene
                .component::<Translation>(content)
                .unwrap()
                .unwrap()
                .text
                .value,
            "translated"
        );
    }

    #[tokio::test]
    async fn project_inpainting_failure_is_reported_as_page_warning() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let next_page = edit
            .add_page(PageDraft::new("next page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        let snapshot = session.snapshot();

        let device = koharu_ml::Device::cpu();
        let resources = ResourceMonitor::new(&device);
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let runner = Arc::new(
            StageRunner::new(
                &crate::PipelineConfig::default(),
                translator,
                &device,
                resources.clone(),
            )
            .unwrap(),
        );
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let event_sink = events.clone();
        let mut committer = SessionCommitter { session };
        let mut execution = Execution::new(
            runner,
            resources,
            snapshot,
            Request {
                operation: crate::Operation::Only {
                    stage: Stage::Inpainting,
                },
                scope: crate::Scope::Project,
                progress: Some(Arc::new(move |event| {
                    event_sink.lock().unwrap().push(event);
                })),
                ..Request::default()
            },
            &mut committer,
        )
        .unwrap();

        execution
            .handle_completion(StageCompletion {
                target: StageTarget::Page(page),
                stage: Stage::Inpainting,
                model: "fal-model".to_owned(),
                elapsed: std::time::Duration::ZERO,
                outcome: Err(PipelineError::new(
                    ErrorKind::Processing,
                    Some(Stage::Inpainting),
                    anyhow::anyhow!("content policy"),
                )),
            })
            .await;

        assert!(execution.failure.is_none());
        assert_eq!(execution.completed, 1);
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, Progress::Warning { .. }))
        );
        let next = execution
            .scheduler
            .start_next(&BTreeSet::new())
            .expect("next page should continue");
        assert_eq!(next, (StageTarget::Page(next_page), Stage::Inpainting));
    }
}
