use std::{collections::HashMap, fmt, sync::Arc};

use anyhow::{Context as _, Result};
use koharu_pipeline::{Committer, Progress, RunStatus, StageOutput, StopToken};
use koharu_scene::{EntityId, ProjectId, Snapshot};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Manager as _, State, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use uuid::Uuid;

use super::{
    ChannelExt as _, Error,
    canvas::CanvasChannel,
    image_edit_preview::DesktopImageEditReviewer,
    project::{CurrentProject, PageWarning},
};
use koharu_desktop::Desktop;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct JobId(Uuid);

impl JobId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Job {
    pub id: JobId,
    pub state: JobState,
    #[specta(type = f64)]
    pub completed: usize,
    #[specta(type = f64)]
    pub total: usize,
    pub target: Option<koharu_pipeline::StageTarget>,
    pub stage: Option<koharu_pipeline::Stage>,
    pub model: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Finished,
    Failed,
    Stopped,
}

#[derive(Default)]
pub(crate) struct Processing {
    running: Mutex<HashMap<JobId, Running>>,
    pub(crate) jobs: Mutex<HashMap<JobId, Job>>,
    pub(crate) warnings: Mutex<HashMap<(ProjectId, EntityId), PageWarning>>,
    resume: Mutex<Option<ProcessingRequest>>,
}

/// A running job and what it may write. The owner's own Remove, Restore and
/// region OCR run beside anything, so the owner keeps cleaning while an agent
/// runs detection and OCR; pipeline runs exclude each other, and one that
/// paints cleanup layers excludes imported cleanup images.
struct Running {
    stop: StopToken,
    manual: bool,
    cleaning: bool,
}

impl Processing {
    /// Admits job `id`, or fails when its work collides with a running job.
    pub(crate) fn admit(
        &self,
        id: JobId,
        stop: StopToken,
        scope: &koharu_pipeline::Scope,
        operation: &koharu_pipeline::Operation,
        inpainting_mask: Option<&koharu_pipeline::InpaintingMask>,
    ) -> anyhow::Result<()> {
        let manual = accepts_queued_manual_request(scope, operation, inpainting_mask);
        let mut running = self.running.lock();
        if !manual && running.values().any(|job| !job.manual) {
            anyhow::bail!("another process is already running");
        }
        running.insert(
            id,
            Running {
                stop,
                manual,
                cleaning: !manual && requests_inpainting(operation),
            },
        );
        Ok(())
    }

    pub(crate) fn release(&self, id: JobId) {
        self.running.lock().remove(&id);
    }

    /// Whether a pipeline run, not one of the owner's own edits, is running.
    pub(crate) fn pipeline_running(&self) -> bool {
        self.running.lock().values().any(|job| !job.manual)
    }

    /// Whether a running pipeline paints cleanup layers.
    pub(crate) fn cleaning_running(&self) -> bool {
        self.running.lock().values().any(|job| job.cleaning)
    }

    pub(crate) fn stop(&self, id: JobId) -> anyhow::Result<()> {
        self.running
            .lock()
            .get(&id)
            .with_context(|| format!("job {id} is not running"))?
            .stop
            .stop();
        Ok(())
    }

    pub(crate) fn stop_all(&self) {
        for job in self.running.lock().drain().map(|(_, job)| job) {
            job.stop.stop();
        }
    }

    pub(crate) fn clear_warnings(&self) {
        self.warnings.lock().clear();
    }

    pub(crate) fn warning(&self, project: ProjectId, page: EntityId) -> Option<PageWarning> {
        self.warnings.lock().get(&(project, page)).cloned()
    }

    /// Clears the page's warning when `stage` raised it.
    fn clear_warning(&self, project: ProjectId, page: EntityId, stage: koharu_pipeline::Stage) {
        let mut warnings = self.warnings.lock();
        if warnings
            .get(&(project, page))
            .is_some_and(|warning| warning.stage == stage)
        {
            warnings.remove(&(project, page));
        }
    }

    fn record_warning(&self, project: ProjectId, page: EntityId, warning: PageWarning) {
        self.warnings.lock().insert((project, page), warning);
    }

    fn clear_inpainting_warnings(
        &self,
        project: ProjectId,
        snapshot: &Snapshot,
        scope: &koharu_pipeline::Scope,
        operation: &koharu_pipeline::Operation,
    ) {
        if !requests_inpainting(operation) {
            return;
        }
        let pages = match scope {
            koharu_pipeline::Scope::Project => snapshot.pages().map(|page| page.id()).collect(),
            koharu_pipeline::Scope::Pages(pages) => pages.clone(),
            koharu_pipeline::Scope::Region { page, .. } => vec![*page],
            koharu_pipeline::Scope::Entities(_) => Vec::new(),
        };
        let mut warnings = self.warnings.lock();
        for page in pages {
            warnings.remove(&(project, page));
        }
    }
}

#[derive(Clone)]
struct ProcessingRequest {
    project: koharu_scene::ProjectId,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    inpainting_mask: Option<koharu_pipeline::InpaintingMask>,
}

impl Processing {
    fn select_request(&self, request: ProcessingRequest) -> ProcessingRequest {
        if !matches!(request.operation, koharu_pipeline::Operation::Stages { .. }) {
            return request;
        }
        // A stopped/failed run is only a retry for the same explicit selection.
        // Reusing it for a different scope or stage set makes a later Run appear
        // to start at a random stage (for example Translation after OCR was
        // deleted).  The current UI selection is authoritative in that case.
        self.resume
            .lock()
            .take()
            .filter(|resume| {
                resume.project == request.project
                    && resume.scope == request.scope
                    && resume.operation == request.operation
            })
            .unwrap_or(request)
    }

    fn finish(&self, request: ProcessingRequest, interrupted: bool) {
        if interrupted && matches!(request.operation, koharu_pipeline::Operation::Stages { .. }) {
            *self.resume.lock() = Some(request);
        }
    }
}

fn requests_inpainting(operation: &koharu_pipeline::Operation) -> bool {
    match operation {
        koharu_pipeline::Operation::Full
        | koharu_pipeline::Operation::Through {
            stage: koharu_pipeline::Stage::Inpainting,
        }
        | koharu_pipeline::Operation::Only {
            stage: koharu_pipeline::Stage::Inpainting,
        } => true,
        koharu_pipeline::Operation::Through { .. }
        | koharu_pipeline::Operation::Only { .. }
        | koharu_pipeline::Operation::Shorten => false,
        koharu_pipeline::Operation::Stages { stages } => {
            stages.contains(&koharu_pipeline::Stage::Inpainting)
        }
    }
}

fn accepts_queued_manual_request(
    scope: &koharu_pipeline::Scope,
    operation: &koharu_pipeline::Operation,
    inpainting_mask: Option<&koharu_pipeline::InpaintingMask>,
) -> bool {
    let masked_inpainting = matches!(
        (scope, operation, inpainting_mask),
        (
            koharu_pipeline::Scope::Region { page, .. },
            koharu_pipeline::Operation::Only {
                stage: koharu_pipeline::Stage::Inpainting,
            },
            Some(mask),
        ) if *page == mask.page
    );
    let entity_ocr = matches!(
        (scope, operation),
        (
            koharu_pipeline::Scope::Entities(entities),
            koharu_pipeline::Operation::Stages { stages },
        ) if !entities.is_empty() && stages.as_slice() == [koharu_pipeline::Stage::Ocr]
    );
    masked_inpainting || entity_ocr
}

#[derive(Default)]
pub(crate) struct JobChannel {
    pub(crate) channel: Mutex<Option<Channel<Job>>>,
}

#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process(
    handle: AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
) -> std::result::Result<JobId, Error> {
    process_with_mask(
        handle,
        scope,
        operation,
        project,
        processing,
        job_channel,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn process_with_mask(
    handle: AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
    inpainting_mask: Option<koharu_pipeline::InpaintingMask>,
) -> std::result::Result<JobId, Error> {
    let (id, _finished) = start_job(
        handle,
        scope,
        operation,
        project,
        processing,
        job_channel,
        inpainting_mask,
    )
    .await?;
    Ok(id)
}

/// Start a processing job; the returned task ends with the job's final state
/// and error.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn start_job(
    handle: AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
    inpainting_mask: Option<koharu_pipeline::InpaintingMask>,
) -> std::result::Result<(JobId, tokio::task::JoinHandle<(JobState, Option<String>)>), Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let id = JobId::new();
    let image_edit_reviewer = DesktopImageEditReviewer::for_job(handle.clone(), id)?;
    let stop = StopToken::default();
    processing.admit(
        id,
        stop.clone(),
        &scope,
        &operation,
        inpainting_mask.as_ref(),
    )?;
    let job = Job {
        id,
        state: JobState::Running,
        completed: 0,
        total: 0,
        target: None,
        stage: None,
        model: None,
        error: None,
    };
    processing.jobs.lock().insert(id, job.clone());
    job_channel.channel.publish(job);

    let request = processing.select_request(ProcessingRequest {
        project: snapshot.project_id(),
        scope,
        operation,
        inpainting_mask,
    });
    processing.clear_inpainting_warnings(
        snapshot.project_id(),
        &snapshot,
        &request.scope,
        &request.operation,
    );
    let retry = request.clone();
    let project_id = snapshot.project_id();
    let pipeline = handle.state::<koharu_pipeline::Pipeline>().inner().clone();
    let task_handle = handle.clone();
    let finished = tokio::spawn(async move {
        let progress = Arc::new(Mutex::new((0_usize, 0_usize)));
        let progress_handle = task_handle.clone();
        let mut request = koharu_pipeline::Request {
            operation: request.operation,
            scope: request.scope,
            stop: stop.clone(),
            progress: None,
            inpainting_mask: request.inpainting_mask,
            image_edit_reviewer,
        };
        request.progress = Some(Arc::new(move |event| {
            enum JobUpdate {
                Replace {
                    completed: usize,
                    total: usize,
                    target: Option<koharu_pipeline::StageTarget>,
                    stage: Option<koharu_pipeline::Stage>,
                    model: Option<String>,
                },
                CountOnly {
                    completed: usize,
                    total: usize,
                },
            }

            let update = match event {
                Progress::Started { total, .. } => {
                    let mut progress = progress.lock();
                    *progress = (0, total);
                    Some(JobUpdate::Replace {
                        completed: 0,
                        total: progress.1,
                        target: None,
                        stage: None,
                        model: None,
                    })
                }
                Progress::Loading {
                    target,
                    stage,
                    model,
                } => {
                    let progress = progress.lock();
                    Some(JobUpdate::Replace {
                        completed: progress.0,
                        total: progress.1,
                        target: Some(target),
                        stage: Some(stage),
                        model: Some(model),
                    })
                }
                Progress::Finished {
                    target,
                    stage,
                    model,
                    ..
                } => {
                    if matches!(
                        stage,
                        koharu_pipeline::Stage::Enhancement | koharu_pipeline::Stage::Inpainting
                    ) && let Some(page) = target.page()
                    {
                        progress_handle
                            .state::<Processing>()
                            .clear_warning(project_id, page, stage);
                    }
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some(JobUpdate::Replace {
                        completed: progress.0,
                        total: progress.1,
                        target: Some(target),
                        stage: Some(stage),
                        model: Some(model),
                    })
                }
                Progress::Skipped { target, stage } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    if matches!(
                        stage,
                        koharu_pipeline::Stage::Enhancement | koharu_pipeline::Stage::Inpainting
                    ) && let Some(page) = target.page()
                    {
                        progress_handle
                            .state::<Processing>()
                            .clear_warning(project_id, page, stage);
                    }
                    // A skipped stage is not active. Keep the last loading or
                    // finished stage visible while concurrent work continues.
                    let _ = target;
                    Some(JobUpdate::CountOnly {
                        completed: progress.0,
                        total: progress.1,
                    })
                }
                Progress::Warning {
                    target,
                    stage,
                    model,
                    error,
                } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    let active_job = progress_handle
                        .state::<Processing>()
                        .jobs
                        .lock()
                        .contains_key(&id);
                    if active_job && let Some(page) = target.page() {
                        progress_handle.state::<Processing>().record_warning(
                            project_id,
                            page,
                            PageWarning {
                                stage,
                                model: model.clone(),
                                message: error.clone(),
                            },
                        );
                    }
                    Some(JobUpdate::Replace {
                        completed: progress.0,
                        total: progress.1,
                        target: Some(target),
                        stage: Some(stage),
                        model: Some(model),
                    })
                }
                Progress::Running { .. } => None,
            };
            if let Some(update) = update {
                let job = {
                    let processing = progress_handle.state::<Processing>();
                    let mut jobs = processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        match update {
                            JobUpdate::Replace {
                                completed,
                                total,
                                target,
                                stage,
                                model,
                            } => {
                                job.completed = completed;
                                job.total = total;
                                job.target = target;
                                job.stage = stage;
                                job.model = model;
                            }
                            JobUpdate::CountOnly { completed, total } => {
                                job.completed = completed;
                                job.total = total;
                            }
                        }
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_handle.state::<JobChannel>().channel.publish(job);
                }
            }
        }));

        struct PipelineCommitter {
            handle: AppHandle<CefRuntime>,
        }

        #[async_trait::async_trait]
        impl Committer for PipelineCommitter {
            async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
                let (commit, page) = {
                    let projects = self.handle.state::<CurrentProject>();
                    let mut projects = projects.project.lock().await;
                    let project = projects.as_mut().context("no project is open")?;
                    let Some(commit) = project.commit_stage_output(output).await? else {
                        return Ok(project.snapshot());
                    };
                    let page = project.active_page();
                    (commit, page)
                };
                let snapshot = commit.snapshot.clone();
                let desktop = self.handle.state::<Desktop>();
                desktop.synchronize(&commit.snapshot, page, &commit).await?;
                let canvas = desktop.canvas_state();
                self.handle.state::<CanvasChannel>().channel.publish(canvas);
                Ok(snapshot)
            }
        }

        let mut committer = PipelineCommitter {
            handle: task_handle.clone(),
        };
        let result = pipeline.execute(snapshot, request, &mut committer).await;
        let (stopped, error) = match result {
            Ok(report) => (report.status == RunStatus::Stopped, None),
            Err(error) => {
                tracing::error!(stage = ?error.stage, %error, "processing failed");
                (false, Some(format!("{error:#}")))
            }
        };
        task_handle
            .state::<Processing>()
            .finish(retry, stopped || error.is_some());
        task_handle.state::<Processing>().release(id);
        let state = if stopped {
            JobState::Stopped
        } else if error.is_some() {
            JobState::Failed
        } else {
            JobState::Finished
        };
        let job = task_handle
            .state::<Processing>()
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = state;
                job.error.clone_from(&error);
                job
            });
        if let Some(job) = job {
            task_handle.state::<JobChannel>().channel.publish(job);
        }
        (state, error)
    });
    Ok((id, finished))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_policy_queues_manual_requests_but_rejects_another_processing_run() {
        let page = koharu_scene::EntityId::new();
        let entity = koharu_scene::EntityId::new();
        let mask = koharu_pipeline::InpaintingMask {
            page,
            png: Arc::from(Vec::<u8>::new()),
            intent: koharu_pipeline::InpaintingIntent::Remove,
        };

        assert!(accepts_queued_manual_request(
            &koharu_pipeline::Scope::Region {
                page,
                bounds: koharu_pipeline::Bounds {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                },
            },
            &koharu_pipeline::Operation::Only {
                stage: koharu_pipeline::Stage::Inpainting,
            },
            Some(&mask),
        ));
        assert!(accepts_queued_manual_request(
            &koharu_pipeline::Scope::Entities(vec![entity]),
            &koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Ocr],
            },
            None,
        ));
        assert!(!accepts_queued_manual_request(
            &koharu_pipeline::Scope::Project,
            &koharu_pipeline::Operation::Full,
            None,
        ));
        assert!(!accepts_queued_manual_request(
            &koharu_pipeline::Scope::Pages(vec![page]),
            &koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Ocr],
            },
            None,
        ));
    }

    #[test]
    fn the_owners_edits_run_beside_a_pipeline_run_and_stop_by_job() {
        let processing = Processing::default();
        let page = koharu_scene::EntityId::new();
        let mask = koharu_pipeline::InpaintingMask {
            page,
            png: Arc::from(Vec::<u8>::new()),
            intent: koharu_pipeline::InpaintingIntent::Remove,
        };
        let remove = koharu_pipeline::Scope::Region {
            page,
            bounds: koharu_pipeline::Bounds {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        };
        let inpainting = koharu_pipeline::Operation::Only {
            stage: koharu_pipeline::Stage::Inpainting,
        };
        let ocr = koharu_pipeline::Operation::Stages {
            stages: vec![
                koharu_pipeline::Stage::Detection,
                koharu_pipeline::Stage::Ocr,
            ],
        };
        let (owner, agent) = (JobId::new(), JobId::new());
        let (owner_stop, agent_stop) = (StopToken::default(), StopToken::default());

        processing
            .admit(owner, owner_stop.clone(), &remove, &inpainting, Some(&mask))
            .unwrap();
        processing
            .admit(
                agent,
                agent_stop.clone(),
                &koharu_pipeline::Scope::Project,
                &ocr,
                None,
            )
            .unwrap();
        assert!(
            processing
                .admit(
                    JobId::new(),
                    StopToken::default(),
                    &koharu_pipeline::Scope::Project,
                    &ocr,
                    None,
                )
                .is_err()
        );
        assert!(processing.pipeline_running());
        assert!(!processing.cleaning_running());

        processing.stop(owner).unwrap();
        assert!(owner_stop.stopped());
        assert!(!agent_stop.stopped());
    }

    #[tokio::test]
    async fn retry_uses_the_original_scope_and_stages_when_selection_matches() {
        let processing = Processing::default();
        let project = koharu_scene::Session::memory()
            .await
            .unwrap()
            .snapshot()
            .project_id();
        let original_page = koharu_scene::EntityId::new();
        let original = ProcessingRequest {
            project,
            scope: koharu_pipeline::Scope::Pages(vec![original_page]),
            operation: koharu_pipeline::Operation::Stages {
                stages: vec![
                    koharu_pipeline::Stage::Detection,
                    koharu_pipeline::Stage::Ocr,
                    koharu_pipeline::Stage::Translation,
                    koharu_pipeline::Stage::Inpainting,
                ],
            },
            inpainting_mask: None,
        };
        processing.finish(original, true);

        let selected = processing.select_request(ProcessingRequest {
            project,
            scope: koharu_pipeline::Scope::Pages(vec![original_page]),
            operation: koharu_pipeline::Operation::Stages {
                stages: vec![
                    koharu_pipeline::Stage::Detection,
                    koharu_pipeline::Stage::Ocr,
                    koharu_pipeline::Stage::Translation,
                    koharu_pipeline::Stage::Inpainting,
                ],
            },
            inpainting_mask: None,
        });

        assert_eq!(
            selected.scope,
            koharu_pipeline::Scope::Pages(vec![original_page])
        );
        assert_eq!(
            selected.operation,
            koharu_pipeline::Operation::Stages {
                stages: vec![
                    koharu_pipeline::Stage::Detection,
                    koharu_pipeline::Stage::Ocr,
                    koharu_pipeline::Stage::Translation,
                    koharu_pipeline::Stage::Inpainting,
                ],
            }
        );
    }

    #[tokio::test]
    async fn a_changed_selection_does_not_consume_an_old_retry() {
        let processing = Processing::default();
        let project = koharu_scene::Session::memory()
            .await
            .unwrap()
            .snapshot()
            .project_id();
        let old_page = koharu_scene::EntityId::new();
        processing.finish(
            ProcessingRequest {
                project,
                scope: koharu_pipeline::Scope::Pages(vec![old_page]),
                operation: koharu_pipeline::Operation::Stages {
                    stages: vec![koharu_pipeline::Stage::Translation],
                },
                inpainting_mask: None,
            },
            true,
        );

        let new_page = koharu_scene::EntityId::new();
        let selected = processing.select_request(ProcessingRequest {
            project,
            scope: koharu_pipeline::Scope::Pages(vec![new_page]),
            operation: koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Ocr],
            },
            inpainting_mask: None,
        });

        assert_eq!(
            selected.scope,
            koharu_pipeline::Scope::Pages(vec![new_page])
        );
        assert_eq!(
            selected.operation,
            koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Ocr],
            }
        );
    }

    #[tokio::test]
    async fn manual_inpainting_does_not_replace_a_pipeline_retry() {
        let processing = Processing::default();
        let project = koharu_scene::Session::memory()
            .await
            .unwrap()
            .snapshot()
            .project_id();
        let original_page = koharu_scene::EntityId::new();
        processing.finish(
            ProcessingRequest {
                project,
                scope: koharu_pipeline::Scope::Pages(vec![original_page]),
                operation: koharu_pipeline::Operation::Stages {
                    stages: vec![koharu_pipeline::Stage::Detection],
                },
                inpainting_mask: None,
            },
            true,
        );
        processing.finish(
            ProcessingRequest {
                project,
                scope: koharu_pipeline::Scope::Region {
                    page: koharu_scene::EntityId::new(),
                    bounds: koharu_pipeline::Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                },
                operation: koharu_pipeline::Operation::Only {
                    stage: koharu_pipeline::Stage::Inpainting,
                },
                inpainting_mask: None,
            },
            true,
        );

        let selected = processing.select_request(ProcessingRequest {
            project,
            scope: koharu_pipeline::Scope::Pages(vec![original_page]),
            operation: koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Detection],
            },
            inpainting_mask: None,
        });

        assert_eq!(
            selected.scope,
            koharu_pipeline::Scope::Pages(vec![original_page])
        );
        assert_eq!(
            selected.operation,
            koharu_pipeline::Operation::Stages {
                stages: vec![koharu_pipeline::Stage::Detection],
            }
        );
    }

    #[tokio::test]
    async fn page_warning_is_replaced_and_cleared_by_page() {
        let processing = Processing::default();
        let project = koharu_scene::Session::memory()
            .await
            .unwrap()
            .snapshot()
            .project_id();
        let page = koharu_scene::EntityId::new();
        let warning = PageWarning {
            stage: koharu_pipeline::Stage::Inpainting,
            model: "fal-model".to_owned(),
            message: "content policy".to_owned(),
        };

        processing.record_warning(project, page, warning.clone());
        assert_eq!(processing.warning(project, page), Some(warning.clone()));

        processing.clear_warning(project, page, koharu_pipeline::Stage::Enhancement);
        assert_eq!(processing.warning(project, page), Some(warning));
        processing.clear_warning(project, page, koharu_pipeline::Stage::Inpainting);
        assert_eq!(processing.warning(project, page), None);
    }

    #[tokio::test]
    async fn warnings_are_cleared_when_processing_state_is_reset() {
        let processing = Processing::default();
        let project = koharu_scene::Session::memory()
            .await
            .unwrap()
            .snapshot()
            .project_id();
        let page = koharu_scene::EntityId::new();
        processing.record_warning(
            project,
            page,
            PageWarning {
                stage: koharu_pipeline::Stage::Inpainting,
                model: "lama".to_owned(),
                message: "failed".to_owned(),
            },
        );

        processing.clear_warnings();

        assert_eq!(processing.warning(project, page), None);
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn stop_job(
    job: JobId,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    processing.stop(job)?;
    Ok(())
}
