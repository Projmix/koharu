use std::{collections::HashMap, fmt, sync::Arc};

use anyhow::{Context as _, Result};
use koharu_pipeline::{Committer, Progress, RunStatus, StageOutput, StopToken};
use koharu_scene::{EntityId, ProjectId, Snapshot};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use specta::Type;
use tauri::{AppHandle, Cef, Manager as _, State, ipc::Channel};
use uuid::Uuid;

use super::{
    ChannelExt as _, Error,
    canvas::CanvasChannel,
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
    pub(crate) stops: Mutex<HashMap<JobId, StopToken>>,
    pub(crate) jobs: Mutex<HashMap<JobId, Job>>,
    pub(crate) inpainting_mask: Mutex<Option<koharu_pipeline::InpaintingMask>>,
    pub(crate) warnings: Mutex<HashMap<(ProjectId, EntityId), PageWarning>>,
    resume: Mutex<Option<ProcessingRequest>>,
}

impl Processing {
    pub(crate) fn clear_warnings(&self) {
        self.warnings.lock().clear();
    }

    pub(crate) fn warning(&self, project: ProjectId, page: EntityId) -> Option<PageWarning> {
        self.warnings.lock().get(&(project, page)).cloned()
    }

    fn clear_warning(&self, project: ProjectId, page: EntityId) {
        self.warnings.lock().remove(&(project, page));
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
        koharu_pipeline::Operation::Through { .. } | koharu_pipeline::Operation::Only { .. } => {
            false
        }
        koharu_pipeline::Operation::Stages { stages } => {
            stages.contains(&koharu_pipeline::Stage::Inpainting)
        }
    }
}

#[derive(Default)]
pub(crate) struct JobChannel {
    pub(crate) channel: Mutex<Option<Channel<Job>>>,
}

#[tauri::command]
#[specta::specta]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process(
    handle: AppHandle<Cef>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    job_channel: State<'_, JobChannel>,
) -> std::result::Result<JobId, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let id = JobId::new();
    let stop = StopToken::default();
    {
        let mut stops = processing.stops.lock();
        if !stops.is_empty() {
            return Err(anyhow::anyhow!("another process is already running").into());
        }
        stops.insert(id, stop.clone());
    }
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

    let inpainting_mask = processing.inpainting_mask.lock().take();
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
    drop(tokio::spawn(async move {
        let progress = Arc::new(Mutex::new((0_usize, 0_usize)));
        let progress_handle = task_handle.clone();
        let mut request = koharu_pipeline::Request {
            operation: request.operation,
            scope: request.scope,
            stop: stop.clone(),
            progress: None,
            inpainting_mask: request.inpainting_mask,
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
                Progress::Started {
                    pages,
                    stages,
                    total,
                } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "pipeline_start",
                        page_count = pages.len(),
                        stage_count = stages.len(),
                    );
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
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_loading",
                        stage = %stage,
                        model,
                    );
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
                    elapsed,
                } => {
                    if stage != koharu_pipeline::Stage::Translation {
                        tracing::info!(
                            target: "koharu_metrics",
                            metric = "model_run",
                            stage = %stage,
                            model,
                            duration_ms = elapsed.as_secs_f64() * 1000.0,
                        );
                    }
                    if stage == koharu_pipeline::Stage::Inpainting
                        && let Some(page) = target.page()
                    {
                        progress_handle
                            .state::<Processing>()
                            .clear_warning(project_id, page);
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
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_skip",
                        stage = %stage,
                    );
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    if stage == koharu_pipeline::Stage::Inpainting
                        && let Some(page) = target.page()
                    {
                        progress_handle
                            .state::<Processing>()
                            .clear_warning(project_id, page);
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
                Progress::Running { stage, model, .. } => {
                    tracing::info!(
                        target: "koharu_metrics",
                        metric = "stage_running",
                        stage = %stage,
                        model,
                    );
                    None
                }
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
            handle: AppHandle<Cef>,
        }

        #[async_trait::async_trait]
        impl Committer for PipelineCommitter {
            async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
                let (commit, page) = {
                    let projects = self.handle.state::<CurrentProject>();
                    let mut projects = projects.project.lock().await;
                    let project = projects.as_mut().context("no project is open")?;
                    let Some(commit) = project.commit_rebased(output.patch).await? else {
                        return Ok(project.snapshot());
                    };
                    project.record_commit(&commit);
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
        tracing::info!(
            target: "koharu_metrics",
            metric = "pipeline_result",
            outcome = if stopped {
                "stopped"
            } else if error.is_some() {
                "failed"
            } else {
                "completed"
            },
        );
        task_handle
            .state::<Processing>()
            .finish(retry, stopped || error.is_some());
        task_handle.state::<Processing>().stops.lock().remove(&id);
        let job = task_handle
            .state::<Processing>()
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = if stopped {
                    JobState::Stopped
                } else if error.is_some() {
                    JobState::Failed
                } else {
                    JobState::Finished
                };
                job.error = error;
                job
            });
        if let Some(job) = job {
            task_handle.state::<JobChannel>().channel.publish(job);
        }
    }));
    Ok(id)
}

#[derive(Serialize)]
struct ChapterTranslationExport {
    format: &'static str,
    version: u32,
    project_id: String,
    revision: String,
    provider: String,
    model: String,
    endpoint: &'static str,
    request: Value,
    system_prompt: Value,
    chapter_payload: Value,
}

/// Save the exact, credential-free OpenRouter request used for a chapter.
#[tauri::command]
#[specta::specta]
pub(crate) async fn export_chapter_translation(
    window: tauri::WebviewWindow<Cef>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    if !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!(
            "chapter translation export is unavailable while processing is running"
        )
        .into());
    }
    let Some(file) = rfd::AsyncFileDialog::new()
        .add_filter("JSON", &["json"])
        .set_file_name("koharu-chapter-translation-request.json")
        .set_parent(&window)
        .save_file()
        .await
    else {
        return Ok(());
    };
    let snapshot = {
        let project = project.project.lock().await;
        project.as_ref().context("no project is open")?.snapshot()
    };
    let config = koharu_pipeline::PipelineConfig::load()?.read()?.clone();
    let chapter =
        koharu_pipeline::ChapterTranslation::from_snapshot(snapshot.clone(), &config.translation)?;
    let request = chapter.openrouter_request()?;
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .context("OpenRouter request did not contain messages")?;
    let system_prompt = messages
        .first()
        .and_then(|message| message.get("content"))
        .cloned()
        .context("OpenRouter request did not contain a system prompt")?;
    let chapter_payload = messages
        .get(1)
        .and_then(|message| message.get("content"))
        .cloned()
        .context("OpenRouter request did not contain a chapter payload")?;
    let model = chapter
        .model()
        .model
        .clone()
        .context("OpenRouter requires a selected translation model")?;
    let export = ChapterTranslationExport {
        format: "koharu.chapter-translation-request",
        version: 1,
        project_id: snapshot.project_id().to_string(),
        revision: snapshot.revision().to_string(),
        provider: chapter.model().provider.to_string(),
        model,
        endpoint: "https://openrouter.ai/api/v1/chat/completions",
        request,
        system_prompt,
        chapter_payload,
    };
    let bytes = serde_json::to_vec_pretty(&export)?;
    tokio::fs::write(file.path(), bytes).await?;
    tracing::info!(target: "koharu_metrics", metric = "chapter_translation_exported");
    Ok(())
}

/// Import a manually completed chapter translation and commit it atomically.
#[tauri::command]
#[specta::specta]
pub(crate) async fn import_chapter_translation(
    window: tauri::WebviewWindow<Cef>,
    handle: AppHandle<Cef>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    if !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!(
            "chapter translation import is unavailable while processing is running"
        )
        .into());
    }
    let Some(file) = rfd::AsyncFileDialog::new()
        .add_filter("JSON", &["json"])
        .set_parent(&window)
        .pick_file()
        .await
    else {
        return Ok(());
    };
    let bytes = tokio::fs::read(file.path()).await?;
    let value: Value =
        serde_json::from_slice(&bytes).context("chapter translation file is not valid JSON")?;
    let response = if value.get("translations").is_some() {
        value
    } else if value.get("choices").is_some() {
        value
    } else if let Some(response) = value.get("response") {
        response.clone()
    } else {
        return Err(anyhow::anyhow!(
            "chapter translation JSON must contain translations or an OpenRouter choices response"
        )
        .into());
    };
    let snapshot = {
        let project = project.project.lock().await;
        project.as_ref().context("no project is open")?.snapshot()
    };
    let config = koharu_pipeline::PipelineConfig::load()?.read()?.clone();
    let chapter =
        koharu_pipeline::ChapterTranslation::from_snapshot(snapshot, &config.translation)?;
    let patch = chapter.patch_from_response(&serde_json::to_string(&response)?)?;
    let (commit, page) = {
        let mut projects = project.project.lock().await;
        let project = projects.as_mut().context("no project is open")?;
        let commit = project
            .commit_rebased(patch)
            .await?
            .context("the project changed while importing; export a fresh chapter request")?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let desktop = handle.state::<Desktop>();
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    handle
        .state::<CanvasChannel>()
        .channel
        .publish(desktop.canvas_state());
    tracing::info!(target: "koharu_metrics", metric = "chapter_translation_imported");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(processing.warning(project, page), Some(warning));

        processing.clear_warning(project, page);
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

#[tracing::instrument(
    target = "koharu_metrics",
    name = "pipeline_stop",
    skip_all,
    fields(state = "requested")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn stop_job(
    job: JobId,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    let stops = processing.stops.lock();
    let stop = stops
        .get(&job)
        .with_context(|| format!("job {job} is not running"))?;
    stop.stop();
    Ok(())
}
