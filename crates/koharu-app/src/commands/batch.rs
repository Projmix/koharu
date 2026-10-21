use std::{collections::HashMap, time::SystemTime};

use anyhow::{Context as _, Result, bail};
use koharu_pipeline::{Operation, Pipeline, Scope, Stage, StageProgress};
use parking_lot::Mutex;
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Manager as _, State};
use tauri_runtime_cef::CefRuntime;

use super::{
    Error,
    lifecycle::replace_project,
    processing::{self, JobChannel, JobState, Processing},
    project::{CurrentProject, ProjectLibrary},
};

/// One library project with what each stage has left and its place in the
/// current batch run.
#[derive(Clone, Debug, Serialize, Type)]
pub struct BatchProject {
    pub name: String,
    pub stages: Vec<StageProgress>,
    pub run: Option<BatchState>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BatchState {
    Queued,
    Running,
    Finished,
    Failed,
    Stopped,
}

/// A batch runs the selected stages on several projects, one project after
/// another, through the same processing job a single project uses.
#[derive(Default)]
pub(crate) struct Batch {
    runs: Mutex<HashMap<String, (BatchState, Option<String>)>>,
    /// Stage progress of closed projects, keyed by when their state was
    /// written, so polling does not reopen unchanged projects.
    progress: Mutex<HashMap<String, (SystemTime, Vec<StageProgress>)>>,
}

impl Batch {
    fn running(&self) -> bool {
        self.runs
            .lock()
            .values()
            .any(|(state, _)| matches!(state, BatchState::Queued | BatchState::Running))
    }

    fn set(&self, name: &str, state: BatchState, error: Option<String>) {
        self.runs.lock().insert(name.to_owned(), (state, error));
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_batch_projects(
    handle: AppHandle<CefRuntime>,
) -> std::result::Result<Vec<BatchProject>, Error> {
    let library = handle.state::<ProjectLibrary>().inner().clone();
    let pipeline = handle.state::<Pipeline>().inner().clone();
    let batch = handle.state::<Batch>();
    let current = {
        let current = handle.state::<CurrentProject>();
        let current = current.project.lock().await;
        current
            .as_ref()
            .map(|project| (project.name.clone(), project.snapshot()))
    };
    let mut projects = Vec::new();
    for summary in library.list()? {
        let name = summary.name;
        let (run, error) = batch
            .runs
            .lock()
            .get(&name)
            .cloned()
            .map_or((None, None), |(state, error)| (Some(state), error));
        let stages = match &current {
            // The open project holds its lock; read its live snapshot.
            Some((open, snapshot)) if *open == name => Ok(pipeline.progress(snapshot)),
            _ => closed_progress(&library, &pipeline, &batch, &name).await,
        };
        let (stages, error) = match stages {
            Ok(stages) => (stages, error),
            Err(failure) => (Vec::new(), error.or(Some(format!("{failure:#}")))),
        };
        projects.push(BatchProject {
            name,
            stages,
            run,
            error,
        });
    }
    Ok(projects)
}

async fn closed_progress(
    library: &ProjectLibrary,
    pipeline: &Pipeline,
    batch: &Batch,
    name: &str,
) -> Result<Vec<StageProgress>> {
    let modified = library.modified(name)?;
    if let Some((cached, stages)) = batch.progress.lock().get(name)
        && *cached == modified
    {
        return Ok(stages.clone());
    }
    let stages = pipeline.progress(&library.snapshot(name).await?);
    batch
        .progress
        .lock()
        .insert(name.to_owned(), (modified, stages.clone()));
    Ok(stages)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn run_batch(
    projects: Vec<String>,
    stages: Vec<Stage>,
    handle: AppHandle<CefRuntime>,
    batch: State<'_, Batch>,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    if projects.is_empty() || stages.is_empty() {
        return Err(anyhow::anyhow!("select at least one project and one stage").into());
    }
    if batch.running() || !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!("another process is already running").into());
    }
    {
        let mut runs = batch.runs.lock();
        runs.clear();
        for name in &projects {
            runs.insert(name.clone(), (BatchState::Queued, None));
        }
    }
    drop(tokio::spawn(async move {
        let batch = handle.state::<Batch>();
        let mut stopped = false;
        for name in &projects {
            if stopped {
                batch.set(name, BatchState::Stopped, None);
                continue;
            }
            batch.set(name, BatchState::Running, None);
            let (state, error) = match run_project(&handle, name, &stages).await {
                Ok((state, error)) => (state, error),
                Err(error) => (JobState::Failed, Some(format!("{error:#}"))),
            };
            if let Some(error) = &error {
                tracing::error!(project = %name, %error, "batch project failed");
            }
            // Stopping the current job stops the batch; a failed project
            // does not keep the others from running.
            stopped = state == JobState::Stopped;
            let state = match state {
                JobState::Finished => BatchState::Finished,
                JobState::Stopped => BatchState::Stopped,
                JobState::Failed | JobState::Running => BatchState::Failed,
            };
            batch.set(name, state, error);
        }
    }));
    Ok(())
}

/// Open a project and run the selected stages on it. Cleaning has no skip,
/// so a batch cleans only pages without a cleanup layer, after the earlier
/// stages have finished.
async fn run_project(
    handle: &AppHandle<CefRuntime>,
    name: &str,
    stages: &[Stage],
) -> Result<(JobState, Option<String>)> {
    let open = handle
        .state::<CurrentProject>()
        .project
        .lock()
        .await
        .as_ref()
        .is_some_and(|project| project.name == name);
    if !open {
        let opened = handle.state::<ProjectLibrary>().open(name).await?;
        replace_project(handle, opened).await?;
    }
    let snapshot = || async {
        handle
            .state::<CurrentProject>()
            .project
            .lock()
            .await
            .as_ref()
            .map(|project| project.snapshot())
            .context("no project is open")
    };
    let pipeline = handle.state::<Pipeline>().inner().clone();
    let progress = pipeline.progress(&snapshot().await?);
    for stage in stages {
        for required in stage.requires() {
            let done = progress
                .iter()
                .find(|entry| entry.stage == *required)
                .is_some_and(|entry| entry.done > 0);
            if !done && !stages.contains(required) {
                bail!("{stage} needs {required}, which has not run on this project");
            }
        }
    }

    let earlier = stages
        .iter()
        .copied()
        .filter(|stage| *stage != Stage::Inpainting)
        .collect::<Vec<_>>();
    if !earlier.is_empty() {
        let (state, error) = run_job(handle, Scope::Project, earlier).await?;
        if state != JobState::Finished {
            return Ok((state, error));
        }
    }
    if stages.contains(&Stage::Inpainting) {
        let pages = pipeline.unfinished(&snapshot().await?, Stage::Inpainting);
        if !pages.is_empty() {
            return run_job(handle, Scope::Pages(pages), vec![Stage::Inpainting]).await;
        }
    }
    Ok((JobState::Finished, None))
}

async fn run_job(
    handle: &AppHandle<CefRuntime>,
    scope: Scope,
    stages: Vec<Stage>,
) -> Result<(JobState, Option<String>)> {
    let (_, finished) = processing::start_job(
        handle.clone(),
        scope,
        Operation::Stages { stages },
        handle.state::<CurrentProject>(),
        handle.state::<Processing>(),
        handle.state::<JobChannel>(),
        None,
    )
    .await
    .map_err(|error| error.0)?;
    finished.await.context("processing task failed")
}
