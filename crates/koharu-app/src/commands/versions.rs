use anyhow::{Context as _, Result, bail};
use koharu_desktop::Desktop;
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Cef, Manager as _, State};

use super::{
    ChannelExt as _, Error,
    agent::AgentState,
    canvas::CanvasChannel,
    lifecycle::ProjectChannel,
    processing::Processing,
    project::{CurrentProject, ProjectInfo},
};

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectVersion {
    pub id: koharu_scene::VersionId,
    pub name: String,
    #[specta(type = f64)]
    pub created_at_ms: u64,
    pub revision: koharu_scene::Revision,
}

impl From<koharu_scene::SavedVersion> for ProjectVersion {
    fn from(value: koharu_scene::SavedVersion) -> Self {
        Self {
            id: value.id,
            name: value.name,
            created_at_ms: value.created_at_ms,
            revision: value.revision,
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_project_versions(
    project: State<'_, CurrentProject>,
) -> std::result::Result<Vec<ProjectVersion>, Error> {
    let project = project.project.lock().await;
    let project = project.as_ref().context("no project is open")?;
    Ok(project
        .list_versions()
        .await?
        .into_iter()
        .map(Into::into)
        .collect())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn save_project_version(
    name: String,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
) -> std::result::Result<ProjectVersion, Error> {
    ensure_idle(&processing)?;
    let project = project.project.lock().await;
    let project = project.as_ref().context("no project is open")?;
    Ok(project.save_version(name).await?.into())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_project_version(
    id: koharu_scene::VersionId,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    ensure_idle(&processing)?;
    let project = project.project.lock().await;
    let project = project.as_ref().context("no project is open")?;
    project.delete_version(id).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn restore_project_version(
    id: koharu_scene::VersionId,
    handle: AppHandle<Cef>,
) -> std::result::Result<ProjectInfo, Error> {
    ensure_idle(&handle.state::<Processing>())?;
    handle.state::<AgentState>().reset().await;
    let (snapshot, active_page, info) = {
        let current = handle.state::<CurrentProject>();
        let mut project = current.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        project.restore_version(id).await?;
        (project.snapshot(), project.active_page(), project.info())
    };

    handle.state::<Processing>().clear_warnings();
    let desktop = handle.state::<Desktop>();
    desktop.show_page(&snapshot, active_page).await?;
    handle
        .state::<CanvasChannel>()
        .channel
        .publish(desktop.canvas_state());
    handle
        .state::<ProjectChannel>()
        .channel
        .publish(Some(info.clone()));
    Ok(info)
}

fn ensure_idle(processing: &Processing) -> Result<()> {
    if !processing.stops.lock().is_empty() {
        bail!("project versions cannot be changed while processing is running");
    }
    Ok(())
}
