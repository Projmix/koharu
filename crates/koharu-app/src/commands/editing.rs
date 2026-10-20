use anyhow::Context as _;
use koharu_desktop::{CanvasState, Desktop};
use koharu_scene::EntityId;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::State;

use super::{
    ChannelExt as _, Error,
    canvas::{CanvasChannel, Point},
    processing::Processing,
    project::{
        CharacterInput, CurrentProject, Page, Project, ProjectLibrary, TextClipboardPayload,
        Typography,
    },
};
async fn synchronize_canvas(
    desktop: &Desktop,
    commit: &koharu_scene::Commit,
    page: Option<EntityId>,
) -> anyhow::Result<CanvasState> {
    desktop.synchronize(&commit.snapshot, page, commit).await?;
    Ok(desktop.canvas_state())
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct GeometryUpdate {
    pub layer: EntityId,
    pub points: Option<Vec<Point>>,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct TypographyUpdate {
    pub layer: EntityId,
    pub typography: Typography,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct DuplicatePagesResult {
    pub project: super::project::ProjectInfo,
    pub page: Page,
    pub duplicated: Vec<EntityId>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct TextTransferResult {
    pub project: super::project::ProjectInfo,
    pub page: Page,
    pub layers: Vec<EntityId>,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn rename_page(
    page: EntityId,
    label: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.rename_page(page, label).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_pages(
    pages: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_pages(pages).await?;
        project.record_commit(&commit);
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn move_page(
    page: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_page(page, index as usize).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_source_text(
    layer: EntityId,
    text: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_source_text(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn duplicate_pages(
    pages: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<DuplicatePagesResult, Error> {
    if !processing.stops.lock().is_empty() {
        return Err(
            anyhow::anyhow!("pages cannot be duplicated while processing is running").into(),
        );
    }
    let (commit, result) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let (commit, duplicated) = project.duplicate_pages(pages).await?;
        project.record_commit(&commit);
        let active = duplicated[0];
        let result = DuplicatePagesResult {
            project: project.info(),
            page: Project::page(&commit.snapshot, active)?,
            duplicated,
        };
        (commit, result)
    };
    let canvas = synchronize_canvas(&desktop, &commit, Some(result.page.id)).await?;
    canvas_channel.channel.publish(canvas);
    Ok(result)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn paste_text_blocks(
    target_page: EntityId,
    payload: TextClipboardPayload,
    position: Option<Point>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<TextTransferResult, Error> {
    if !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!("text cannot be pasted while processing is running").into());
    }
    let (commit, result) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let (commit, layers) = project
            .paste_text_blocks(target_page, payload, position)
            .await?;
        project.record_commit(&commit);
        let result = TextTransferResult {
            project: project.info(),
            page: Project::page(&commit.snapshot, target_page)?,
            layers,
        };
        (commit, result)
    };
    let canvas = synchronize_canvas(&desktop, &commit, Some(target_page)).await?;
    canvas_channel.channel.publish(canvas);
    Ok(result)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_text_role(
    layer: EntityId,
    role: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_text_role(layer, role).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_translation(
    layer: EntityId,
    text: Option<String>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_translation(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_speech_markup(
    layer: EntityId,
    kind: koharu_scene::SpeechKind,
    speaker: Option<koharu_scene::CharacterId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_speech_markup(layer, kind, speaker).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_characters(
    characters: Vec<CharacterInput>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_characters(characters).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

/// Add another project's characters to the open project. Returns how many
/// were new.
#[tauri::command]
#[specta::specta]
pub(crate) async fn import_characters(
    source: String,
    desktop: State<'_, Desktop>,
    library: State<'_, ProjectLibrary>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<u32, Error> {
    let (commit, page, added) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        if project.name == source {
            return Err(anyhow::anyhow!("a project cannot import its own characters").into());
        }
        let imported = library.characters(&source).await?;
        let (commit, added) = project.import_characters(imported).await?;
        project.record_commit(&commit);
        (commit, project.active_page(), added)
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(u32::try_from(added).context("too many characters")?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_typography(
    updates: Vec<TypographyUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project
            .set_typography(updates, |layer| desktop.presented_font_size(layer))
            .await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_geometry(
    updates: Vec<GeometryUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_geometry(updates).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn set_visibility(
    layers: Vec<EntityId>,
    visible: Option<bool>,
    opacity: Option<f32>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_visibility(layers, visible, opacity).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_layers(
    layers: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_layers(layers).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn move_layer(
    layer: EntityId,
    parent: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<Page, Error> {
    let (commit, page, view) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_layer(layer, parent, index as usize).await?;
        project.record_commit(&commit);
        let page = project.active_page().context("no active page")?;
        let view = Project::page(&commit.snapshot, page)?;
        (commit, page, view)
    };
    desktop
        .synchronize(&commit.snapshot, Some(page), &commit)
        .await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(view)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn reorder_layers(
    parent: EntityId,
    order: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<Page, Error> {
    let (commit, page, view) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.reorder_layers(parent, order).await?;
        project.record_commit(&commit);
        let page = project.active_page().context("no active page")?;
        let view = Project::page(&commit.snapshot, page)?;
        (commit, page, view)
    };
    desktop
        .synchronize(&commit.snapshot, Some(page), &commit)
        .await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(view)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn undo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.undo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn redo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.redo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.channel.publish(canvas);
    Ok(())
}
