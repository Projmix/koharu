use std::io::{Cursor, Write};

use anyhow::{Context as _, Result};
use futures::future::try_join_all;
use koharu_desktop::Desktop;
use koharu_pipeline::{ChapterTranslation, PipelineConfig, Typeset};
use koharu_rasterizer::RasterOptions;
use koharu_renderer::TypesettingConfig;
use koharu_scene::Snapshot;
use rayon::prelude::*;
use serde_json::Value;
use tauri::{AppHandle, Manager as _, State};
use tauri_runtime_cef::CefRuntime;

use super::{
    ChannelExt as _, Error, canvas::CanvasChannel, processing::Processing, project::CurrentProject,
};

fn archive(prompt: &str, document: &Value, images: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    zip.start_file("prompt.txt", options)?;
    zip.write_all(prompt.as_bytes())?;
    zip.start_file("chapter.json", options)?;
    zip.write_all(&serde_json::to_vec_pretty(document)?)?;
    // JPEG is already compressed.
    let stored = options.compression_method(zip::CompressionMethod::Stored);
    for (path, bytes) in images {
        zip.start_file(path.as_str(), stored)?;
        zip.write_all(bytes)?;
    }
    Ok(zip.finish()?.into_inner())
}

/// Every page drawn as the export draws it, for proofreading and typesetting.
async fn typeset(desktop: &Desktop, snapshot: &Snapshot) -> Result<Typeset> {
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let frames = try_join_all(
        snapshot
            .pages()
            .map(|page| renderer.render(snapshot, page.id())),
    )
    .await?;
    let pages = tokio_rayon::spawn(move || {
        frames
            .into_par_iter()
            .map(|frame| -> Result<_> {
                let image = rasterizer
                    .rasterize(&frame.raster_frame()?, RasterOptions::default())?
                    .image;
                let image = image::DynamicImage::ImageRgba8(image).to_rgb8();
                Ok((frame.page(), (frame, image)))
            })
            .collect::<Result<_>>()
    })
    .await?;
    Ok(Typeset {
        fonts: TypesettingConfig::load()?.read()?.clone(),
        pages,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn export_chapter_translation(
    window: tauri::WebviewWindow<CefRuntime>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<bool, Error> {
    if !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!(
            "chapter translation export is unavailable while processing is running"
        )
        .into());
    }
    let snapshot = {
        let project = project.project.lock().await;
        project.as_ref().context("no project is open")?.snapshot()
    };
    let config = PipelineConfig::load()?.read()?.clone();
    let typeset = typeset(&desktop, &snapshot).await?;
    let export = ChapterTranslation::from_snapshot(snapshot, &config.translation)?
        .manual_export(typeset)
        .await?;
    let bytes = archive(&export.prompt, &export.document, &export.images)?;
    let Some(file) = rfd::AsyncFileDialog::new()
        .add_filter("ZIP — prompt.txt + chapter.json + pages", &["zip"])
        .set_file_name("koharu-translation.zip")
        .set_parent(&window)
        .save_file()
        .await
    else {
        return Ok(false);
    };
    tokio::fs::write(file.path(), bytes).await?;
    Ok(true)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn import_chapter_translation(
    window: tauri::WebviewWindow<CefRuntime>,
    handle: AppHandle<CefRuntime>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
) -> std::result::Result<bool, Error> {
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
        return Ok(false);
    };
    let response = tokio::fs::read_to_string(file.path()).await?;
    let translator = handle
        .state::<koharu_pipeline::Pipeline>()
        .translator()
        .clone();
    let (commit, page) = {
        let mut projects = project.project.lock().await;
        if !processing.stops.lock().is_empty() {
            return Err(anyhow::anyhow!(
                "chapter translation import is unavailable while processing is running"
            )
            .into());
        }
        let project = projects.as_mut().context("no project is open")?;
        // The file dialog can outlive pending edits. Validate the current chapter
        // and language while holding the same lock used to commit the import.
        let config = PipelineConfig::load()?.read()?.clone();
        let chapter = ChapterTranslation::from_snapshot(project.snapshot(), &config.translation)?;
        let patch = chapter
            .patch_from_response(&translator, response.trim_start_matches('\u{feff}'))
            .await?;
        let commit = project
            .commit_rebased(patch)
            .await?
            .context("the project changed while importing; export a fresh chapter")?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let desktop = handle.state::<Desktop>();
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    handle
        .state::<CanvasChannel>()
        .channel
        .publish(desktop.canvas_state());
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn manual_translation_archive_contains_prompt_json_and_pages() {
        let document = serde_json::json!({"project_id": "project", "pages": []});
        let images = [("pages/001.jpg".to_owned(), vec![1, 2, 3])];
        let bytes = archive("Translate chapter.json", &document, &images).unwrap();
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        assert_eq!(zip.len(), 3);
        let mut page = Vec::new();
        zip.by_name("pages/001.jpg")
            .unwrap()
            .read_to_end(&mut page)
            .unwrap();
        assert_eq!(page, [1, 2, 3]);
        let mut prompt = String::new();
        zip.by_name("prompt.txt")
            .unwrap()
            .read_to_string(&mut prompt)
            .unwrap();
        assert_eq!(prompt, "Translate chapter.json");
        let json: Value = serde_json::from_reader(zip.by_name("chapter.json").unwrap()).unwrap();
        assert_eq!(json, document);
    }
}
