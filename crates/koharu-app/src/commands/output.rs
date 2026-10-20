use anyhow::{Context as _, Result};
use futures::future::try_join_all;
use image::{
    ExtendedColorType, ImageEncoder as _,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_psd::{PsdExportOptions, export_page};
use koharu_rasterizer::{Raster, RasterOptions, Rasterizer};
use koharu_renderer::{Frame, Renderer};
use koharu_scene::{AssetRole, EntityId, Snapshot, Visibility};
use koharu_translator::Language;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::{io::Write as _, sync::Arc};
use tauri::{State, WebviewWindow, ipc::IpcResponse};
use tauri_runtime_cef::CefRuntime;

use super::{Error, project::CurrentProject};
use koharu_desktop::{Desktop, TextFitWarningKind};

const THUMBNAIL_EDGE: u32 = 128;

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct ThumbnailBytes(#[specta(type = Vec<u8>)] Vec<u8>);

impl IpcResponse for ThumbnailBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
    Cbz,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn export(
    window: WebviewWindow<CefRuntime>,
    format: ExportFormat,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<(), Error> {
    let (name, snapshot) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        (project.name.clone(), project.snapshot())
    };
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    if pages.is_empty() {
        return Err(anyhow::anyhow!("there are no pages to export").into());
    }
    let dialog = rfd::AsyncFileDialog::new().set_parent(&window);
    let destination = match format {
        ExportFormat::Png | ExportFormat::Psd => dialog.pick_folder().await,
        ExportFormat::Cbz => {
            dialog
                .add_filter("Comic Book Archive", &["cbz"])
                .set_file_name(format!("{name}.cbz"))
                .save_file()
                .await
        }
    };
    let Some(destination) = destination.map(|destination| destination.path().to_owned()) else {
        return Ok(());
    };
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let frames = try_join_all(pages.iter().map(|&page| renderer.render(&snapshot, page))).await?;
    let (extension, images) = match format {
        ExportFormat::Png | ExportFormat::Cbz => {
            let images = tokio_rayon::spawn(move || {
                frames
                    .par_iter()
                    .map(|frame| -> Result<_> {
                        let image = rasterizer
                            .rasterize(&frame.raster_frame()?, RasterOptions::default())?
                            .image;
                        let mut bytes = Vec::new();
                        PngEncoder::new_with_quality(
                            &mut bytes,
                            CompressionType::Best,
                            FilterType::Adaptive,
                        )
                        .write_image(
                            image.as_raw(),
                            image.width(),
                            image.height(),
                            ExtendedColorType::Rgba8,
                        )?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .await?;
            ("png", images)
        }
        ExportFormat::Psd => {
            let options = PsdExportOptions::default();
            let images = try_join_all(
                frames
                    .iter()
                    .map(|frame| export_page(Arc::clone(&rasterizer), &snapshot, frame, &options)),
            )
            .await?;
            ("psd", images)
        }
    };
    tokio_rayon::spawn(move || -> Result<()> {
        let mut archive = if matches!(format, ExportFormat::Cbz) {
            let directory = destination.parent().context("archive path has no parent")?;
            Some(zip::ZipWriter::new(tempfile::NamedTempFile::new_in(
                directory,
            )?))
        } else {
            None
        };
        let width = pages.len().to_string().len().max(4);
        // Async preparation and indexed encoding both preserve project order.
        for (index, (page_id, bytes)) in pages.into_iter().zip(images).enumerate() {
            let page = snapshot.page(page_id)?.page()?;
            let name = page
                .label
                .trim()
                .trim_end_matches(|character: char| character == '.' || character.is_whitespace());
            let name = name
                .rsplit_once('.')
                .map_or(name, |(stem, _)| stem)
                .replace(['<', '>', ':', '"', '/', '\\', '|', '?', '*'], "_");
            let name = format!(
                "{:0width$}_{}.{extension}",
                index + 1,
                if name.is_empty() { "page" } else { &name }
            );
            if let Some(archive) = &mut archive {
                // PNG data is already compressed.
                let options = zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored);
                archive.start_file(name, options)?;
                archive.write_all(&bytes)?;
            } else {
                std::fs::write(destination.join(name), bytes)?;
            }
        }
        if let Some(archive) = archive {
            archive.finish()?.persist(destination)?;
        }
        Ok(())
    })
    .await?;
    Ok(())
}

/// What the final chapter check flags before export. Export is not blocked;
/// the lists only point at what still needs a look.
#[derive(Clone, Debug, Serialize, Type)]
pub struct ChapterCheck {
    pub texts: Vec<ChapterIssue>,
    pub unboxed_text: Vec<UnboxedTextIssue>,
}

/// A page where scene markup saw lettering outside every text region.
#[derive(Clone, Debug, Serialize, Type)]
pub struct UnboxedTextIssue {
    pub page: EntityId,
    pub count: u32,
}

/// A text block the final chapter check flags.
#[derive(Clone, Debug, Serialize, Type)]
pub struct ChapterIssue {
    pub page: EntityId,
    pub element: EntityId,
    /// The translation, or the source text when there is none, to name the block.
    pub text: String,
    pub translation: Option<TranslationIssue>,
    pub fit: Option<TextFitWarningKind>,
    /// The source text is a local OCR draft no API read has confirmed yet.
    pub ocr_unverified: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TranslationIssue {
    /// Source text was recognized but has no translation.
    Empty,
    /// The translation still holds Japanese characters.
    JapaneseText,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn check_chapter(
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<ChapterCheck, Error> {
    let (source_language, target_language) = {
        let config = koharu_pipeline::PipelineConfig::load()?;
        let config = config.read()?;
        (
            config.translation.source_language,
            config.translation.target_language,
        )
    };
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let fit = desktop.chapter_text_fit_warnings(&snapshot).await?;
    let mut issues = Vec::new();
    for (page, warnings) in fit {
        let Some(group) = snapshot.page(page)?.text_group()? else {
            continue;
        };
        for layer in group.text_layers()? {
            if snapshot
                .component::<Visibility>(layer.id())?
                .is_some_and(|visibility| !visibility.visible || visibility.opacity <= 0.0)
            {
                continue;
            }
            let content = layer.content()?;
            let source = content.source()?.map(|source| source.text.value);
            let translation = content
                .translation()?
                .map(|translation| translation.text.value);
            let text = translation
                .clone()
                .filter(|text| !text.trim().is_empty())
                .or_else(|| source.clone())
                .unwrap_or_default();
            let translation = match translation.as_deref().map(str::trim) {
                None | Some("")
                    if source
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty()) =>
                {
                    Some(TranslationIssue::Empty)
                }
                Some(text) if keeps_japanese(text, source_language, target_language) => {
                    Some(TranslationIssue::JapaneseText)
                }
                _ => None,
            };
            let fit = warnings
                .iter()
                .find(|warning| warning.element == layer.id())
                .map(|warning| warning.kind);
            let ocr_unverified = koharu_pipeline::ocr_unverified(content)?;
            if translation.is_some() || fit.is_some() || ocr_unverified {
                issues.push(ChapterIssue {
                    page,
                    element: layer.id(),
                    text,
                    translation,
                    fit,
                    ocr_unverified,
                });
            }
        }
    }
    let mut unboxed_text = Vec::new();
    for page in snapshot.pages() {
        let count = koharu_pipeline::open_unboxed_text(&snapshot, page.id())?.len();
        if count > 0 {
            unboxed_text.push(UnboxedTextIssue {
                page: page.id(),
                count: u32::try_from(count).unwrap_or(u32::MAX),
            });
        }
    }
    Ok(ChapterCheck {
        texts: issues,
        unboxed_text,
    })
}

/// Kana never belongs in a translation out of Japanese; kanji does only when
/// the target is Chinese. The prolonged sound mark and the middle dot stay
/// allowed because translators use them as a dash and an ellipsis.
fn keeps_japanese(text: &str, source: Language, target: Language) -> bool {
    if source != Language::Japanese || target == Language::Japanese {
        return false;
    }
    let chinese = matches!(
        target,
        Language::ChineseSimplified | Language::ChineseTraditional
    );
    text.chars().any(|character| match character {
        'ー' | '・' => false,
        '\u{3040}'..='\u{30FF}' | '\u{FF66}'..='\u{FF9D}' => true,
        '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' => !chinese,
        _ => false,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_thumbnail(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .with_context(|| format!("page {page} has no source image"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let bytes = tokio_rayon::spawn(move || -> Result<Vec<u8>> {
        let image = image::load_from_memory(&bytes).context("failed to decode source image")?;
        if image.width() == 0 || image.height() == 0 {
            return Err(anyhow::anyhow!("source image is empty"));
        }
        let image = image.thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE).to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok(encoder.encode(80.0).to_vec())
    })
    .await?;
    Ok(ThumbnailBytes(bytes))
}

pub(crate) async fn rendered_preview(
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Vec<u8>> {
    snapshot.page(page)?;
    let frame = renderer.render(snapshot, page).await?;
    let image = rasterize(rasterizer, &frame, RasterOptions::default())
        .await?
        .image;
    tokio_rayon::spawn(move || {
        let image = image::DynamicImage::ImageRgba8(image)
            .resize(1024, 1024, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok::<_, anyhow::Error>(encoder.encode(85.0).to_vec())
    })
    .await
}

async fn rasterize(
    rasterizer: Arc<Rasterizer>,
    frame: &Frame,
    options: RasterOptions,
) -> Result<Raster> {
    let frame = frame.raster_frame()?;
    tokio_rayon::spawn(move || rasterizer.rasterize(&frame, options))
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use koharu_translator::Language;

    use super::keeps_japanese;

    #[test]
    fn japanese_left_in_a_translation_depends_on_the_target_script() {
        let check = |text, target| keeps_japanese(text, Language::Japanese, target);
        assert!(check("Офис 事務所", Language::Russian));
        assert!(check("Ну ぬちゅ", Language::Russian));
        assert!(!check("Да-а-ー… ・・・", Language::Russian));
        assert!(!check("事務所", Language::ChineseSimplified));
        assert!(check("事務所です", Language::ChineseSimplified));
    }
}
