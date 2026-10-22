use std::sync::Arc;

use anyhow::{Context as _, anyhow, bail};
use image::{GrayImage, ImageEncoder as _, codecs::png::PngEncoder};
use koharu_desktop::{CanvasState, Desktop, Frame, TransformFrame};
use koharu_rasterizer::ResourceId;
use koharu_scene::{EntityId, Revision};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{
    AppHandle, Manager as _, State,
    ipc::{Channel, IpcResponse},
};
use tauri_runtime_cef::CefRuntime;

use super::{
    ChannelExt as _, Error, processing,
    processing::{JobChannel, JobId, Processing},
    project::{CurrentProject, Page, Project, RasterStrokeMode},
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, Type)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
pub struct PaintBrush {
    pub diameter: f32,
    pub color: [u8; 4],
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct LayerCommit {
    pub revision: Revision,
    pub layer: EntityId,
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct OcrRegionCommit {
    pub revision: Revision,
    pub layer: EntityId,
    pub region: EntityId,
}

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct CanvasBytes(#[specta(type = Vec<u8>)] Vec<u8>);

#[derive(Clone, Copy, Deserialize, Type)]
#[specta(transparent)]
pub(crate) struct CanvasGeneration(#[specta(type = f64)] u64);

#[derive(Clone, Debug, Serialize, Type)]
pub struct CanvasPagePreparation {
    pub revision: Revision,
    pub page: Page,
}

impl IpcResponse for CanvasBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Default)]
pub(crate) struct CanvasChannel {
    pub(crate) channel: Mutex<Option<Channel<CanvasState>>>,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_manifest(
    generation: CanvasGeneration,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    Ok(CanvasBytes(desktop.frame_manifest_bytes(generation.0)?))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_resource(
    generation: CanvasGeneration,
    resource: String,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    let resource = resource
        .parse::<ResourceId>()
        .context("canvas resource id is invalid")?;
    Ok(CanvasBytes(
        desktop.frame_resource_bytes(generation.0, resource)?,
    ))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn prepare_canvas_page(
    page: EntityId,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
) -> Result<Option<CanvasPagePreparation>, Error> {
    let (snapshot, prepared_page) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        let snapshot = project.snapshot();
        let prepared_page = Project::page(&snapshot, page)?;
        (snapshot, prepared_page)
    };
    let revision = snapshot.revision();
    Ok(desktop
        .prepare_page(&snapshot, page)
        .await?
        .then_some(CanvasPagePreparation {
            revision,
            page: prepared_page,
        }))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_page_manifest(
    page: EntityId,
    revision: Revision,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    Ok(CanvasBytes(desktop.page_manifest_bytes(page, revision)?))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_canvas_page_resource(
    page: EntityId,
    revision: Revision,
    resource: String,
    desktop: State<'_, Desktop>,
) -> Result<CanvasBytes, Error> {
    let resource = resource
        .parse::<ResourceId>()
        .context("canvas resource id is invalid")?;
    Ok(CanvasBytes(
        desktop.page_resource_bytes(page, revision, resource)?,
    ))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn add_point_text(
    point: Point,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    let (commit, page, layer) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer) = project.add_point_text(page, point).await?;
        project.record_commit(&commit);
        (commit, project.active_page(), layer)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(LayerCommit {
        revision: commit.revision,
        layer,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn add_text_box(
    frame: Frame,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    let (commit, page, layer) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer) = project.add_text_box(page, frame).await?;
        project.record_commit(&commit);
        (commit, project.active_page(), layer)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(LayerCommit {
        revision: commit.revision,
        layer,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn add_ocr_region(
    frame: Frame,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<OcrRegionCommit, Error> {
    let source_language = koharu_pipeline::PipelineConfig::load()?
        .read()?
        .translation
        .source_language;
    let (commit, page, layer, region) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer, region) = project.add_ocr_region(page, frame, source_language).await?;
        project.record_commit(&commit);
        (commit, project.active_page(), layer, region)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(OcrRegionCommit {
        revision: commit.revision,
        layer,
        region,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_paint(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    brush: PaintBrush,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    commit_raster_stroke(
        expected_revision,
        layer,
        points,
        brush.diameter,
        brush.color,
        RasterStrokeMode::Paint,
        &desktop,
        &project,
        &canvas_channel,
    )
    .await
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_erase(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    diameter: f32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    commit_raster_stroke(
        expected_revision,
        layer,
        points,
        diameter,
        [0; 4],
        RasterStrokeMode::Erase,
        &desktop,
        &project,
        &canvas_channel,
    )
    .await
}

/// The active page as it shows without text, PNG encoded: what the Clone
/// Stamp copies from, for its live preview.
#[tauri::command]
#[specta::specta]
pub(crate) async fn get_clone_source(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> Result<super::output::ImageBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let image = Project::composited_page(&snapshot, page).await?;
    let bytes = tokio_rayon::spawn(move || -> anyhow::Result<Vec<u8>> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image).write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(bytes.into_inner())
    })
    .await?;
    Ok(super::output::ImageBytes(bytes))
}

/// Clone Stamp stroke: paints the page as it shows, read `offset` page units
/// from each stroke point, into `layer` or the page's visible cleanup layer.
#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_clone(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    diameter: f32,
    offset: Point,
    opacity: f32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    if !offset.x.is_finite() || !offset.y.is_finite() || !(0.0..=1.0).contains(&opacity) {
        return Err(anyhow!("the clone offset must be finite and its opacity within 0..1").into());
    }
    commit_raster_stroke(
        expected_revision,
        layer,
        points,
        diameter,
        [0; 4],
        RasterStrokeMode::Clone {
            offset: (offset.x, offset.y),
            opacity,
        },
        &desktop,
        &project,
        &canvas_channel,
    )
    .await
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn import_cleanup_image(
    data_url: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    processing: State<'_, Processing>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<LayerCommit, Error> {
    if !processing.stops.lock().is_empty() {
        return Err(
            anyhow!("cleanup images cannot be imported while processing is running").into(),
        );
    }
    let (commit, page, layer) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, layer) = project.import_cleanup_image(page, data_url).await?;
        project.record_commit(&commit);
        (commit, page, layer)
    };
    desktop
        .synchronize(&commit.snapshot, Some(page), &commit)
        .await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(LayerCommit {
        revision: commit.revision,
        layer,
    })
}

#[allow(clippy::too_many_arguments)]
async fn commit_raster_stroke(
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    diameter: f32,
    color: [u8; 4],
    mode: RasterStrokeMode,
    desktop: &Desktop,
    project: &CurrentProject,
    canvas_channel: &CanvasChannel,
) -> Result<LayerCommit, Error> {
    let (commit, page, element) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        ensure_revision(project.snapshot().revision(), expected_revision)?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let (commit, element) = project
            .apply_raster_stroke(
                page,
                layer,
                mode,
                color,
                diameter,
                points
                    .into_iter()
                    .map(|point| koharu_scene::Point {
                        x: point.x,
                        y: point.y,
                    })
                    .collect(),
            )
            .await?;
        project.record_commit(&commit);
        (commit, project.active_page(), element)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(LayerCommit {
        revision: commit.revision,
        layer: element,
    })
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_transform(
    expected_revision: Revision,
    elements: Vec<TransformFrame>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<Option<Revision>, Error> {
    let geometries = desktop.transform_geometries(expected_revision, &elements)?;
    if geometries.is_empty() {
        return Ok(None);
    }
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        ensure_revision(project.snapshot().revision(), expected_revision)?;
        let commit = project.set_geometries(geometries).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.channel.publish(desktop.canvas_state());
    Ok(Some(commit.revision))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_inpaint(
    expected_revision: Revision,
    points: Vec<Point>,
    diameter: f32,
    handle: AppHandle<CefRuntime>,
    project: State<'_, CurrentProject>,
) -> Result<Option<JobId>, Error> {
    if !diameter.is_finite() || diameter <= 0.0 || points.is_empty() {
        return Err(anyhow!(
            "an inpaint stroke requires a positive diameter and at least one point"
        )
        .into());
    }
    if points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(anyhow!("inpaint stroke points must be finite").into());
    }
    let (page, width, height) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        let snapshot = project.snapshot();
        ensure_revision(snapshot.revision(), expected_revision)?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let page_value = snapshot.page(page)?.page()?;
        (
            page,
            page_value.width.round() as u32,
            page_value.height.round() as u32,
        )
    };
    let (png, bounds) =
        tokio_rayon::spawn(move || encode_mask(width, height, &points, diameter)).await?;
    let mask = koharu_pipeline::InpaintingMask {
        page,
        png: Arc::from(png),
        intent: koharu_pipeline::InpaintingIntent::Remove,
    };
    Ok(Some(
        processing::process_with_mask(
            handle.clone(),
            koharu_pipeline::Scope::Region { page, bounds },
            koharu_pipeline::Operation::Only {
                stage: koharu_pipeline::Stage::Inpainting,
            },
            handle.state::<CurrentProject>(),
            handle.state::<Processing>(),
            handle.state::<JobChannel>(),
            Some(mask),
        )
        .await?,
    ))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn commit_restore(
    expected_revision: Revision,
    frame: Frame,
    handle: AppHandle<CefRuntime>,
    project: State<'_, CurrentProject>,
) -> Result<Option<JobId>, Error> {
    let (page, width, height) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        let snapshot = project.snapshot();
        ensure_revision(snapshot.revision(), expected_revision)?;
        let page = project
            .active_page()
            .context("the project has no active page")?;
        let page_value = snapshot.page(page)?.page()?;
        (
            page,
            page_value.width.round() as u32,
            page_value.height.round() as u32,
        )
    };
    let (png, bounds) = encode_rectangle_mask(width, height, frame)?;
    let mask = koharu_pipeline::InpaintingMask {
        page,
        png: Arc::from(png),
        intent: koharu_pipeline::InpaintingIntent::Restore,
    };
    Ok(Some(
        processing::process_with_mask(
            handle.clone(),
            koharu_pipeline::Scope::Region { page, bounds },
            koharu_pipeline::Operation::Only {
                stage: koharu_pipeline::Stage::Inpainting,
            },
            handle.state::<CurrentProject>(),
            handle.state::<Processing>(),
            handle.state::<JobChannel>(),
            Some(mask),
        )
        .await?,
    ))
}

fn ensure_revision(actual: Revision, expected: Revision) -> anyhow::Result<()> {
    if actual != expected {
        bail!("canvas edit expected revision {expected}, but the project is at {actual}");
    }
    Ok(())
}

fn encode_mask(
    width: u32,
    height: u32,
    points: &[Point],
    diameter: f32,
) -> anyhow::Result<(Vec<u8>, koharu_pipeline::Bounds)> {
    if width == 0 || height == 0 {
        bail!("page dimensions must be positive");
    }
    let mut image = GrayImage::new(width, height);
    let radius = f64::from(diameter) * 0.5;
    let mut dirty = None::<[u32; 4]>;
    for (start, end) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
    {
        let left = (start.x.min(end.x) - radius - 0.5).floor().max(0.0) as u32;
        let top = (start.y.min(end.y) - radius - 0.5).floor().max(0.0) as u32;
        let right = (start.x.max(end.x) + radius + 0.5)
            .ceil()
            .min(f64::from(width)) as u32;
        let bottom = (start.y.max(end.y) + radius + 0.5)
            .ceil()
            .min(f64::from(height)) as u32;
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx.mul_add(dx, dy * dy);
        for y in top..bottom {
            for x in left..right {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let progress = if length_squared <= f64::EPSILON {
                    0.0
                } else {
                    (((px - start.x) * dx + (py - start.y) * dy) / length_squared).clamp(0.0, 1.0)
                };
                let nearest_x = start.x + progress * dx;
                let nearest_y = start.y + progress * dy;
                let distance = (px - nearest_x).hypot(py - nearest_y);
                let coverage = ((radius + 0.5 - distance).clamp(0.0, 1.0) * 255.0).round() as u8;
                if coverage == 0 {
                    continue;
                }
                let pixel = image.get_pixel_mut(x, y);
                pixel.0[0] = pixel.0[0].max(coverage);
                dirty = Some(match dirty {
                    Some([min_x, min_y, max_x, max_y]) => {
                        [min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y)]
                    }
                    None => [x, y, x, y],
                });
            }
        }
    }
    let [left, top, right, bottom] = dirty.context("inpaint stroke does not intersect the page")?;
    let mut png = Vec::new();
    PngEncoder::new(&mut png).write_image(
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::L8,
    )?;
    Ok((
        png,
        koharu_pipeline::Bounds {
            x: f64::from(left),
            y: f64::from(top),
            width: f64::from(right - left + 1),
            height: f64::from(bottom - top + 1),
        },
    ))
}

fn encode_rectangle_mask(
    width: u32,
    height: u32,
    frame: Frame,
) -> anyhow::Result<(Vec<u8>, koharu_pipeline::Bounds)> {
    if width == 0 || height == 0 {
        bail!("page dimensions must be positive");
    }
    if !frame.x.is_finite()
        || !frame.y.is_finite()
        || !frame.width.is_finite()
        || !frame.height.is_finite()
        || !frame.angle_degrees.is_finite()
        || frame.x < 0.0
        || frame.y < 0.0
        || frame.width <= 0.0
        || frame.height <= 0.0
        || frame.angle_degrees != 0.0
        || frame.x + frame.width > width as f32
        || frame.y + frame.height > height as f32
    {
        bail!("restore frame must be an axis-aligned rectangle inside the page");
    }
    let left = frame.x.floor() as u32;
    let top = frame.y.floor() as u32;
    let right = (frame.x + frame.width).ceil() as u32;
    let bottom = (frame.y + frame.height).ceil() as u32;
    let mut image = GrayImage::new(width, height);
    for y in top..bottom {
        for x in left..right {
            image.get_pixel_mut(x, y).0[0] = 255;
        }
    }
    let mut png = Vec::new();
    PngEncoder::new(&mut png).write_image(
        image.as_raw(),
        width,
        height,
        image::ExtendedColorType::L8,
    )?;
    Ok((
        png,
        koharu_pipeline::Bounds {
            x: f64::from(left),
            y: f64::from(top),
            width: f64::from(right - left),
            height: f64::from(bottom - top),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_restore_mask_uses_exact_rounded_page_bounds() {
        let (png, bounds) = encode_rectangle_mask(
            8,
            6,
            Frame {
                x: 2.25,
                y: 1.5,
                width: 3.5,
                height: 2.25,
                angle_degrees: 0.0,
            },
        )
        .unwrap();
        let mask = image::load_from_memory(&png).unwrap().to_luma8();

        assert_eq!(bounds.x, 2.0);
        assert_eq!(bounds.y, 1.0);
        assert_eq!(bounds.width, 4.0);
        assert_eq!(bounds.height, 3.0);
        assert_eq!(mask.pixels().filter(|pixel| pixel[0] == 255).count(), 12);
        assert_eq!(mask.get_pixel(2, 1)[0], 255);
        assert_eq!(mask.get_pixel(5, 3)[0], 255);
        assert_eq!(mask.get_pixel(1, 1)[0], 0);
        assert_eq!(mask.get_pixel(6, 3)[0], 0);
    }

    #[test]
    fn rectangle_restore_mask_rejects_invalid_page_coordinates() {
        for frame in [
            Frame {
                x: -1.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                angle_degrees: 0.0,
            },
            Frame {
                x: 7.5,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                angle_degrees: 0.0,
            },
            Frame {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                angle_degrees: 1.0,
            },
        ] {
            assert!(encode_rectangle_mask(8, 6, frame).is_err());
        }
    }
}
