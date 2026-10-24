//! Typesetting tools for agents: a page's text layers with what the renderer
//! drew for them, crops of the page as it shows, with or without its text or
//! as imported, and trial renders of text, box and typography changes that
//! commit nothing. A fit is a trial with auto-fit in the intended box.

use std::io::Cursor;

use anyhow::{Context as _, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::Desktop;
use koharu_rasterizer::RasterOptions;
use koharu_renderer::{Frame, LayerKind, RenderDiagnostic};
use koharu_scene::{AssetRole, EntityId, Geometry, Origin, Snapshot, Visibility};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::commands::{
    canvas::Point,
    editing::{GeometryUpdate, TypographyUpdate},
    output,
    project::{Project, Typography},
};

use super::host::{AgentFontStyle, AgentTextAlignment, AgentWritingMode, entity};

#[derive(Deserialize, JsonSchema)]
pub(super) struct PageLayers {
    pub(super) page: String,
}

#[derive(Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(super) enum CropView {
    /// The page as it exports, text included.
    #[default]
    Typeset,
    /// The page as it shows without its text layers: the cleaning.
    Clean,
    /// The imported page source, before enhancement and cleaning.
    Source,
}

#[derive(Deserialize, JsonSchema)]
pub(super) struct Crop {
    pub(super) page: String,
    /// [x, y, width, height] in page pixels; the whole page when absent.
    #[serde(default)]
    pub(super) region: Option<[f64; 4]>,
    #[serde(default)]
    pub(super) view: CropView,
    /// Output scale, 0.1 to 4; 1 keeps page pixels.
    #[serde(default)]
    pub(super) scale: Option<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub(super) struct TrialTypography {
    preferred_font: Option<String>,
    font_weight: Option<u16>,
    font_style: Option<AgentFontStyle>,
    size: Option<f32>,
    auto_fit: bool,
    color: Option<[u8; 4]>,
    stroke_color: Option<[u8; 4]>,
    stroke_width: Option<f32>,
    alignment: Option<AgentTextAlignment>,
    writing_mode: Option<AgentWritingMode>,
}

#[derive(Deserialize, JsonSchema)]
pub(super) struct TrialChange {
    element: String,
    /// New translation text.
    #[serde(default)]
    text: Option<String>,
    /// New text box, [x, y, width, height] in page pixels.
    #[serde(default)]
    region: Option<[f64; 4]>,
    /// Whole new typography, as set_typography takes it.
    #[serde(default)]
    typography: Option<TrialTypography>,
}

#[derive(Deserialize, JsonSchema)]
pub(super) struct PreviewLayers {
    page: String,
    changes: Vec<TrialChange>,
    /// Area of the returned image, [x, y, width, height]; around the changed
    /// layers when absent.
    #[serde(default)]
    region: Option<[f64; 4]>,
}

/// The text layers of `page` in their numbered order with what the renderer
/// drew for each.
pub(super) async fn page_layers(
    desktop: &Desktop,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Value> {
    let frame = desktop.renderer().render(snapshot, page).await?;
    let mut layers = Vec::new();
    if let Some(group) = snapshot.page(page)?.text_group()? {
        for (index, layer) in group.text_layers()?.enumerate() {
            layers.push(layer_value(snapshot, &frame, layer.id(), index + 1)?);
        }
    }
    let size = snapshot.page(page)?.page()?;
    Ok(json!({
        "page": page,
        "label": size.label,
        "size": [size.width, size.height],
        "layers": layers,
    }))
}

/// A crop of `page` in the chosen view, as PNG.
pub(super) async fn crop(
    desktop: &Desktop,
    snapshot: &Snapshot,
    arguments: &Crop,
) -> Result<(Value, String)> {
    let page = entity(&arguments.page)?;
    let image = match arguments.view {
        CropView::Typeset => render_image(desktop, snapshot, page).await?,
        CropView::Clean => {
            let hidden = without_text(snapshot, page)?;
            render_image(desktop, &hidden, page).await?
        }
        CropView::Source => {
            let asset = snapshot
                .asset(page, &AssetRole::new("source")?)?
                .context("the page has no source image")?;
            image::load_from_memory(&snapshot.read_blob(asset.blob).await?)?.to_rgba8()
        }
    };
    let (region, png) = cut(&image, arguments.region, arguments.scale.unwrap_or(1.0))?;
    Ok((json!({ "page": page, "region": region }), png))
}

/// Renders `page` with the changes applied to a preview of the project:
/// nothing is committed. Returns what the renderer drew for each changed
/// layer and an image of the area around them.
pub(super) async fn preview_layers(
    desktop: &Desktop,
    snapshot: &Snapshot,
    arguments: PreviewLayers,
) -> Result<(Value, String)> {
    let page = entity(&arguments.page)?;
    let mut trial = snapshot.clone();
    let mut changed = Vec::new();
    for change in arguments.changes {
        let layer = entity(&change.element)?;
        if trial.page(layer).is_ok() || owning_page(&trial, layer)? != page {
            bail!("element {layer} is not a layer of page {page}");
        }
        if let Some(text) = change.text {
            let patch = Project::translation_patch(&trial, layer, Some(text))?;
            trial = trial.preview([&patch])?;
        }
        if let Some([x, y, width, height]) = change.region {
            let points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ]
            .map(|(x, y)| Point { x, y })
            .to_vec();
            let patch = Project::geometry_patch(
                &trial,
                vec![GeometryUpdate {
                    layer,
                    points: Some(points),
                }],
            )?;
            trial = trial.preview([&patch])?;
        }
        if let Some(typography) = change.typography {
            let typography = Typography {
                preferred_font: typography.preferred_font,
                font_weight: typography.font_weight,
                font_style: typography.font_style.map(Into::into),
                size: typography.size,
                auto_fit: typography.auto_fit,
                color: typography.color,
                stroke_color: typography.stroke_color,
                stroke_width: typography.stroke_width,
                alignment: typography.alignment.map(Into::into),
                writing_mode: typography.writing_mode.map(Into::into),
            };
            let patch = Project::typography_patch(
                &trial,
                vec![TypographyUpdate { layer, typography }],
                |_| None,
            )?;
            trial = trial.preview([&patch])?;
        }
        changed.push(layer);
    }
    let frame = desktop.renderer().render(&trial, page).await?;
    let numbers = numbering(&trial, page)?;
    let layers = changed
        .iter()
        .map(|layer| layer_value(&trial, &frame, *layer, numbers(*layer)))
        .collect::<Result<Vec<_>>>()?;
    let region = arguments.region.or_else(|| {
        let boxes = changed
            .iter()
            .filter_map(|layer| rendered_box(&frame, *layer))
            .collect::<Vec<_>>();
        let left = boxes.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
        let top = boxes.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
        let right = boxes
            .iter()
            .map(|b| b[0] + b[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let bottom = boxes
            .iter()
            .map(|b| b[1] + b[3])
            .fold(f64::NEG_INFINITY, f64::max);
        (left < right).then(|| {
            [
                left - 48.0,
                top - 48.0,
                right - left + 96.0,
                bottom - top + 96.0,
            ]
        })
    });
    let image = rasterize(desktop, &frame).await?;
    let (region, png) = cut(&image, region, 1.0)?;
    Ok((
        json!({ "page": page, "region": region, "layers": layers }),
        png,
    ))
}

pub(super) fn image_url(png: String) -> String {
    format!("data:image/png;base64,{png}")
}

fn numbering(snapshot: &Snapshot, page: EntityId) -> Result<impl Fn(EntityId) -> usize> {
    let order = match snapshot.page(page)?.text_group()? {
        Some(group) => group.text_layers()?.map(|layer| layer.id()).collect(),
        None => Vec::new(),
    };
    Ok(move |layer| {
        order
            .iter()
            .position(|id| *id == layer)
            .map_or(0, |index| index + 1)
    })
}

fn owning_page(snapshot: &Snapshot, entity: EntityId) -> Result<EntityId> {
    let mut current = entity;
    while snapshot.page(current).is_err() {
        current = snapshot
            .parent(current)?
            .context("the element is not on a page")?;
    }
    Ok(current)
}

fn layer_value(
    snapshot: &Snapshot,
    frame: &Frame,
    layer: EntityId,
    number: usize,
) -> Result<Value> {
    let text = snapshot.text_layer(layer)?;
    let content = text.content()?;
    let visible = text
        .visibility()?
        .is_none_or(|visibility| visibility.visible && visibility.opacity > 0.0);
    let balloon = text
        .balloon_target()?
        .map(|region| region.geometry())
        .transpose()?;
    let rendered = frame
        .layer(layer)
        .and_then(|rendered| match rendered.kind() {
            LayerKind::Text(metadata) => Some(metadata),
            LayerKind::Image(_) => None,
        });
    let typography = text.typography()?.map(Project::typography_view);
    Ok(json!({
        "id": layer,
        "number": number,
        "visible": visible,
        "role": content.role()?.map(|role| role.role),
        "source_text": content.source()?.map(|source| source.text.value),
        "translated_text": content.translation()?.map(|translation| translation.text.value),
        "box": text.frame()?.as_ref().and_then(extent),
        "balloon": balloon.as_ref().map(|geometry| json!({
            "box": extent(geometry),
            "inner_box": inner_box(geometry),
        })),
        "typography": typography,
        "rendered": rendered.map(|rendered| json!({
            "fonts": rendered.post_script_fonts,
            "font_size": (f64::from(rendered.font_size) * 10.0).round() / 10.0,
            "text_box": rendered_box(frame, layer),
            "warning": warning(frame.diagnostics(), layer),
        })),
    }))
}

fn rendered_box(frame: &Frame, layer: EntityId) -> Option<[f64; 4]> {
    let LayerKind::Text(metadata) = frame.layer(layer)?.kind() else {
        return None;
    };
    let bounds = metadata.rendered_bounds;
    Some([bounds.x, bounds.y, bounds.width, bounds.height].map(|value| f64::from(value).round()))
}

fn warning(diagnostics: &[RenderDiagnostic], layer: EntityId) -> Option<&'static str> {
    diagnostics.iter().find_map(|diagnostic| match diagnostic {
        RenderDiagnostic::TextOverflow { entity, .. } if *entity == layer => Some("overflow"),
        RenderDiagnostic::TextBelowReadableSize { entity, .. } if *entity == layer => {
            Some("below_readable_size")
        }
        RenderDiagnostic::TextLowContrast { entity, .. } if *entity == layer => {
            Some("low_contrast")
        }
        _ => None,
    })
}

fn bounds(geometry: &Geometry) -> Option<(f64, f64, f64, f64)> {
    let mut points = geometry.points.iter();
    let first = points.next()?;
    Some(points.fold(
        (first.x, first.y, first.x, first.y),
        |(left, top, right, bottom), point| {
            (
                left.min(point.x),
                top.min(point.y),
                right.max(point.x),
                bottom.max(point.y),
            )
        },
    ))
}

fn extent(geometry: &Geometry) -> Option<[f64; 4]> {
    let (left, top, right, bottom) = bounds(geometry)?;
    Some([left, top, right - left, bottom - top].map(f64::round))
}

/// The largest axis-aligned rectangle inside a balloon's outline, inset by a
/// twelfth of its shorter side: where lettering can grow without touching the
/// outline. Searched on a 2 px grid.
fn inner_box(geometry: &Geometry) -> Option<[f64; 4]> {
    let points = geometry
        .points
        .iter()
        .map(|point| (point.x, point.y))
        .collect::<Vec<_>>();
    let (left, top, right, bottom) = bounds(geometry)?;
    const STEP: f64 = 2.0;
    let columns = ((right - left) / STEP).ceil() as usize;
    let rows = ((bottom - top) / STEP).ceil() as usize;
    if columns == 0 || rows == 0 || points.len() < 3 {
        return None;
    }
    let inside = |x: f64, y: f64| {
        let mut inside = false;
        let mut previous = points[points.len() - 1];
        for &current in &points {
            if (current.1 > y) != (previous.1 > y)
                && x < (previous.0 - current.0) * (y - current.1) / (previous.1 - current.1)
                    + current.0
            {
                inside = !inside;
            }
            previous = current;
        }
        inside
    };
    // Largest rectangle of inside cells, by histogram over rows.
    let mut heights = vec![0usize; columns];
    let mut best = (0usize, 0usize, 0usize, 0usize, 0usize);
    for row in 0..rows {
        let y = top + (row as f64 + 0.5) * STEP;
        for (column, height) in heights.iter_mut().enumerate() {
            let x = left + (column as f64 + 0.5) * STEP;
            *height = if inside(x, y) { *height + 1 } else { 0 };
        }
        let mut stack: Vec<usize> = Vec::new();
        for column in 0..=columns {
            let height = if column < columns { heights[column] } else { 0 };
            while let Some(&last) = stack.last() {
                if heights[last] <= height {
                    break;
                }
                stack.pop();
                let start = stack.last().map_or(0, |previous| previous + 1);
                let area = heights[last] * (column - start);
                if area > best.0 {
                    best = (
                        area,
                        start,
                        row + 1 - heights[last],
                        column - start,
                        heights[last],
                    );
                }
            }
            stack.push(column);
        }
    }
    let (area, column, row, width, height) = best;
    if area == 0 {
        return None;
    }
    let (width, height) = (width as f64 * STEP, height as f64 * STEP);
    let inset = width.min(height) / 12.0;
    Some(
        [
            left + column as f64 * STEP + inset,
            top + row as f64 * STEP + inset,
            width - 2.0 * inset,
            height - 2.0 * inset,
        ]
        .map(f64::round),
    )
}

/// A preview of the project with the page's text group hidden.
fn without_text(snapshot: &Snapshot, page: EntityId) -> Result<Snapshot> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(snapshot.clone());
    };
    let patch = snapshot.patch(|edit| {
        edit.set(
            group.id(),
            &Visibility {
                origin: Origin::User,
                visible: false,
                opacity: 1.0,
            },
        )
    })?;
    Ok(snapshot.preview([&patch])?)
}

async fn render_image(desktop: &Desktop, snapshot: &Snapshot, page: EntityId) -> Result<RgbaImage> {
    let frame = desktop.renderer().render(snapshot, page).await?;
    rasterize(desktop, &frame).await
}

async fn rasterize(desktop: &Desktop, frame: &Frame) -> Result<RgbaImage> {
    let rasterizer = desktop.rasterizer().await?;
    Ok(
        output::rasterize(rasterizer, frame, RasterOptions::default())
            .await?
            .image,
    )
}

/// The region of `image` (clamped to it) at `scale`, as base64 PNG.
fn cut(image: &RgbaImage, region: Option<[f64; 4]>, scale: f64) -> Result<(Value, String)> {
    let (width, height) = image.dimensions();
    let [x, y, w, h] = region.unwrap_or([0.0, 0.0, f64::from(width), f64::from(height)]);
    let left = x.max(0.0).floor().min(f64::from(width)) as u32;
    let top = y.max(0.0).floor().min(f64::from(height)) as u32;
    let right = (x + w).ceil().clamp(f64::from(left), f64::from(width)) as u32;
    let bottom = (y + h).ceil().clamp(f64::from(top), f64::from(height)) as u32;
    if right <= left || bottom <= top {
        bail!("the region lies outside the page");
    }
    let mut crop =
        image::imageops::crop_imm(image, left, top, right - left, bottom - top).to_image();
    let scale = scale.clamp(0.1, 4.0);
    if (scale - 1.0).abs() > f64::EPSILON {
        crop = image::imageops::resize(
            &crop,
            ((f64::from(crop.width()) * scale).round() as u32).max(1),
            ((f64::from(crop.height()) * scale).round() as u32).max(1),
            image::imageops::FilterType::Lanczos3,
        );
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(crop).write_to(&mut bytes, ImageFormat::Png)?;
    Ok((
        json!([left, top, right - left, bottom - top]),
        STANDARD.encode(bytes.into_inner()),
    ))
}

#[cfg(test)]
mod tests {
    use koharu_scene::Point as ScenePoint;

    use super::*;

    #[test]
    fn the_inner_box_of_a_round_balloon_is_its_inscribed_square_inset() {
        let circle = Geometry {
            origin: Origin::User,
            points: (0..64)
                .map(|step| {
                    let angle = f64::from(step) * std::f64::consts::TAU / 64.0;
                    ScenePoint {
                        x: 200.0 + 100.0 * angle.cos(),
                        y: 200.0 + 100.0 * angle.sin(),
                    }
                })
                .collect(),
        };
        let [x, y, width, height] = inner_box(&circle).unwrap();
        // The inscribed square is 141 px; a twelfth of it goes on each side.
        assert!((115.0..=125.0).contains(&width), "width {width}");
        assert!((115.0..=125.0).contains(&height), "height {height}");
        assert!((x + width / 2.0 - 200.0).abs() <= 3.0 && (y + height / 2.0 - 200.0).abs() <= 3.0);
    }
}
