//! Typesetting tools for agents: a page's text layers with what the renderer
//! drew for them, line by line, crops of the page as it shows, with or
//! without its text or as imported, trial renders of text, box and typography
//! changes that commit nothing, and the chapter's text and check with page
//! and layer numbers. A fit is a trial with auto-fit in the intended box.

use std::io::Cursor;

use anyhow::{Context as _, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::{Desktop, Frame as TextFrame};
use koharu_rasterizer::RasterOptions;
use koharu_renderer::{Frame, LayerKind, RenderDiagnostic, RenderedLine};
use koharu_scene::{
    AssetRole, EntityId, Geometry, Origin, Snapshot, TextLayout as SceneTextLayout, Visibility,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::commands::{
    canvas::Point,
    editing::{GeometryUpdate, TypographyUpdate},
    output::{self, ChapterCheck},
    project::{EditStep, Project, Typography},
};

use super::host::{
    AgentFontStyle, AgentLineBreaking, AgentTextAlignment, AgentWritingMode, entity,
};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Put the imported source on the left of the same region.
    #[serde(default)]
    pub(super) compare: bool,
    /// Draw every text layer's box with its number, and the measured inside
    /// of its balloon.
    #[serde(default)]
    pub(super) outline: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
    /// Where lines may break; automatic (hyphenating a word as a last
    /// resort) when absent.
    #[serde(default)]
    line_breaking: Option<AgentLineBreaking>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct TrialChange {
    element: String,
    /// New translation text.
    #[serde(default)]
    text: Option<String>,
    /// New translation as its lines, instead of text with line breaks.
    #[serde(default)]
    lines: Option<Vec<String>>,
    /// New source text. Writing it confirms an unverified OCR reading.
    #[serde(default)]
    source: Option<String>,
    /// New text box, [x, y, width, height] in page pixels.
    #[serde(default)]
    region: Option<[f64; 4]>,
    /// Turn the new box about its centre by this many degrees, clockwise;
    /// 180 sets text upside down. Needs region.
    #[serde(default)]
    angle: Option<f32>,
    /// Whole new typography, as set_typography takes it.
    #[serde(default)]
    typography: Option<TrialTypography>,
    /// Show or hide the layer.
    #[serde(default)]
    visible: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ApplyLayers {
    pub(super) changes: Vec<TrialChange>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct PreviewLayers {
    page: String,
    changes: Vec<TrialChange>,
    /// Area of the returned image, [x, y, width, height]; around the changed
    /// layers when absent.
    #[serde(default)]
    region: Option<[f64; 4]>,
    /// Output scale, 0.1 to 4; 1 keeps page pixels.
    #[serde(default)]
    scale: Option<f64>,
    /// Put the imported source on the left of the same region.
    #[serde(default)]
    compare: bool,
    /// Draw the changed layers' boxes with their numbers, and the measured
    /// inside of their balloons.
    #[serde(default)]
    outline: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ChapterText {
    /// First and last page number to list, from 1; every page when absent.
    #[serde(default)]
    pages: Option<[usize; 2]>,
}

/// The text layers of `page` in their numbered order with what the renderer
/// drew for each.
pub(super) async fn page_layers(
    desktop: &Desktop,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Value> {
    let frame = desktop.renderer().render(snapshot, page).await?;
    // Balloons are measured on the page as it shows without its text.
    let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
    let mut layers = Vec::new();
    if let Some(group) = snapshot.page(page)?.text_group()? {
        for (index, layer) in group.text_layers()?.enumerate() {
            let mut value = layer_value(snapshot, &frame, layer.id(), index + 1)?;
            measure_balloon(&clean, &mut value);
            layers.push(value);
        }
    }
    let size = snapshot.page(page)?.page()?;
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    let position = pages.iter().position(|candidate| *candidate == page);
    let neighbour = |offset: isize, last: bool| -> Result<Value> {
        let Some(other) = position
            .and_then(|position| position.checked_add_signed(offset))
            .and_then(|position| pages.get(position))
        else {
            return Ok(Value::Null);
        };
        let texts = page_texts(snapshot, *other)?
            .into_iter()
            .filter(|text| text["shows"] == json!(true))
            .collect::<Vec<_>>();
        let keep = if last {
            texts[texts.len().saturating_sub(2)..].to_vec()
        } else {
            texts.into_iter().take(2).collect()
        };
        Ok(Value::Array(keep))
    };
    Ok(json!({
        "page": page,
        "number": position.map(|position| position + 1),
        "label": size.label,
        "size": [size.width, size.height],
        "layers": layers,
        "previous_page_end": neighbour(-1, true)?,
        "next_page_start": neighbour(1, false)?,
    }))
}

fn read_box(value: &Value) -> Option<[f64; 4]> {
    let values = value
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()?;
    values.try_into().ok()
}

/// Replaces a layer value's outline estimate of the balloon's inside with
/// the one measured on the clean page, when the balloon is closed.
fn measure_balloon(clean: &RgbaImage, value: &mut Value) {
    let measured = read_box(&value["balloon"]["box"])
        .zip(read_box(&value["box"]))
        .and_then(|(balloon, text)| pixel_inner_box(clean, balloon, text));
    match measured {
        Some(measured) => {
            value["balloon"]["inner_box"] = json!(measured);
            value["balloon"]["measured"] = json!("pixels");
        }
        None if value["balloon"].is_object() => {
            value["balloon"]["measured"] = json!("outline");
        }
        None => {}
    }
}

/// The chapter's pages in order with the text of each, for `chapter_text`.
pub(super) fn chapter_text(snapshot: &Snapshot, arguments: &ChapterText) -> Result<Value> {
    let [first, last] = arguments.pages.unwrap_or([1, usize::MAX]);
    let mut pages = Vec::new();
    for (index, page) in snapshot.pages().enumerate() {
        let number = index + 1;
        if number < first || number > last {
            continue;
        }
        let label = page.page()?.label;
        pages.push(json!({
            "number": number,
            "id": page.id(),
            "label": label,
            "layers": page_texts(snapshot, page.id())?,
        }));
    }
    Ok(json!({ "pages": pages }))
}

/// A page's text layers in their numbered order, without geometry.
fn page_texts(snapshot: &Snapshot, page: EntityId) -> Result<Vec<Value>> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    let group_shown = shown(snapshot, group.id())?;
    group
        .text_layers()?
        .enumerate()
        .map(|(index, layer)| {
            let content = layer.content()?;
            Ok(json!({
                "number": index + 1,
                "id": layer.id(),
                "role": content.role()?.map(|role| role.role),
                "shows": group_shown && shown(snapshot, layer.id())?,
                "source": content.source()?.map(|source| source.text.value),
                "translation": content.translation()?.map(|translation| translation.text.value),
            }))
        })
        .collect()
}

fn shown(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
    Ok(snapshot
        .component::<Visibility>(entity)?
        .is_none_or(|visibility| visibility.visible && visibility.opacity > 0.0))
}

/// The chapter check with page and layer numbers, both texts, and what hides
/// a hidden block, for agents.
pub(super) fn chapter_issues(snapshot: &Snapshot, check: ChapterCheck) -> Result<Value> {
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    let page_value = |page: EntityId| -> Result<Value> {
        Ok(json!({
            "id": page,
            "number": pages.iter().position(|candidate| *candidate == page).map(|index| index + 1),
            "label": snapshot.page(page)?.page()?.label,
        }))
    };
    let mut texts = Vec::new();
    for issue in check.texts {
        let numbers = numbering(snapshot, issue.page)?;
        let content = snapshot.text_layer(issue.element)?.content()?;
        let hidden_by = if !issue.hidden {
            Value::Null
        } else if !shown(snapshot, issue.element)? {
            json!({ "layer": issue.element })
        } else {
            json!({ "text_group": snapshot.page(issue.page)?.text_group()?.map(|group| group.id()) })
        };
        texts.push(json!({
            "page": page_value(issue.page)?,
            "layer": numbers(issue.element),
            "element": issue.element,
            "source": content.source()?.map(|source| source.text.value),
            "translation": content.translation()?.map(|translation| translation.text.value),
            "translation_issue": issue.translation,
            "fit": issue.fit,
            "ocr_unverified": issue.ocr_unverified,
            "cleaning_unverified": issue.cleaning_unverified,
            "hidden": issue.hidden,
            "hidden_by": hidden_by,
        }));
    }
    let unboxed_text = check
        .unboxed_text
        .into_iter()
        .map(|issue| Ok(json!({ "page": page_value(issue.page)?, "count": issue.count })))
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({ "texts": texts, "unboxed_text": unboxed_text }))
}

/// A crop of `page` in the chosen view, as PNG.
pub(super) async fn crop(
    desktop: &Desktop,
    snapshot: &Snapshot,
    arguments: &Crop,
) -> Result<(Value, String)> {
    let page = entity(&arguments.page)?;
    let mut image = match arguments.view {
        CropView::Typeset => render_image(desktop, snapshot, page).await?,
        CropView::Clean => {
            let hidden = without_text(snapshot, page)?;
            render_image(desktop, &hidden, page).await?
        }
        CropView::Source => source_image(snapshot, page).await?,
    };
    let scale = arguments.scale.unwrap_or(1.0).clamp(0.1, 4.0);
    if arguments.outline {
        let layers = match snapshot.page(page)?.text_group()? {
            Some(group) => group.text_layers()?.map(|layer| layer.id()).collect(),
            None => Vec::new(),
        };
        let frame = desktop.renderer().render(snapshot, page).await?;
        let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
        draw_outlines(&mut image, snapshot, &frame, &clean, page, &layers, scale)?;
    }
    let source = if arguments.compare {
        Some(source_image(snapshot, page).await?)
    } else {
        None
    };
    let (region, png) = picture(&image, source.as_ref(), arguments.region, scale)?;
    Ok((
        json!({
            "page": page,
            "region": region,
            "scale": scale,
            "layout": if arguments.compare { "source | view" } else { "view" },
        }),
        png,
    ))
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
    let changed = elements(&arguments.changes)?;
    for layer in &changed {
        if snapshot.page(*layer).is_ok() || owning_page(snapshot, *layer)? != page {
            bail!("element {layer} is not a layer of page {page}");
        }
    }
    let steps = steps(snapshot, arguments.changes, |_| None)?;
    let patch = snapshot.patch(|edit| {
        for step in steps {
            step(edit)?;
        }
        Ok(())
    })?;
    let trial = snapshot.preview([&patch])?;
    let frame = desktop.renderer().render(&trial, page).await?;
    let numbers = numbering(&trial, page)?;
    let layers = changed
        .iter()
        .map(|layer| layer_value(&trial, &frame, *layer, numbers(*layer)))
        .collect::<Result<Vec<_>>>()?;
    let region = arguments.region.or_else(|| {
        // The layers' boxes as well as what was drawn, so text that shrank
        // still shows in its box.
        let boxes = changed
            .iter()
            .filter_map(|layer| rendered_box(&frame, *layer))
            .chain(layers.iter().filter_map(|value| read_box(&value["box"])))
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
    let mut layers = layers;
    let scale = arguments.scale.unwrap_or(1.0).clamp(0.1, 4.0);
    let mut image = rasterize(desktop, &frame).await?;
    if arguments.outline {
        let clean = render_image(desktop, &without_text(&trial, page)?, page).await?;
        for value in &mut layers {
            measure_balloon(&clean, value);
        }
        draw_outlines(&mut image, &trial, &frame, &clean, page, &changed, scale)?;
    }
    let source = if arguments.compare {
        Some(source_image(snapshot, page).await?)
    } else {
        None
    };
    let (region, png) = picture(&image, source.as_ref(), region, scale)?;
    Ok((
        json!({
            "page": page,
            "region": region,
            "scale": scale,
            "layout": if arguments.compare { "source | preview" } else { "preview" },
            "layers": layers,
        }),
        png,
    ))
}

/// The elements the changes name, in order.
pub(super) fn elements(changes: &[TrialChange]) -> Result<Vec<EntityId>> {
    changes
        .iter()
        .map(|change| entity(&change.element))
        .collect()
}

/// The changes as checked edit steps against `snapshot`: the same steps the
/// trial render previews and `apply_layers` commits as one edit.
pub(super) fn steps(
    snapshot: &Snapshot,
    changes: Vec<TrialChange>,
    laid_out_size: impl Fn(EntityId) -> Option<f32>,
) -> Result<Vec<EditStep>> {
    let mut steps = Vec::new();
    for change in changes {
        let layer = entity(&change.element)?;
        let text = match (change.text, change.lines) {
            (Some(_), Some(_)) => bail!("give {layer} either text or lines, not both"),
            (text, lines) => text.or_else(|| lines.map(|lines| lines.join("\n"))),
        };
        if let Some(text) = text {
            steps.push(Project::translation_step(snapshot, layer, Some(text))?);
        }
        if let Some(source) = change.source {
            steps.push(Project::source_step(snapshot, layer, source)?);
        }
        if let Some(angle) = change.angle {
            let [x, y, width, height] = change
                .region
                .with_context(|| format!("turning {layer} needs its region"))?;
            let geometry = Project::geometry_from_frame(TextFrame {
                x: x as f32,
                y: y as f32,
                width: width as f32,
                height: height as f32,
                angle_degrees: angle,
            })?;
            let mut layout = snapshot
                .component::<SceneTextLayout>(layer)?
                .context("only text layers can turn")?;
            layout.origin = Origin::User;
            layout.angle_degrees = Some(angle);
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(layer)?;
                edit.set(layer, &geometry)?;
                edit.set(layer, &layout)
            }));
        } else if let Some([x, y, width, height]) = change.region {
            let points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ]
            .map(|(x, y)| Point { x, y })
            .to_vec();
            steps.push(Project::geometry_step(
                snapshot,
                vec![GeometryUpdate {
                    layer,
                    points: Some(points),
                }],
            )?);
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
                line_breaking: typography.line_breaking.map(Into::into),
            };
            steps.push(Project::typography_step(
                snapshot,
                vec![TypographyUpdate { layer, typography }],
                &laid_out_size,
            )?);
        }
        if let Some(visible) = change.visible {
            let opacity = snapshot
                .component::<Visibility>(layer)?
                .map_or(1.0, |visibility| visibility.opacity);
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(layer)?;
                edit.set(
                    layer,
                    &Visibility {
                        origin: Origin::User,
                        visible,
                        opacity,
                    },
                )
            }));
        }
    }
    Ok(steps)
}

/// What the renderer drew for each of `layers`, page by page, once committed.
pub(super) async fn report(
    desktop: &Desktop,
    snapshot: &Snapshot,
    layers: &[EntityId],
) -> Result<Value> {
    let mut pages = Vec::<(EntityId, Vec<EntityId>)>::new();
    for layer in layers {
        let page = owning_page(snapshot, *layer)?;
        match pages.iter_mut().find(|(candidate, _)| *candidate == page) {
            Some((_, members)) if !members.contains(layer) => members.push(*layer),
            Some(_) => {}
            None => pages.push((page, vec![*layer])),
        }
    }
    let mut values = Vec::new();
    for (page, members) in pages {
        let frame = desktop.renderer().render(snapshot, page).await?;
        let numbers = numbering(snapshot, page)?;
        for layer in members {
            let mut value = layer_value(snapshot, &frame, layer, numbers(layer))?;
            value["page"] = json!(page);
            values.push(value);
        }
    }
    Ok(Value::Array(values))
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
        "rendered": rendered.map(|rendered| {
            let bounds = rendered.layout_bounds;
            json!({
                "fonts": rendered.post_script_fonts,
                "font_size": (f64::from(rendered.font_size) * 10.0).round() / 10.0,
                "text_box": rendered_box(frame, layer),
                "layout_box": ([bounds.x, bounds.y, bounds.width, bounds.height]
                    .map(|value| f64::from(value).round())),
                "lines": rendered.lines.iter().map(|line| json!({
                    "text": line.text,
                    "width": f64::from(line.width).round(),
                    "room": f64::from(line.available).round(),
                })).collect::<Vec<_>>(),
                "hyphenated": hyphenated_words(&rendered.lines),
                "warning": warning(frame.diagnostics(), layer),
                "overflow": overflow(frame.diagnostics(), layer),
            })
        }),
    }))
}

/// The words the renderer split across lines, rejoined: «ВЗРОС-ЛЫХ».
fn hyphenated_words(lines: &[RenderedLine]) -> Vec<String> {
    lines
        .windows(2)
        .filter(|pair| pair[0].hyphenated)
        .map(|pair| {
            let head = pair[0]
                .text
                .split_whitespace()
                .next_back()
                .unwrap_or_default();
            let tail = pair[1].text.split_whitespace().next().unwrap_or_default();
            format!("{head}{tail}")
        })
        .collect()
}

/// What overflowed: the room the layout had and the size it drew.
fn overflow(diagnostics: &[RenderDiagnostic], layer: EntityId) -> Option<Value> {
    diagnostics.iter().find_map(|diagnostic| match diagnostic {
        RenderDiagnostic::TextOverflow {
            entity,
            available,
            actual_width,
            actual_height,
            ..
        } if *entity == layer => Some(json!({
            "room": ([available.width, available.height].map(|value| f64::from(value).round())),
            "drawn": ([*actual_width, *actual_height].map(|value| f64::from(value).round())),
        })),
        _ => None,
    })
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

/// The largest axis-aligned rectangle inside the balloon as drawn on the
/// clean page: the light area flooded from the light pixel nearest the text
/// box's centre, its holes filled, kept a margin from the outline. `None`
/// when the flood leaks out of the balloon's box (an open balloon) or finds
/// no light pixel, so the detected outline decides instead.
fn pixel_inner_box(clean: &RgbaImage, balloon: [f64; 4], text: [f64; 4]) -> Option<[f64; 4]> {
    const LIGHT: u32 = 200;
    let (page_width, page_height) = clean.dimensions();
    let grow = |value: f64| value * 0.15;
    let left = (balloon[0] - grow(balloon[2])).max(0.0).floor() as u32;
    let top = (balloon[1] - grow(balloon[3])).max(0.0).floor() as u32;
    let right = (balloon[0] + balloon[2] * 1.15)
        .min(f64::from(page_width))
        .ceil() as u32;
    let bottom = (balloon[1] + balloon[3] * 1.15)
        .min(f64::from(page_height))
        .ceil() as u32;
    if right <= left + 4 || bottom <= top + 4 {
        return None;
    }
    let (width, height) = ((right - left) as usize, (bottom - top) as usize);
    let light = |x: usize, y: usize| {
        let pixel = clean.get_pixel(left + x as u32, top + y as u32);
        (u32::from(pixel[0]) * 299 + u32::from(pixel[1]) * 587 + u32::from(pixel[2]) * 114) / 1000
            >= LIGHT
    };
    // The light pixel nearest the text box's centre.
    let centre = (
        (text[0] + text[2] / 2.0 - f64::from(left)).clamp(0.0, width as f64 - 1.0) as usize,
        (text[1] + text[3] / 2.0 - f64::from(top)).clamp(0.0, height as f64 - 1.0) as usize,
    );
    let reach = (text[2].min(text[3]) / 2.0).max(4.0) as usize;
    let seed = (0..=reach).find_map(|radius| {
        let (x0, y0) = (
            centre.0.saturating_sub(radius),
            centre.1.saturating_sub(radius),
        );
        let (x1, y1) = (
            (centre.0 + radius).min(width - 1),
            (centre.1 + radius).min(height - 1),
        );
        (y0..=y1)
            .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
            .find(|&(x, y)| light(x, y))
    })?;
    let mut region = vec![false; width * height];
    let mut stack = vec![seed];
    region[seed.1 * width + seed.0] = true;
    let mut leaked = false;
    while let Some((x, y)) = stack.pop() {
        if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
            leaked = true;
        }
        let neighbours = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbours {
            if nx < width && ny < height && !region[ny * width + nx] && light(nx, ny) {
                region[ny * width + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    if leaked {
        return None;
    }
    // Fill holes (specks and leftover lettering): whatever the outside
    // cannot reach through non-region pixels belongs to the balloon.
    let mut outside = vec![false; width * height];
    let mut stack = Vec::new();
    for x in 0..width {
        for y in [0, height - 1] {
            if !region[y * width + x] && !outside[y * width + x] {
                outside[y * width + x] = true;
                stack.push((x, y));
            }
        }
    }
    for y in 0..height {
        for x in [0, width - 1] {
            if !region[y * width + x] && !outside[y * width + x] {
                outside[y * width + x] = true;
                stack.push((x, y));
            }
        }
    }
    while let Some((x, y)) = stack.pop() {
        let neighbours = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbours {
            if nx < width && ny < height && !region[ny * width + nx] && !outside[ny * width + nx] {
                outside[ny * width + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    // Keep a margin: a pixel stays when no outside pixel lies within it.
    let (min_x, max_x, min_y, max_y) = (0..width * height).filter(|at| !outside[*at]).fold(
        (width, 0, height, 0),
        |(a, b, c, d), at| {
            let (x, y) = (at % width, at / width);
            (a.min(x), b.max(x), c.min(y), d.max(y))
        },
    );
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    let margin = ((max_x - min_x).min(max_y - min_y) / 14).max(4);
    let stride = width + 1;
    let mut table = vec![0u32; stride * (height + 1)];
    for y in 0..height {
        let mut row = 0;
        for x in 0..width {
            row += u32::from(outside[y * width + x]);
            table[(y + 1) * stride + x + 1] = table[y * stride + x + 1] + row;
        }
    }
    let clear = |x: usize, y: usize| {
        let (x0, y0) = (x.saturating_sub(margin), y.saturating_sub(margin));
        let (x1, y1) = ((x + margin + 1).min(width), (y + margin + 1).min(height));
        table[y1 * stride + x1] + table[y0 * stride + x0]
            == table[y0 * stride + x1] + table[y1 * stride + x0]
            && x >= margin
            && y >= margin
            && x + margin < width
            && y + margin < height
    };
    let mut heights = vec![0usize; width];
    let mut best = (0usize, 0usize, 0usize, 0usize, 0usize);
    for y in 0..height {
        for (x, column) in heights.iter_mut().enumerate() {
            *column = if clear(x, y) { *column + 1 } else { 0 };
        }
        let mut stack: Vec<usize> = Vec::new();
        for x in 0..=width {
            let current = if x < width { heights[x] } else { 0 };
            while let Some(&last) = stack.last() {
                if heights[last] <= current {
                    break;
                }
                stack.pop();
                let start = stack.last().map_or(0, |previous| previous + 1);
                let area = heights[last] * (x - start);
                if area > best.0 {
                    best = (area, start, y + 1 - heights[last], x - start, heights[last]);
                }
            }
            stack.push(x);
        }
    }
    let (area, x, y, rect_width, rect_height) = best;
    (area > 0).then(|| {
        [
            f64::from(left) + x as f64,
            f64::from(top) + y as f64,
            rect_width as f64,
            rect_height as f64,
        ]
    })
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

async fn source_image(snapshot: &Snapshot, page: EntityId) -> Result<RgbaImage> {
    let asset = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .context("the page has no source image")?;
    Ok(image::load_from_memory(&snapshot.read_blob(asset.blob).await?)?.to_rgba8())
}

/// The region of `image` at `scale` as base64 PNG, with the same region of
/// `source` on its left when given.
fn picture(
    image: &RgbaImage,
    source: Option<&RgbaImage>,
    region: Option<[f64; 4]>,
    scale: f64,
) -> Result<(Value, String)> {
    const GAP: u32 = 8;
    let (region, view) = cut(image, region, scale)?;
    let picture = match source {
        Some(source) => {
            let (_, source) = cut(source, read_box(&region), scale)?;
            let mut joined = RgbaImage::from_pixel(
                source.width() + GAP + view.width(),
                source.height().max(view.height()),
                image::Rgba([128, 128, 128, 255]),
            );
            image::imageops::replace(&mut joined, &source, 0, 0);
            image::imageops::replace(&mut joined, &view, i64::from(source.width() + GAP), 0);
            joined
        }
        None => view,
    };
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(picture).write_to(&mut bytes, ImageFormat::Png)?;
    Ok((region, STANDARD.encode(bytes.into_inner())))
}

/// The region of `image` (clamped to it) at `scale`.
fn cut(image: &RgbaImage, region: Option<[f64; 4]>, scale: f64) -> Result<(Value, RgbaImage)> {
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
    Ok((json!([left, top, right - left, bottom - top]), crop))
}

const LAYER_COLOR: [u8; 4] = [230, 0, 160, 255];
const BALLOON_COLOR: [u8; 4] = [0, 170, 230, 255];

/// Draws each of `layers`' box in magenta with its number, and the inside of
/// its balloon as measured for `page_layers` in blue. Strokes and digits are
/// sized to stay legible at `scale`.
fn draw_outlines(
    image: &mut RgbaImage,
    snapshot: &Snapshot,
    frame: &Frame,
    clean: &RgbaImage,
    page: EntityId,
    layers: &[EntityId],
    scale: f64,
) -> Result<()> {
    let numbers = numbering(snapshot, page)?;
    let stroke = (2.0 / scale).ceil().max(1.0) as u32;
    let cell = (3.0 / scale).ceil().max(2.0) as u32;
    for layer in layers {
        let number = numbers(*layer);
        let mut value = layer_value(snapshot, frame, *layer, number)?;
        measure_balloon(clean, &mut value);
        if let Some(inner) = read_box(&value["balloon"]["inner_box"]) {
            draw_box(image, inner, stroke, BALLOON_COLOR);
        }
        if let Some(region) = read_box(&value["box"]) {
            draw_box(image, region, stroke, LAYER_COLOR);
            draw_number(image, region, number, cell);
        }
    }
    Ok(())
}

fn draw_box(image: &mut RgbaImage, [x, y, width, height]: [f64; 4], stroke: u32, color: [u8; 4]) {
    let (image_width, image_height) = image.dimensions();
    if image_width == 0 || image_height == 0 {
        return;
    }
    let left = (x.max(0.0) as u32).min(image_width - 1);
    let top = (y.max(0.0) as u32).min(image_height - 1);
    let right = ((x + width).max(0.0) as u32).min(image_width - 1);
    let bottom = ((y + height).max(0.0) as u32).min(image_height - 1);
    for py in top..=bottom {
        for px in left..=right {
            let edge = px < left + stroke
                || px + stroke > right
                || py < top + stroke
                || py + stroke > bottom;
            if edge {
                image.put_pixel(px, py, image::Rgba(color));
            }
        }
    }
}

/// Glyphs of the digits 0-9, three columns by five rows, top row first.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// White digits on a magenta tag at the box's top left corner.
fn draw_number(image: &mut RgbaImage, [x, y, _, _]: [f64; 4], number: usize, cell: u32) {
    let digits = number.to_string();
    let tag_width = cell * (4 * digits.len() as u32 + 1);
    let tag_height = cell * 7;
    let (image_width, image_height) = image.dimensions();
    let left = (x.max(0.0) as u32).min(image_width.saturating_sub(tag_width));
    let top = (y.max(0.0) as u32).min(image_height.saturating_sub(tag_height));
    for py in top..(top + tag_height).min(image_height) {
        for px in left..(left + tag_width).min(image_width) {
            image.put_pixel(px, py, image::Rgba(LAYER_COLOR));
        }
    }
    for (index, digit) in digits.bytes().enumerate() {
        let glyph = DIGITS[usize::from(digit - b'0')];
        let origin_x = left + cell * (1 + 4 * index as u32);
        let origin_y = top + cell;
        for (row, bits) in glyph.iter().enumerate() {
            for column in 0..3u32 {
                if bits & (0b100 >> column) == 0 {
                    continue;
                }
                for dy in 0..cell {
                    for dx in 0..cell {
                        let px = origin_x + column * cell + dx;
                        let py = origin_y + row as u32 * cell + dy;
                        if px < image_width && py < image_height {
                            image.put_pixel(px, py, image::Rgba([255, 255, 255, 255]));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::Point as ScenePoint;

    use super::*;

    #[test]
    fn a_balloon_is_measured_on_its_pixels_with_its_letters_filled() {
        // A white disc of radius 80 with a black outline on grey, with dark
        // "letters" across its middle.
        let page = RgbaImage::from_fn(300, 300, |x, y| {
            let distance = (f64::from(x) - 150.0).hypot(f64::from(y) - 150.0);
            let value = if distance < 80.0 {
                if (130..170).contains(&y) && (110..190).contains(&x) && x % 6 < 3 {
                    20
                } else {
                    255
                }
            } else if distance < 84.0 {
                0
            } else {
                120
            };
            image::Rgba([value, value, value, 255])
        });
        let [x, y, width, height] = pixel_inner_box(
            &page,
            [66.0, 66.0, 168.0, 168.0],
            [110.0, 130.0, 80.0, 40.0],
        )
        .unwrap();
        // The inscribed square is 113 px; the margin is a fourteenth of 160.
        assert!((85.0..=100.0).contains(&width), "width {width}");
        assert!((85.0..=100.0).contains(&height), "height {height}");
        assert!((x + width / 2.0 - 150.0).abs() <= 3.0 && (y + height / 2.0 - 150.0).abs() <= 3.0);
        // An open balloon leaks into the page and is left to the outline.
        let open = RgbaImage::from_pixel(300, 300, image::Rgba([255, 255, 255, 255]));
        assert!(
            pixel_inner_box(
                &open,
                [66.0, 66.0, 168.0, 168.0],
                [110.0, 130.0, 80.0, 40.0]
            )
            .is_none()
        );
    }

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
