//! Typesetting tools for agents: a page's text layers with what the renderer
//! drew for them, line by line, crops of the page as it shows, with or
//! without its text or as imported, trial renders of text, box and typography
//! changes that commit nothing, and the chapter's text and check with page
//! and layer numbers. A fit is a trial with auto-fit in the intended box.

use std::{collections::HashMap, io::Cursor};

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
use super::lines::{Along, MINIMUM_STRENGTH, printed_lines};

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
    /// Draw every text layer's box with its number (magenta), the measured
    /// inside of its balloon (blue) and the shape a balloon flow lays it out
    /// in (green).
    #[serde(default)]
    pub(super) outline: bool,
}

/// Typography fields to change. A field left out keeps the layer's current
/// value, so a new size keeps the font, colour and outline.
#[derive(Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct TrialTypography {
    /// A family name from available_fonts.
    #[serde(default)]
    preferred_font: Option<String>,
    #[serde(default)]
    font_weight: Option<u16>,
    #[serde(default)]
    font_style: Option<AgentFontStyle>,
    /// Fixed size in page pixels; it turns auto-fit off unless auto_fit is
    /// given too.
    #[serde(default)]
    size: Option<f32>,
    /// True fits the text to its box, up to a page-scaled ceiling.
    #[serde(default)]
    auto_fit: Option<bool>,
    #[serde(default)]
    color: Option<[u8; 4]>,
    #[serde(default)]
    stroke_color: Option<[u8; 4]>,
    /// Outline width in page pixels; 0 removes the outline.
    #[serde(default)]
    stroke_width: Option<f32>,
    #[serde(default)]
    alignment: Option<AgentTextAlignment>,
    #[serde(default)]
    writing_mode: Option<AgentWritingMode>,
    /// Where lines may break; automatic (hyphenating a word as a last
    /// resort) when never set or set back to automatic.
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
    /// Confirm the OCR reading as it is.
    #[serde(default)]
    confirm_source: bool,
    /// New text box, [x, y, width, height] in page pixels.
    #[serde(default)]
    region: Option<[f64; 4]>,
    /// Turn the new box about its centre by this many degrees, clockwise;
    /// 180 sets text upside down. Needs region, or with corners turns the
    /// text on the surface about its centre.
    #[serde(default)]
    angle: Option<f32>,
    /// `balloon` drops the layer's box and flows it in the balloon that holds
    /// it: each line as wide as the balloon at its height, the block placed
    /// in the balloon, a joined balloon shared among its layers. A region
    /// sets a box again.
    #[serde(default)]
    flow: Option<TrialFlow>,
    /// `along_source` lays the text along the source lettering: the box of
    /// its detected region turned to the slant of the printed lines measured
    /// on the page (source_lines), a lone Japanese column turned 90 degrees
    /// to read top to bottom. Instead of region and angle. With corners it
    /// turns the text on the surface until its lines take the slant of the
    /// lines printed inside the corners.
    /// A drawn line that stretches a sound (a long wavy or straight stroke
    /// after a kana, OCR'd as 〜 or ー) is art when the clean view still shows
    /// it: letter only the sound before it, in that kana's place and size,
    /// and let the line carry the length; never stretch letters along it.
    #[serde(default)]
    orient: Option<TrialOrient>,
    /// Letter on a surface seen at an angle (a newspaper, a phone or TV
    /// screen, a board, a sign): its four corners in page pixels, top-left,
    /// top-right, bottom-right, bottom-left as the lettering reads, e.g. the
    /// column or screen the original text is printed in. The text is laid out
    /// flat in a box the size of that surface and drawn in perspective onto
    /// it, its lines along the surface's top and bottom edges unless angle or
    /// orient turns them. Instead of region; a region alone sets a flat box
    /// again. A surface seen nearly square-on takes a turned region instead.
    #[serde(default)]
    corners: Option<[[f64; 2]; 4]>,
    /// Typography fields to change; the others keep their current values.
    #[serde(default)]
    typography: Option<TrialTypography>,
    /// Show or hide the layer.
    #[serde(default)]
    visible: Option<bool>,
    /// What the lettering is, when detection or a merge named it wrongly.
    #[serde(default)]
    role: Option<TrialRole>,
}

#[derive(Clone, Copy, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrialRole {
    /// Speech or thought in a balloon.
    Dialogue,
    /// A sound effect.
    Onomatopoeia,
    /// Narration, a sign or other text outside balloons.
    FreeText,
}

impl TrialRole {
    fn kind(self) -> &'static str {
        match self {
            Self::Dialogue => "dev.koharu.text.dialogue",
            Self::Onomatopoeia => "dev.koharu.text.onomatopoeia",
            Self::FreeText => "dev.koharu.text.free-text",
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrialFlow {
    Balloon,
}

#[derive(Clone, Copy, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum TrialOrient {
    AlongSource,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ApplyLayers {
    pub(super) changes: Vec<TrialChange>,
    /// Return each changed layer in full, as page_layers does, instead of
    /// the short form.
    #[serde(default)]
    pub(super) full: bool,
    /// Return no layers, only that the changes landed: for text and source
    /// passes that do not need the layout.
    #[serde(default)]
    pub(super) quiet: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct HideUntranslated {
    /// First and last page number, from 1; every page when absent.
    #[serde(default)]
    pub(super) pages: Option<[usize; 2]>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct MergeTextLayers {
    /// Two or more layers of one page, in reading order; the first keeps the
    /// merged text, its role and its balloon flow, the others go. A role
    /// change on apply_layers renames what the merged lettering is.
    pub(super) layers: Vec<String>,
    /// Put between the joined source texts and translations; nothing by
    /// default, as for one sound effect split in two.
    #[serde(default)]
    pub(super) joiner: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckChapterArguments {
    /// Kinds to list: fit, translation, ocr, cleaning, hidden, unboxed,
    /// style (shown translated text whose font and colour were never chosen),
    /// and three slow kinds listed only when asked for: cleaned (lettering
    /// the cleaning removed with no translated layer over it), balloons
    /// (shown translated text in a balloon that crosses its inside, sits off
    /// its centre, is small, or is set well below the size of similar
    /// balloons) and lettering (shown translated text whose fill tone or
    /// outline differs from the source lettering's); every other kind when
    /// absent.
    #[serde(default)]
    pub(super) kinds: Option<Vec<String>>,
    /// JSON instead of one line per issue.
    #[serde(default)]
    pub(super) full: bool,
    /// First and last page number, from 1, to check; every page when
    /// absent. Check a long chapter's slow kinds in ranges of about 30 pages.
    #[serde(default)]
    pub(super) pages: Option<[usize; 2]>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct AddTextLayer {
    pub(super) page: String,
    /// [x, y, width, height] in page pixels. Inside a balloon the text flows
    /// in the balloon's shape.
    pub(super) region: [f64; 4],
    /// The lettering as read off the page, in the source language.
    #[serde(default)]
    pub(super) source: Option<String>,
    /// The translation, or its lines.
    #[serde(default)]
    pub(super) text: Option<String>,
    #[serde(default)]
    pub(super) lines: Option<Vec<String>>,
    /// Typography, as apply_layers takes it. Without a colour the text is
    /// black, or white on a dark ground.
    #[serde(default)]
    pub(super) typography: Option<TrialTypography>,
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
    /// Draw the changed layers' boxes with their numbers (magenta), the
    /// measured inside of their balloons (blue) and the shape a balloon flow
    /// lays each out in (green).
    #[serde(default)]
    outline: bool,
    /// Return each changed layer in full, as page_layers does, instead of
    /// the short form.
    #[serde(default)]
    full: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ChapterText {
    /// First and last page number to list, from 1; every page when absent.
    #[serde(default)]
    pages: Option<[usize; 2]>,
    /// JSON with ids instead of one line per layer.
    #[serde(default)]
    full: bool,
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
    let source = base_image(snapshot, page).await?;
    let reading = reading_order(snapshot, page)?;
    let shapes = koharu_renderer::balloon_flow_shapes(snapshot, page)?;
    let mut layers = Vec::new();
    let mut filled = Vec::new();
    if let Some(group) = snapshot.page(page)?.text_group()? {
        for (index, layer) in group.text_layers()?.enumerate() {
            let mut value = layer_value(snapshot, &frame, layer.id(), index + 1)?;
            value["reading"] = json!(
                reading
                    .iter()
                    .position(|id| *id == layer.id())
                    .map(|position| position + 1)
            );
            measure_balloon(&clean, &mut value, &shapes);
            measure_source_lines(snapshot, &source, layer.id(), &mut value)?;
            if let Some(region) = layer.content()?.source_region()? {
                filled.extend(
                    snapshot
                        .relations_from_as::<koharu_scene::Inside>(region.id())
                        .map(|relation| relation.value().target),
                );
                measure_lettering(snapshot, &source, &clean, region.id(), &mut value)?;
            }
            if let Some(balloon) = layer.balloon_target()? {
                filled.push(balloon.id());
            }
            layers.push(value);
        }
    }
    // Balloons no text layer flows in or was read from: emptied by the
    // cleaning, or lettering detection missed.
    let mut empty_balloons = Vec::new();
    for entity in snapshot.descendants(page)? {
        let id = entity.id();
        let is_balloon = snapshot
            .component::<koharu_scene::Region>(id)?
            .is_some_and(|region| {
                region.kind == <koharu_scene::BubbleRegion as koharu_scene::RegionSpec>::kind()
            });
        if !is_balloon || filled.contains(&id) {
            continue;
        }
        if let Some(geometry) = snapshot.component::<Geometry>(id)? {
            let Some(bounds) = extent(&geometry) else {
                continue;
            };
            let inner = pixel_inner_box(&clean, bounds, bounds).or_else(|| inner_box(&geometry));
            empty_balloons.push(json!({ "box": bounds, "inner_box": inner }));
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
        "empty_balloons": empty_balloons,
        "cleaned_without_text": cleaned_without_text(
            &source,
            &clean,
            &translated_boxes(snapshot, page)?,
        ),
        "previous_page_end": neighbour(-1, true)?,
        "next_page_start": neighbour(1, false)?,
    }))
}

/// Records on `value` what the cleaning did to the source lettering of a
/// layer read from `region`, and whether the layer's fill and outline follow
/// it.
fn measure_lettering(
    snapshot: &Snapshot,
    source: &RgbaImage,
    clean: &RgbaImage,
    region: EntityId,
    value: &mut Value,
) -> Result<()> {
    let Some(bounds) = snapshot
        .component::<Geometry>(region)?
        .as_ref()
        .and_then(extent)
    else {
        return Ok(());
    };
    let lettering = lettering(source, clean, bounds);
    // Detection can swap fill and outline; the removed
    // lettering shows which tone the source used. Outlined
    // text matches when either paint has that tone.
    let light = |color: &Value| {
        let color = color.as_array()?;
        let channel = |index: usize| color.get(index)?.as_f64();
        Some(channel(0)? * 0.299 + channel(1)? * 0.587 + channel(2)? * 0.114 > 128.0)
    };
    let typography = &value["typography"];
    let mut paints = vec![light(&typography["color"])];
    if typography["stroke_width"]
        .as_f64()
        .is_some_and(|width| width >= 1.0)
    {
        paints.push(light(&typography["stroke_color"]));
    }
    let mismatch = value["visible"] == json!(true)
        && match lettering.tone {
            Some(tone) => paints
                .iter()
                .flatten()
                .all(|light| *light != (tone == "light")),
            None => false,
        };
    // The source's outline is the style to repeat: outlined
    // lettering wants an outline, plain lettering none.
    let outlined = typography["stroke_width"]
        .as_f64()
        .is_some_and(|width| width >= 1.0);
    // Text on art takes an outline for readability even
    // where the source has none; in a balloon it follows
    // the source either way.
    let check_outline = value["visible"] == json!(true)
        && lettering.tone.is_some()
        && if lettering.outline.is_some() {
            !outlined
        } else {
            outlined && value["balloon"].is_object()
        };
    value["lettering"] = json!({
        "left": lettering.left,
        "tone": lettering.tone,
        "outline": lettering.outline,
        "check_color": mismatch.then_some(true),
        "check_outline": check_outline.then_some(true),
    });
    Ok(())
}

struct Lettering {
    left: &'static str,
    /// The tone of the lettering's fill.
    tone: Option<&'static str>,
    /// The tone of its outline, when it has one.
    outline: Option<&'static str>,
}

/// What the cleaning did to the source lettering in a text region: how many
/// of its strong edges the clean page still shows (enhancement moves tones,
/// not edges), and the tone of the pixels it removed, the lettering's own.
/// Outlined lettering is dark text ringed in white over screentone (or the
/// reverse): a ring that stands out from the ground the cleaning left.
fn lettering(source: &RgbaImage, clean: &RgbaImage, [x, y, width, height]: [f64; 4]) -> Lettering {
    let luma = |image: &RgbaImage, px: u32, py: u32| {
        let pixel = image.get_pixel(px, py);
        (u32::from(pixel[0]) * 299 + u32::from(pixel[1]) * 587 + u32::from(pixel[2]) * 114) / 1000
    };
    let (image_width, image_height) = source.dimensions().min(clean.dimensions());
    let left = (x.max(1.0) as u32).min(image_width.saturating_sub(2));
    let top = (y.max(1.0) as u32).min(image_height.saturating_sub(2));
    let right = ((x + width) as u32).clamp(left, image_width.saturating_sub(2));
    let bottom = ((y + height) as u32).clamp(top, image_height.saturating_sub(2));
    let edge = |image: &RgbaImage, px: u32, py: u32| {
        let dx = luma(image, px + 1, py).abs_diff(luma(image, px - 1, py));
        let dy = luma(image, px, py + 1).abs_diff(luma(image, px, py - 1));
        dx.max(dy) > 60
    };
    // Mean tones: a screentone is dots, and a cleaning that redraws it moves
    // them, so the lettering is what differs between the source's tone over
    // three pixels and the clean ground over seven.
    // Summed tables over the region and a seven-pixel margin give any mean
    // in four reads.
    let (width_all, height_all) = clean.dimensions().min(source.dimensions());
    let (x_from, y_from) = (left.saturating_sub(7), top.saturating_sub(7));
    let (x_to, y_to) = ((right + 7).min(width_all), (bottom + 7).min(height_all));
    let table = |image: &RgbaImage| {
        let (w, h) = ((x_to - x_from) as usize, (y_to - y_from) as usize);
        let mut sums = vec![0u64; (w + 1) * (h + 1)];
        for row in 0..h {
            let mut line = 0u64;
            for column in 0..w {
                line += u64::from(luma(image, x_from + column as u32, y_from + row as u32));
                sums[(row + 1) * (w + 1) + column + 1] = sums[row * (w + 1) + column + 1] + line;
            }
        }
        (sums, w + 1)
    };
    let (source_sums, clean_sums) = (table(source), table(clean));
    let mean = |sums: &(Vec<u64>, usize), px: u32, py: u32, reach: u32| {
        let (sums, stride) = sums;
        let x0 = px.saturating_sub(reach).max(x_from) - x_from;
        let y0 = py.saturating_sub(reach).max(y_from) - y_from;
        let x1 = (px + reach + 1).min(x_to) - x_from;
        let y1 = (py + reach + 1).min(y_to) - y_from;
        let at = |x: u32, y: u32| sums[y as usize * stride + x as usize];
        let total = at(x1, y1) + at(x0, y0) - at(x0, y1) - at(x1, y0);
        (total / u64::from((x1 - x0) * (y1 - y0)).max(1)) as u32
    };
    let ground = |px: u32, py: u32| mean(&clean_sums, px, py, 3);
    let (mut before, mut after) = (0u64, 0u64);
    let (columns, rows) = ((right - left) as usize, (bottom - top) as usize);
    let mut kept = vec![true; columns * rows];
    let mut removed = Vec::new();
    for py in top..bottom {
        for px in left..right {
            before += u64::from(edge(source, px, py));
            after += u64::from(edge(clean, px, py));
            // Both the pixel and its neighbourhood differ from the ground: a
            // dot of screentone averages into it, and the light fringe
            // around a stroke is ground-coloured itself.
            let (own, from, to) = (
                luma(source, px, py),
                mean(&source_sums, px, py, 1),
                ground(px, py),
            );
            if from.abs_diff(to) > 80 && own.abs_diff(to) > 80 {
                removed.push(own);
                kept[(py - top) as usize * columns + (px - left) as usize] = false;
            }
        }
    }
    // An outline is a ring of the opposite tone just outside the strokes
    // that stands out from the ground the cleaning left there: three pixels
    // out from a removed stroke, the source is far lighter (or darker) than
    // the clean page. In a white balloon the ring and the ground agree, so
    // plain text reads as plain; redrawn screentone does not form a ring.
    let removed_at = |px: u32, py: u32| {
        px >= left
            && px < right
            && py >= top
            && py < bottom
            && !kept[(py - top) as usize * columns + (px - left) as usize]
    };
    // The share of the ring three pixels out from strokes of one tone that
    // is far lighter (dark fill) or darker (light fill) than the ground the
    // cleaning left there.
    let ringed = |fill_dark: bool| {
        let (mut ring, mut ringed) = (0u64, 0u64);
        for py in top..bottom {
            for px in left..right {
                if !removed_at(px, py) || (luma(source, px, py) < 128) != fill_dark {
                    continue;
                }
                for (dx, dy) in [(3i64, 0i64), (-3, 0), (0, 3), (0, -3)] {
                    let (nx, ny) = (i64::from(px) + dx, i64::from(py) + dy);
                    // A tight detected box can cut the ring: it may reach
                    // into the margin the tables cover.
                    if nx < i64::from(x_from)
                        || ny < i64::from(y_from)
                        || nx >= i64::from(x_to)
                        || ny >= i64::from(y_to)
                    {
                        continue;
                    }
                    let (nx, ny) = (nx as u32, ny as u32);
                    if removed_at(nx, ny) && (luma(source, nx, ny) < 128) == fill_dark {
                        continue;
                    }
                    ring += 1;
                    let (from, to) = (mean(&source_sums, nx, ny, 1), ground(nx, ny));
                    ringed += u64::from(if fill_dark {
                        from > to + 30
                    } else {
                        to > from + 30
                    });
                }
            }
        }
        (ring >= 64).then(|| ringed as f64 / ring as f64)
    };
    // The fill is the tone whose strokes the ring surrounds best; an outline
    // as thick as the strokes outweighs them, so the median cannot tell.
    // A fill holds at least a quarter of what the cleaning removed: the light
    // anti-aliased fringe of dark text is not a light fill with a dark ring.
    let dark_share =
        removed.iter().filter(|tone| **tone < 128).count() as f64 / removed.len().max(1) as f64;
    let (dark_ring, light_ring) = if removed.len() >= 16 {
        (
            (dark_share >= 0.25).then(|| ringed(true)).flatten(),
            (dark_share <= 0.75).then(|| ringed(false)).flatten(),
        )
    } else {
        (None, None)
    };
    let (fill, outline) = match (dark_ring, light_ring) {
        // Where there is no ring the share stays near nothing; a real one
        // reaches past a third (0.45 on Osananajimi 6#10: a white ring on
        // tone, with strokes whose inner pixels look into other strokes).
        (Some(dark), light) if dark >= 0.35 && dark >= light.unwrap_or(0.0) => {
            (Some("dark"), Some("light"))
        }
        (_, Some(light)) if light >= 0.35 => (Some("light"), Some("dark")),
        _ => (
            (removed.len() >= 16).then(|| {
                let mut tones = removed.clone();
                let middle = tones.len() / 2;
                if *tones.select_nth_unstable(middle).1 < 128 {
                    "dark"
                } else {
                    "light"
                }
            }),
            None,
        ),
    };
    let left = if before == 0 {
        "none"
    } else {
        let ratio = after as f64 / before as f64;
        if ratio < 0.35 {
            "removed"
        } else if ratio > 0.75 {
            "kept"
        } else {
            "partly"
        }
    };
    Lettering {
        left,
        tone: fill,
        outline,
    }
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
fn measure_balloon(clean: &RgbaImage, value: &mut Value, shapes: &HashMap<EntityId, Geometry>) {
    // A flowing layer is measured in the part of its balloon the renderer
    // lays it out in; a layer with its own box, in the lobe its box sits in.
    let shape = (value["flow"] == "balloon")
        .then(|| value["id"].as_str().and_then(|id| entity(id).ok()))
        .flatten()
        .and_then(|id| shapes.get(&id));
    let measured = read_box(&value["balloon"]["box"])
        .zip(read_box(&value["box"]))
        .and_then(|(balloon, text)| balloon_interior(clean, balloon, text, shape));
    match measured {
        Some(interior) => {
            value["balloon"]["inner_box"] = json!(interior.inner_box);
            value["balloon"]["interior"] = json!({
                "box": interior.bounds,
                "centre": interior.centre,
            });
            // A panel line or hatching across a balloon closes the flood
            // early: an inside far smaller than the detected balloon, or one
            // the text's own box mostly lies outside of, is not the balloon.
            let partial = read_box(&value["balloon"]["box"]).is_some_and(|balloon| {
                interior.bounds[2] * interior.bounds[3] < balloon[2] * balloon[3] * 0.3
            }) || read_box(&value["box"])
                .is_some_and(|text| overlap(text, interior.bounds) < text[2] * text[3] * 0.5);
            value["balloon"]["measured"] = json!(if partial { "partial" } else { "pixels" });
            if let Some(drawn) = read_box(&value["rendered"]["text_box"]) {
                // How far the drawn block's centre sits from the lobe's
                // centroid, and the narrowest gap between a line and the
                // balloon's inside at that line's height.
                value["balloon"]["off_centre"] = json!([
                    (drawn[0] + drawn[2] / 2.0 - interior.centre[0]).round(),
                    (drawn[1] + drawn[3] / 2.0 - interior.centre[1]).round(),
                ]);
                let lines = value["rendered"]["lines"]
                    .as_array()
                    .map(|lines| {
                        lines
                            .iter()
                            .filter_map(|line| line["width"].as_f64())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let alignment = value["typography"]["alignment"]
                    .as_str()
                    .unwrap_or("Center");
                let vertical = value["typography"]["writing_mode"] == json!("Vertical");
                if !lines.is_empty() && !vertical {
                    let step = drawn[3] / lines.len() as f64;
                    let gap = lines
                        .iter()
                        .enumerate()
                        .filter_map(|(index, width)| {
                            let y = drawn[1] + step * (index as f64 + 0.5);
                            let (start, end) = match alignment {
                                "Start" => (drawn[0], drawn[0] + width),
                                "End" => (drawn[0] + drawn[2] - width, drawn[0] + drawn[2]),
                                _ => {
                                    let middle = drawn[0] + drawn[2] / 2.0;
                                    (middle - width / 2.0, middle + width / 2.0)
                                }
                            };
                            let (left, right) = interior.row(y)?;
                            Some((start - left).min(right - end))
                        })
                        .fold(f64::INFINITY, f64::min);
                    if gap.is_finite() {
                        value["balloon"]["edge_gap"] = json!(gap.round());
                    }
                }
            }
        }
        None if value["balloon"].is_object() => {
            value["balloon"]["measured"] = json!("outline");
        }
        None => {}
    }
}

/// The area two [x, y, width, height] boxes share.
fn overlap(left: [f64; 4], right: [f64; 4]) -> f64 {
    let width = (left[0] + left[2]).min(right[0] + right[2]) - left[0].max(right[0]);
    let height = (left[1] + left[3]).min(right[1] + right[3]) - left[1].max(right[1]);
    width.max(0.0) * height.max(0.0)
}

/// The chapter's pages in order with the text of each, for `chapter_text`.
pub(super) fn chapter_text(snapshot: &Snapshot, arguments: &ChapterText) -> Result<Value> {
    let [first, last] = arguments.pages.unwrap_or([1, usize::MAX]);
    if !arguments.full {
        // One line per layer keeps a whole chapter readable in one result.
        let mut text = String::new();
        let flat = |value: &Value| value.as_str().unwrap_or("").replace('\n', " / ");
        for (index, page) in snapshot.pages().enumerate() {
            let number = index + 1;
            if number < first || number > last {
                continue;
            }
            text.push_str(&format!(
                "# page {number} {} {}\n",
                page.page()?.label,
                page.id()
            ));
            let reading = reading_order(snapshot, page.id())?;
            let mut texts = page_texts(snapshot, page.id())?;
            texts.sort_by_key(|layer| {
                let id = layer["id"].as_str().and_then(|id| entity(id).ok());
                reading
                    .iter()
                    .position(|candidate| Some(*candidate) == id)
                    .unwrap_or(usize::MAX)
            });
            for layer in texts {
                let role = layer["role"]
                    .as_str()
                    .and_then(|role| role.rsplit('.').next())
                    .unwrap_or("text");
                let hidden = if layer["shows"] == json!(true) {
                    ""
                } else {
                    " hidden"
                };
                text.push_str(&format!(
                    "{} {role}{hidden} | {} | {}\n",
                    layer["number"],
                    flat(&layer["source"]),
                    flat(&layer["translation"]),
                ));
            }
        }
        return Ok(Value::String(text));
    }
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

/// One entry per page for an agent's overview: its number, label, id, size,
/// and how many of its text layers there are, carry a translation and show.
pub(super) fn page_summaries(snapshot: &Snapshot) -> Result<Vec<Value>> {
    let mut pages = Vec::new();
    for (index, page) in snapshot.pages().enumerate() {
        let value = page.page()?;
        let texts = page_texts(snapshot, page.id())?;
        let count = |test: fn(&Value) -> bool| texts.iter().filter(|text| test(text)).count();
        pages.push(json!({
            "number": index + 1,
            "label": value.label,
            "id": page.id(),
            "size": [value.width, value.height],
            "text_layers": texts.len(),
            "translated": count(|text| {
                text["translation"].as_str().is_some_and(|text| !text.trim().is_empty())
            }),
            "shown": count(|text| text["shows"] == true),
        }));
    }
    Ok(pages)
}

/// What an agent's page and layer addresses rest on, page by page in project
/// order: each page's text layers, their texts, visibility and boxes. The
/// owner cleaning a page leaves it alone, so an agent is told to reread only
/// when someone changed the text side.
pub(super) fn text_signatures(snapshot: &Snapshot) -> Result<Vec<(EntityId, u64)>> {
    use std::hash::{Hash as _, Hasher as _};
    let mut pages = Vec::new();
    for page in snapshot.pages().map(|page| page.id()).collect::<Vec<_>>() {
        let mut hasher = std::hash::DefaultHasher::new();
        for text in page_texts(snapshot, page)? {
            text.to_string().hash(&mut hasher);
        }
        if let Some(group) = snapshot.page(page)?.text_group()? {
            for layer in group.text_layers()? {
                format!("{:?}", layer.frame()?).hash(&mut hasher);
            }
        }
        pages.push((page, hasher.finish()));
    }
    Ok(pages)
}

/// Whether `page`'s text group is hidden, which hides every text layer on it
/// whatever the layer's own visibility.
fn text_group_hidden(snapshot: &Snapshot, page: EntityId) -> Result<bool> {
    Ok(match snapshot.page(page)?.text_group()? {
        Some(group) => !shown(snapshot, group.id())?,
        None => false,
    })
}

fn shown(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
    Ok(snapshot
        .component::<Visibility>(entity)?
        .is_none_or(|visibility| visibility.visible && visibility.opacity > 0.0))
}

/// The chapter check with page and layer numbers, both texts, and what hides
/// a hidden block, for agents: one line per issue by default, kinds to keep
/// optional (fit, translation, ocr, cleaning, hidden, unboxed).
/// The chapter check an agent reads, with the places cleaned without text
/// when the kinds ask for them.
pub(super) struct AgentCheck {
    pub(super) check: ChapterCheck,
    pub(super) cleaned_without_text: Option<Vec<(EntityId, Vec<[f64; 4]>)>>,
    /// Text layers as `page_layers` reports them, balloons measured, for the
    /// pages the asked kinds need: every page for balloons, else those with
    /// a fit issue.
    pub(super) layers: HashMap<EntityId, Value>,
    /// Shown layers whose colour or outline differs from the source
    /// lettering's, as `page_layers` reports them, by page.
    pub(super) lettering: Option<Vec<(EntityId, Vec<Value>)>>,
}

impl CheckChapterArguments {
    /// Every kind is listed by default except `cleaned`, `balloons` and
    /// `lettering`, which render every page and are asked for by name.
    pub(super) fn wants(&self, kind: &str) -> bool {
        match &self.kinds {
            Some(kinds) => kinds.iter().any(|wanted| wanted == kind),
            None => !matches!(kind, "cleaned" | "balloons" | "lettering"),
        }
    }

    /// Whether page `number` (from 1) is in the checked range.
    pub(super) fn covers(&self, number: usize) -> bool {
        self.pages
            .is_none_or(|[first, last]| (first..=last).contains(&number))
    }
}

/// Pages a check renders run this many at a time: rendering and the pixel
/// work overlap without holding every page image at once.
const CONCURRENT_PAGES: usize = 4;

/// `pages`' text layers as `page_layers` reports them, keyed by layer.
pub(super) async fn measured_layers(
    desktop: &Desktop,
    snapshot: &Snapshot,
    pages: &[EntityId],
) -> Result<HashMap<EntityId, Value>> {
    use futures::{StreamExt as _, TryStreamExt as _};
    let pages = futures::stream::iter(pages.iter().copied())
        .map(|page| async move {
            let Some(group) = snapshot.page(page)?.text_group()? else {
                return Ok(Vec::new());
            };
            let layers = group
                .text_layers()?
                .map(|layer| layer.id())
                .collect::<Vec<_>>();
            let frame = desktop.renderer().render(snapshot, page).await?;
            let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
            let shapes = koharu_renderer::balloon_flow_shapes(snapshot, page)?;
            let snapshot = snapshot.clone();
            tokio_rayon::spawn(move || {
                layers
                    .into_iter()
                    .enumerate()
                    .map(|(index, layer)| {
                        let mut value = layer_value(&snapshot, &frame, layer, index + 1)?;
                        measure_balloon(&clean, &mut value, &shapes);
                        Ok((layer, value))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .await
        })
        .buffer_unordered(CONCURRENT_PAGES)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(pages.into_iter().flatten().collect())
}

/// What a fit warning needs to be fixed without another call: the drawn
/// size, the line that overflows most against its room, and the largest
/// size the box takes when auto-fit stopped at its ceiling.
fn fit_details(value: &Value) -> String {
    let rendered = &value["rendered"];
    let mut details = Vec::new();
    if let Some(size) = rendered["font_size"].as_f64() {
        details.push(format!("size={size}"));
    }
    let worst = rendered["lines"].as_array().and_then(|lines| {
        lines
            .iter()
            .filter_map(|line| Some((line["width"].as_f64()?, line["room"].as_f64()?)))
            .max_by(|left, right| (left.0 - left.1).total_cmp(&(right.0 - right.1)))
    });
    if let Some((width, room)) = worst {
        details.push(format!("widest={width}/{room}"));
    }
    if let Some(size) = rendered["fits_up_to"].as_f64() {
        details.push(format!("fits_up_to={size}"));
    }
    details.join(" ")
}

/// The balloon issues of shown, translated text in balloons, page by page.
fn balloon_issues(
    snapshot: &Snapshot,
    pages: &[EntityId],
    layers: &HashMap<EntityId, Value>,
) -> Result<Vec<(EntityId, EntityId, Vec<String>)>> {
    let mut issues = Vec::new();
    for page in pages {
        let Some(group) = snapshot.page(*page)?.text_group()? else {
            continue;
        };
        let page_layers = group
            .text_layers()?
            .filter_map(|layer| layers.get(&layer.id()).map(|value| (layer.id(), value)))
            .collect::<Vec<_>>();
        let values = page_layers
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        for ((layer, _), found) in page_layers.iter().zip(page_balloon_issues(&values)) {
            if !found.is_empty() {
                issues.push((*page, *layer, found));
            }
        }
    }
    Ok(issues)
}

/// One page's layers as `page_layers` reports them, each with its balloon
/// issues: a line crossing the balloon's inside, a block off the centre of
/// its lobe by more than an eighth of the lobe, a size the renderer calls
/// small, or a size under 85% of the page's median balloon size (a larger
/// shout is the letterer's choice). Only shown, translated, horizontal text
/// in a balloon measured on the page's pixels counts.
fn page_balloon_issues(layers: &[&Value]) -> Vec<Vec<String>> {
    let lettered = |value: &Value| {
        value["visible"] == true
            && value["translated_text"]
                .as_str()
                .is_some_and(|text| !text.trim().is_empty())
            && matches!(
                value["balloon"]["measured"].as_str(),
                Some("pixels" | "partial")
            )
            && value["typography"]["writing_mode"] != "Vertical"
    };
    let width = |value: &Value| {
        read_box(&value["balloon"]["interior"]["box"])
            .or_else(|| read_box(&value["balloon"]["box"]))
            .map(|bounds| bounds[2])
    };
    // A narrow balloon takes a smaller size than a wide one on the same page:
    // text is compared with balloons of a similar width.
    let median = |of: &Value| {
        let own = width(of)?;
        let mut sizes = layers
            .iter()
            .filter(|value| lettered(value))
            .filter(|value| width(value).is_some_and(|other| (other / own - 1.0).abs() <= 0.35))
            .filter_map(|value| value["rendered"]["font_size"].as_f64())
            .collect::<Vec<_>>();
        sizes.sort_by(f64::total_cmp);
        (sizes.len() >= 3).then(|| sizes[sizes.len() / 2])
    };
    layers
        .iter()
        .map(|value| {
            let mut found = Vec::new();
            if !lettered(value) {
                return found;
            }
            let balloon = &value["balloon"];
            // A partial measure cannot tell a crossing or the centre.
            let measured = balloon["measured"] == "pixels";
            if let Some(gap) = balloon["edge_gap"]
                .as_f64()
                .filter(|gap| measured && *gap < -2.0)
            {
                found.push(format!("crosses={gap}"));
            }
            if let (Some([dx, dy]), Some([_, _, width, height])) = (
                balloon["off_centre"]
                    .as_array()
                    .and_then(|off| Some([off.first()?.as_f64()?, off.get(1)?.as_f64()?])),
                read_box(&balloon["interior"]["box"]),
            ) && measured
                && (dx.abs() > width / 8.0 || dy.abs() > height / 8.0)
            {
                found.push(format!("off_centre={dx},{dy}"));
            }
            let size = value["rendered"]["font_size"].as_f64();
            if value["rendered"]["small"] == true {
                found.push(format!("small={}", size.unwrap_or_default()));
            } else if let (Some(size), Some(median)) = (size, median(value))
                && size < median * 0.85
            {
                found.push(format!("size={size} similar_median={median}"));
            }
            found
        })
        .collect()
}

/// `pages`' shown layers whose colour or outline differs from the source
/// lettering's, measured as `page_layers` does.
pub(super) async fn chapter_lettering(
    desktop: &Desktop,
    snapshot: &Snapshot,
    pages: &[EntityId],
) -> Result<Vec<(EntityId, Vec<Value>)>> {
    use futures::{StreamExt as _, TryStreamExt as _};
    futures::stream::iter(pages.iter().copied())
        .map(|page| async move {
            let Some(group) = snapshot.page(page)?.text_group()? else {
                return anyhow::Ok((page, Vec::new()));
            };
            let frame = desktop.renderer().render(snapshot, page).await?;
            let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
            let source = base_image(snapshot, page).await?;
            let mut layers = Vec::new();
            for (index, layer) in group.text_layers()?.enumerate() {
                let Some(region) = layer.content()?.source_region()? else {
                    continue;
                };
                let mut value = layer_value(snapshot, &frame, layer.id(), index + 1)?;
                measure_lettering(snapshot, &source, &clean, region.id(), &mut value)?;
                if value["lettering"]["check_color"] == json!(true)
                    || value["lettering"]["check_outline"] == json!(true)
                {
                    layers.push(value);
                }
            }
            Ok((page, layers))
        })
        .buffered(CONCURRENT_PAGES)
        .try_collect()
        .await
}

/// `pages`' places cleaned without a translated layer over them.
pub(super) async fn chapter_cleaned_without_text(
    desktop: &Desktop,
    snapshot: &Snapshot,
    pages: &[EntityId],
) -> Result<Vec<(EntityId, Vec<[f64; 4]>)>> {
    use futures::{StreamExt as _, TryStreamExt as _};
    futures::stream::iter(pages.iter().copied())
        .map(|page| async move {
            let source = base_image(snapshot, page).await?;
            let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
            let covered = translated_boxes(snapshot, page)?;
            let places =
                tokio_rayon::spawn(move || cleaned_without_text(&source, &clean, &covered)).await;
            anyhow::Ok((page, places))
        })
        .buffered(CONCURRENT_PAGES)
        .try_collect()
        .await
}

pub(super) fn chapter_issues(
    snapshot: &Snapshot,
    check: AgentCheck,
    arguments: &CheckChapterArguments,
) -> Result<Value> {
    let AgentCheck {
        check,
        cleaned_without_text: cleaned_places,
        layers: measured,
        lettering,
    } = check;
    let wanted = |kind: &str| arguments.wants(kind);
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    let page_number = |page: EntityId| {
        pages
            .iter()
            .position(|candidate| *candidate == page)
            .map_or(0, |index| index + 1)
    };
    let page_value = |page: EntityId| -> Result<Value> {
        Ok(json!({
            "id": page,
            "number": page_number(page),
            "label": snapshot.page(page)?.page()?.label,
        }))
    };
    let mut texts = Vec::new();
    let mut lines = Vec::new();
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    let covered = |page: EntityId| arguments.covers(page_number(page));
    let mut hidden_groups = std::collections::BTreeMap::<usize, usize>::new();
    for issue in check.texts {
        if !covered(issue.page) {
            continue;
        }
        let mut kinds = Vec::new();
        if let Some(fit) = issue.fit {
            let details = measured
                .get(&issue.element)
                .map(fit_details)
                .unwrap_or_default();
            kinds.push((
                "fit",
                format!("fit={} {details}", json!(fit).as_str().unwrap_or("warning"))
                    .trim_end()
                    .to_owned(),
            ));
        }
        if let Some(translation) = issue.translation {
            kinds.push((
                "translation",
                format!(
                    "translation={}",
                    json!(translation).as_str().unwrap_or("issue")
                ),
            ));
        }
        if issue.ocr_unverified {
            kinds.push(("ocr", "ocr_unverified".to_owned()));
        }
        if issue.cleaning_unverified {
            kinds.push(("cleaning", "cleaning_unverified".to_owned()));
        }
        if issue.hidden {
            kinds.push(("hidden", "hidden".to_owned()));
        }
        kinds.retain(|(kind, _)| wanted(kind));
        if kinds.is_empty() {
            continue;
        }
        for (kind, _) in &kinds {
            *counts.entry(kind).or_default() += 1;
        }
        let numbers = numbering(snapshot, issue.page)?;
        let content = snapshot.text_layer(issue.element)?.content()?;
        let source = content.source()?.map(|source| source.text.value);
        let translation = content
            .translation()?
            .map(|translation| translation.text.value);
        let hidden_by = if !issue.hidden {
            Value::Null
        } else if !shown(snapshot, issue.element)? {
            json!({ "layer": issue.element })
        } else {
            json!({ "text_group": snapshot.page(issue.page)?.text_group()?.map(|group| group.id()) })
        };
        let address = format!("{}#{}", page_number(issue.page), numbers(issue.element));
        // A hidden text group hides every layer of its page: one line says so.
        if !arguments.full
            && issue.hidden
            && hidden_by.get("text_group").is_some()
            && kinds.len() == 1
        {
            *hidden_groups.entry(page_number(issue.page)).or_default() += 1;
            continue;
        }
        if arguments.full {
            texts.push(json!({
                "address": address,
                "page": page_value(issue.page)?,
                "layer": numbers(issue.element),
                "element": issue.element,
                "source": source,
                "translation": translation,
                "translation_issue": issue.translation,
                "fit": issue.fit,
                "ocr_unverified": issue.ocr_unverified,
                "cleaning_unverified": issue.cleaning_unverified,
                "hidden": issue.hidden,
                "hidden_by": hidden_by,
            }));
        } else {
            let text = translation
                .filter(|text| !text.trim().is_empty())
                .or(source)
                .unwrap_or_default()
                .replace('\n', " / ");
            let text = text.chars().take(48).collect::<String>();
            lines.push(format!(
                "{address} {} | {text}",
                kinds
                    .iter()
                    .map(|(_, label)| label.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ));
        }
    }
    for (page, count) in hidden_groups {
        lines.push(format!(
            "{page} hidden_by=text_group layers={count} | the page's text group is hidden"
        ));
    }
    let mut automatic_style = Vec::new();
    if wanted("style") {
        for page in pages.iter().filter(|page| covered(**page)) {
            for layer in automatic_style_layers(snapshot, *page)? {
                *counts.entry("style").or_default() += 1;
                let address = format!(
                    "{}#{}",
                    page_number(*page),
                    numbering(snapshot, *page)?(layer)
                );
                let text = snapshot
                    .text_layer(layer)?
                    .content()?
                    .translation()?
                    .map(|translation| translation.text.value)
                    .unwrap_or_default()
                    .replace('\n', " / ");
                if arguments.full {
                    automatic_style
                        .push(json!({ "address": address, "element": layer, "translation": text }));
                } else {
                    let text = text.chars().take(48).collect::<String>();
                    lines.push(format!("{address} style=automatic | {text}"));
                }
            }
        }
    }
    let mut unboxed_text = Vec::new();
    if wanted("unboxed") {
        for issue in check.unboxed_text {
            if !covered(issue.page) {
                continue;
            }
            let boxes = koharu_pipeline::open_unboxed_text(snapshot, issue.page)?
                .into_iter()
                .map(|flag| [flag.x, flag.y, flag.width, flag.height].map(f64::round))
                .collect::<Vec<_>>();
            *counts.entry("unboxed").or_default() += boxes.len();
            if arguments.full {
                unboxed_text.push(
                    json!({ "page": page_value(issue.page)?, "count": issue.count, "boxes": boxes }),
                );
            } else {
                lines.push(format!(
                    "{} unboxed {}",
                    page_number(issue.page),
                    boxes
                        .iter()
                        .map(|[x, y, width, height]| format!("[{x},{y},{width},{height}]"))
                        .collect::<Vec<_>>()
                        .join(" "),
                ));
            }
        }
    }
    let mut balloons = Vec::new();
    if wanted("balloons") {
        for (page, layer, found) in balloon_issues(snapshot, &pages, &measured)? {
            *counts.entry("balloons").or_default() += 1;
            let address = format!(
                "{}#{}",
                page_number(page),
                numbering(snapshot, page)?(layer)
            );
            if arguments.full {
                balloons.push(json!({ "address": address, "element": layer, "issues": found }));
            } else {
                let text = measured[&layer]["translated_text"]
                    .as_str()
                    .unwrap_or_default()
                    .replace('\n', " / ");
                let text = text.chars().take(48).collect::<String>();
                lines.push(format!("{address} {} | {text}", found.join(" ")));
            }
        }
    }
    let mut styled = Vec::new();
    for (page, layers) in lettering.into_iter().flatten() {
        for value in layers {
            let address = format!("{}#{}", page_number(page), value["number"]);
            let source = &value["lettering"];
            let mut found = Vec::new();
            if source["check_color"] == json!(true) {
                found.push(format!(
                    "color source_fill={}",
                    source["tone"].as_str().unwrap_or("?")
                ));
            }
            if source["check_outline"] == json!(true) {
                found.push(match source["outline"].as_str() {
                    Some(outline) => format!("outline_missing source_outline={outline}"),
                    None => "outline_extra source_has_none".to_owned(),
                });
            }
            *counts.entry("lettering").or_default() += 1;
            if arguments.full {
                styled.push(json!({ "address": address, "lettering": source }));
            } else {
                let text = value["translated_text"]
                    .as_str()
                    .unwrap_or_default()
                    .replace('\n', " / ");
                let text = text.chars().take(48).collect::<String>();
                lines.push(format!("{address} {} | {text}", found.join(" ")));
            }
        }
    }
    let mut cleaned = Vec::new();
    if let Some(places) = cleaned_places {
        for (page, boxes) in places {
            if boxes.is_empty() {
                continue;
            }
            *counts.entry("cleaned").or_default() += boxes.len();
            if arguments.full {
                cleaned.push(json!({ "page": page_value(page)?, "boxes": boxes }));
            } else {
                lines.push(format!(
                    "{} cleaned_without_text {}",
                    page_number(page),
                    boxes
                        .iter()
                        .map(|[x, y, width, height]| format!("[{x},{y},{width},{height}]"))
                        .collect::<Vec<_>>()
                        .join(" "),
                ));
            }
        }
    }
    if arguments.full {
        return Ok(json!({
            "texts": texts,
            "unboxed_text": unboxed_text,
            "automatic_style": automatic_style,
            "cleaned_without_text": cleaned,
            "balloons": balloons,
            "lettering": styled,
            "counts": counts,
        }));
    }
    let summary = if counts.is_empty() {
        "No issues.".to_owned()
    } else {
        counts
            .iter()
            .map(|(kind, count)| format!("{kind} {count}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Ok(Value::String(format!("{summary}\n{}", lines.join("\n"))))
}

/// Visibility off for every shown text layer without a translation: blocks
/// an agent decided are not lettering, or left untranslated.
pub(super) fn hide_untranslated_steps(
    snapshot: &Snapshot,
    arguments: &HideUntranslated,
) -> Result<(Vec<EditStep>, usize)> {
    let [first, last] = arguments.pages.unwrap_or([1, usize::MAX]);
    let mut steps: Vec<EditStep> = Vec::new();
    for (index, page) in snapshot.pages().enumerate() {
        if index + 1 < first || index + 1 > last {
            continue;
        }
        let Some(group) = page.text_group()? else {
            continue;
        };
        for layer in group.text_layers()? {
            let translated = layer
                .content()?
                .translation()?
                .is_some_and(|translation| !translation.text.value.trim().is_empty());
            if translated || !shown(snapshot, layer.id())? {
                continue;
            }
            let id = layer.id();
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(id)?;
                edit.set(
                    id,
                    &Visibility {
                        origin: Origin::User,
                        visible: false,
                        opacity: 1.0,
                    },
                )
            }));
        }
    }
    let count = steps.len();
    Ok((steps, count))
}

/// One sound effect detection split in two (ぐ|ぽっ): the first layer takes
/// the joined source and translation and the box around all of them.
pub(super) fn merge_steps(
    snapshot: &Snapshot,
    layers: &[EntityId],
    joiner: &str,
) -> Result<Vec<EditStep>> {
    let [first, ..] = layers else {
        bail!("name two or more layers to merge");
    };
    if layers.len() < 2 {
        bail!("name two or more layers to merge");
    }
    let page = owning_page(snapshot, *first)?;
    let mut sources = Vec::new();
    let mut translations = Vec::new();
    let (mut left, mut top, mut right, mut bottom) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for layer in layers {
        if owning_page(snapshot, *layer)? != page {
            bail!("merged layers must be on one page");
        }
        let text = snapshot.text_layer(*layer)?;
        let content = text.content()?;
        sources.extend(content.source()?.map(|source| source.text.value));
        translations.extend(
            content
                .translation()?
                .map(|translation| translation.text.value)
                .filter(|text| !text.trim().is_empty()),
        );
        if let Some((l, t, r, b)) = text.frame()?.as_ref().and_then(bounds) {
            left = left.min(l);
            top = top.min(t);
            right = right.max(r);
            bottom = bottom.max(b);
        }
    }
    let mut steps = vec![Project::source_step(
        snapshot,
        *first,
        sources.join(joiner),
    )?];
    if !translations.is_empty() {
        steps.push(Project::translation_step(
            snapshot,
            *first,
            Some(translations.join(joiner)),
        )?);
    }
    // A first layer flowing in its balloon keeps flowing: the merged text
    // takes the balloon's shape rather than the boxes' union.
    let flows = snapshot.text_layer(*first)?.frame()?.is_none();
    if !flows && right > left && bottom > top {
        let points = [(left, top), (right, top), (right, bottom), (left, bottom)]
            .map(|(x, y)| Point { x, y })
            .to_vec();
        steps.push(Project::geometry_step(
            snapshot,
            vec![GeometryUpdate {
                layer: *first,
                points: Some(points),
            }],
        )?);
    }
    Ok(steps)
}

/// A crop of `page` in the chosen view, as PNG.
pub(super) async fn crop(
    desktop: &Desktop,
    snapshot: &Snapshot,
    arguments: &Crop,
) -> Result<(Value, String)> {
    let page = address(snapshot, &arguments.page)?;
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
        let stroke = (2.0 / scale).ceil().max(1.0) as u32;
        for flag in koharu_pipeline::open_unboxed_text(snapshot, page)? {
            draw_box(
                &mut image,
                [flag.x, flag.y, flag.width, flag.height],
                stroke,
                UNBOXED_COLOR,
            );
        }
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
            "text_group_hidden": text_group_hidden(snapshot, page)?.then_some(
                "the page's text group is hidden: none of its text layers is drawn",
            ),
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
    let page = address(snapshot, &arguments.page)?;
    let changed = elements(snapshot, &arguments.changes)?;
    for layer in &changed {
        if snapshot.page(*layer).is_ok() || owning_page(snapshot, *layer)? != page {
            bail!("element {layer} is not a layer of page {page}");
        }
    }
    let sources = orient_sources(snapshot, &arguments.changes).await?;
    let steps = steps(snapshot, arguments.changes, &sources, |_| None)?;
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
    let clean = render_image(desktop, &without_text(&trial, page)?, page).await?;
    let shapes = koharu_renderer::balloon_flow_shapes(&trial, page)?;
    let printed = base_image(snapshot, page).await?;
    for (value, layer) in layers.iter_mut().zip(&changed) {
        measure_balloon(&clean, value, &shapes);
        measure_source_lines(&trial, &printed, *layer, value)?;
    }
    if arguments.outline {
        draw_outlines(&mut image, &trial, &frame, &clean, page, &changed, scale)?;
    }
    let source = if arguments.compare {
        Some(source_image(snapshot, page).await?)
    } else {
        None
    };
    let (region, png) = picture(&image, source.as_ref(), region, scale)?;
    let layers = if arguments.full {
        layers
    } else {
        layers.iter().map(compact_layer).collect()
    };
    Ok((
        json!({
            "page": page,
            "region": region,
            "scale": scale,
            "layout": if arguments.compare { "source | preview" } else { "preview" },
            "text_group_hidden": text_group_hidden(snapshot, page)?.then_some(
                "the page's text group is hidden: none of its text layers is drawn",
            ),
            "layers": layers,
        }),
        png,
    ))
}

impl TrialTypography {
    /// These fields over `current`, a layer's typography; a new layer has
    /// none and auto-fits unless told otherwise.
    pub(super) fn onto(self, current: Option<&koharu_scene::Typography>) -> Typography {
        Typography {
            preferred_font: self
                .preferred_font
                .or_else(|| current.and_then(|current| current.preferred_font.clone())),
            font_weight: self
                .font_weight
                .or_else(|| current.and_then(|current| current.font_weight)),
            font_style: self
                .font_style
                .map(Into::into)
                .or_else(|| current.and_then(|current| current.font_style)),
            size: self
                .size
                .or_else(|| current.and_then(|current| current.size)),
            auto_fit: self.auto_fit.unwrap_or_else(|| {
                // A size given alone is the size to draw at.
                self.size.is_none() && current.is_none_or(|current| current.auto_fit)
            }),
            color: self
                .color
                .or_else(|| current.and_then(|current| current.color)),
            stroke_color: self
                .stroke_color
                .or_else(|| current.and_then(|current| current.stroke_color)),
            stroke_width: self
                .stroke_width
                .or_else(|| current.and_then(|current| current.stroke_width)),
            alignment: self
                .alignment
                .map(Into::into)
                .or_else(|| current.and_then(|current| current.alignment)),
            // Detection records the source lettering's direction, which the
            // renderer ignores for a translation; kept on a user edit it
            // would turn the translation vertical.
            writing_mode: self.writing_mode.map(Into::into).or_else(|| {
                current
                    .filter(|current| current.origin == koharu_scene::Origin::User)
                    .and_then(|current| current.writing_mode)
            }),
            line_breaking: match self.line_breaking {
                Some(breaking) => breaking.scene(),
                None => current.and_then(|current| current.line_breaking),
            },
        }
    }
}

/// The typography of a layer `add_text_layer` makes: the given one, black
/// text by default, or white where the page under the region is dark on
/// average (black ovals, night panels).
pub(super) async fn new_layer_typography(
    desktop: &Desktop,
    snapshot: &Snapshot,
    page: EntityId,
    [x, y, width, height]: [f64; 4],
    typography: Option<TrialTypography>,
) -> Result<Typography> {
    let mut typography = typography.unwrap_or_default().onto(None);
    if typography.color.is_none() {
        let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
        let (image_width, image_height) = clean.dimensions();
        let mut levels = Vec::new();
        for py in (y.max(0.0) as u32)..((y + height) as u32).min(image_height) {
            for px in (x.max(0.0) as u32)..((x + width) as u32).min(image_width) {
                let pixel = clean.get_pixel(px, py);
                levels.push(
                    (u32::from(pixel[0]) * 299
                        + u32::from(pixel[1]) * 587
                        + u32::from(pixel[2]) * 114)
                        / 1000,
                );
            }
        }
        // The mean tone, not the median: on a tinted balloon the screentone's
        // dots, or source lettering not cleaned yet, can make more than half
        // the pixels dark while the ground still reads light.
        let dark = !levels.is_empty()
            && levels.iter().map(|level| u64::from(*level)).sum::<u64>()
                < 100 * levels.len() as u64;
        typography.color = Some(if dark {
            [255, 255, 255, 255]
        } else {
            [0, 0, 0, 255]
        });
    }
    Ok(typography)
}

/// Places where the cleaning took lettering off the page and no shown,
/// translated layer covers it: cleaned text whose layer is missing or
/// hidden, which detection no longer sees once it is gone. A pixel counts as
/// removed lettering when source and clean page differ by more than
/// enhancement moves a tone (dark to light or light to dark across 90
/// levels); removed pixels within 6 px join into one place. Cleaning fills
/// removed lettering with flat tone, so a place counts only when nine tenths
/// of the clean page there sits near one tone; a redrawn page differs from
/// its source everywhere but keeps its texture. A thin strip along the page
/// edge is the scan border, not lettering.
pub(super) fn cleaned_without_text(
    source: &RgbaImage,
    clean: &RgbaImage,
    covered: &[[f64; 4]],
) -> Vec<[f64; 4]> {
    let (width, height) = source.dimensions();
    if clean.dimensions() != (width, height) {
        return Vec::new();
    }
    let (w, h) = (width as usize, height as usize);
    let luma = |pixel: &image::Rgba<u8>| {
        (u32::from(pixel[0]) * 299 + u32::from(pixel[1]) * 587 + u32::from(pixel[2]) * 114) / 1000
    };
    // A cleaned page a model returned whole is redrawn a pixel or two off the
    // source, so a pixel counts only when the clean page is far lighter (for
    // dark lettering) or darker (for light lettering) across its whole 5x5
    // neighbourhood.
    let before = source.pixels().map(luma).collect::<Vec<_>>();
    let after = clean.pixels().map(luma).collect::<Vec<_>>();
    let window = |values: &[u32], pick: fn(u32, u32) -> u32| {
        let mut rows = values.to_vec();
        for y in 0..h {
            for x in 0..w {
                let (from, to) = (x.saturating_sub(2), (x + 2).min(w - 1));
                rows[y * w + x] = (from..=to)
                    .map(|nx| values[y * w + nx])
                    .reduce(pick)
                    .unwrap();
            }
        }
        let mut out = rows.clone();
        for y in 0..h {
            for x in 0..w {
                let (from, to) = (y.saturating_sub(2), (y + 2).min(h - 1));
                out[y * w + x] = (from..=to).map(|ny| rows[ny * w + x]).reduce(pick).unwrap();
            }
        }
        out
    };
    let (lightest_near, darkest_near) = (window(&after, u32::min), window(&after, u32::max));
    let removed = (0..w * h)
        .map(|at| {
            (before[at] < 110 && lightest_near[at] > before[at] + 90)
                || (before[at] > 150 && darkest_near[at] + 90 < before[at])
        })
        .collect::<Vec<_>>();
    let near = chamfer(&removed, w, h);
    let joined = near
        .iter()
        .map(|distance| *distance <= 18)
        .collect::<Vec<_>>();
    let mut seen = vec![false; w * h];
    let mut places = Vec::new();
    for start in 0..w * h {
        if !joined[start] || seen[start] {
            continue;
        }
        let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
        let mut count = 0usize;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(at) = stack.pop() {
            let (x, y) = (at % w, at / w);
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
            count += usize::from(removed[at]);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h && joined[ny * w + nx] && !seen[ny * w + nx] {
                    seen[ny * w + nx] = true;
                    stack.push(ny * w + nx);
                }
            }
        }
        let place = [
            left as f64,
            top as f64,
            (right - left + 1) as f64,
            (bottom - top + 1) as f64,
        ];
        if count < 120 || place[2] < 12.0 || place[3] < 12.0 {
            continue;
        }
        let on_edge = left == 0 || top == 0 || right == w - 1 || bottom == h - 1;
        if on_edge && place[2].min(place[3]) < 50.0 {
            continue;
        }
        let mut tones = [0usize; 256];
        for y in top..=bottom {
            for x in left..=right {
                tones[after[y * w + x] as usize] += 1;
            }
        }
        let total = (right - left + 1) * (bottom - top + 1);
        let mut below = 0;
        let median = (0..256)
            .find(|tone| {
                below += tones[*tone];
                below * 2 >= total
            })
            .unwrap_or(0);
        let flat = tones[median.saturating_sub(23)..=(median + 23).min(255)]
            .iter()
            .sum::<usize>();
        if flat * 10 < total * 9 {
            continue;
        }
        let area = place[2] * place[3];
        let overlap = covered
            .iter()
            .map(|[x, y, cw, ch]| {
                let dx = (place[0] + place[2]).min(x + cw) - place[0].max(*x);
                let dy = (place[1] + place[3]).min(y + ch) - place[1].max(*y);
                dx.max(0.0) * dy.max(0.0)
            })
            .sum::<f64>();
        if overlap < area * 0.3 {
            places.push(place);
        }
    }
    places
}

/// The boxes of a page's shown, translated text layers, grown by 8 px.
pub(super) fn translated_boxes(snapshot: &Snapshot, page: EntityId) -> Result<Vec<[f64; 4]>> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    let group_shown = shown(snapshot, group.id())?;
    let mut boxes = Vec::new();
    for layer in group.text_layers()? {
        let translated = layer
            .content()?
            .translation()?
            .is_some_and(|translation| !translation.text.value.trim().is_empty());
        if !(group_shown && translated && shown(snapshot, layer.id())?) {
            continue;
        }
        if let Some((left, top, right, bottom)) = layer.frame()?.as_ref().and_then(bounds) {
            boxes.push([
                left - 8.0,
                top - 8.0,
                right - left + 16.0,
                bottom - top + 16.0,
            ]);
        }
    }
    Ok(boxes)
}

/// The box and slant of `layer`'s source lettering, for one line laid along
/// it: the detected region's long side gives the direction, so a Japanese
/// column turns 90 degrees and reads top to bottom.
/// The page images `changes` that orient along the source measure on, by page.
pub(super) async fn orient_sources(
    snapshot: &Snapshot,
    changes: &[TrialChange],
) -> Result<HashMap<EntityId, RgbaImage>> {
    let mut sources = HashMap::new();
    for change in changes.iter().filter(|change| change.orient.is_some()) {
        let page = owning_page(snapshot, address(snapshot, &change.element)?)?;
        if !sources.contains_key(&page) {
            sources.insert(page, base_image(snapshot, page).await?);
        }
    }
    Ok(sources)
}

fn along_source(
    snapshot: &Snapshot,
    layer: EntityId,
    sources: &HashMap<EntityId, RgbaImage>,
) -> Result<([f64; 4], f32)> {
    let geometry = snapshot
        .text_layer(layer)?
        .content()?
        .source_region()?
        .map(|region| region.geometry())
        .transpose()?
        .with_context(|| format!("{layer} has no detected source region"))?;
    let [a, b, c, ..] = geometry.points.as_slice() else {
        bail!("{layer}'s source region is not a box");
    };
    let (first, second) = ((b.x - a.x, b.y - a.y), (c.x - b.x, c.y - b.y));
    let (along, across) = if first.0.hypot(first.1) >= second.0.hypot(second.1) {
        (first, second)
    } else {
        (second, first)
    };
    let mut angle = along.1.atan2(along.0).to_degrees();
    // A line reads left to right or top to bottom, never upside down.
    if angle > 90.0 {
        angle -= 180.0;
    } else if angle <= -90.0 {
        angle += 180.0;
    }
    let (count, sum_x, sum_y) = geometry
        .points
        .iter()
        .fold((0.0, 0.0, 0.0), |(n, x, y), point| {
            (n + 1.0, x + point.x, y + point.y)
        });
    let (cx, cy) = (sum_x / count, sum_y / count);
    let (width, height) = (along.0.hypot(along.1), across.0.hypot(across.1));
    let angle = if angle.abs() < 3.0 { 0.0 } else { angle };
    // The detector's boxes are upright; the slant of lettering printed on a
    // tilted surface is in its pixels.
    if angle == 0.0
        && let Some(lines) = sources
            .get(&owning_page(snapshot, layer)?)
            .and_then(|image| {
                printed_lines(
                    image,
                    &geometry
                        .points
                        .iter()
                        .map(|point| [point.x, point.y])
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|lines| lines.strength >= MINIMUM_STRENGTH && lines.angle.abs() >= 1.0)
    {
        let [x, y, width, height] = lines.ink_box(lines.angle);
        // A lone column reads top to bottom.
        return Ok(if height > width * 1.5 {
            (
                [
                    x + (width - height) / 2.0,
                    y + (height - width) / 2.0,
                    height,
                    width,
                ]
                .map(f64::round),
                (lines.angle + 90.0) as f32,
            )
        } else {
            ([x, y, width, height].map(f64::round), lines.angle as f32)
        });
    }
    Ok((
        [cx - width / 2.0, cy - height / 2.0, width, height].map(f64::round),
        angle as f32,
    ))
}

/// The page's text layers in reading order: panels right to left in rows
/// from the top, layers inside a panel the same way, a layer outside every
/// panel taking its own place among them.
pub(super) fn reading_order(snapshot: &Snapshot, page: EntityId) -> Result<Vec<EntityId>> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    let mut layers = Vec::new();
    for layer in group.text_layers()? {
        let geometry = match layer.frame()? {
            Some(geometry) => Some(geometry),
            None => layer
                .content()?
                .source_region()?
                .map(|region| region.geometry())
                .transpose()?,
        };
        let Some((left, top, right, bottom)) = geometry.as_ref().and_then(bounds) else {
            continue;
        };
        layers.push((layer.id(), [left, top, right - left, bottom - top]));
    }
    let mut panels = Vec::new();
    for entity in snapshot.descendants(page)? {
        let is_panel = snapshot
            .component::<koharu_scene::Region>(entity.id())?
            .is_some_and(|region| {
                region.kind == <koharu_scene::PanelRegion as koharu_scene::RegionSpec>::kind()
            });
        if !is_panel {
            continue;
        }
        if let Some((left, top, right, bottom)) = snapshot
            .component::<Geometry>(entity.id())?
            .as_ref()
            .and_then(bounds)
        {
            panels.push([left, top, right - left, bottom - top]);
        }
    }
    let inside = |[x, y, width, height]: [f64; 4], [px, py, pw, ph]: [f64; 4]| {
        let (cx, cy) = (x + width / 2.0, y + height / 2.0);
        cx >= px && cx <= px + pw && cy >= py && cy <= py + ph
    };
    // Units: each panel with its layers, and each layer outside every panel.
    let mut units = panels
        .iter()
        .map(|panel| (*panel, Vec::new()))
        .collect::<Vec<([f64; 4], Vec<(EntityId, [f64; 4])>)>>();
    for (layer, frame) in layers {
        let holder = units
            .iter()
            .enumerate()
            .filter(|(_, (panel, _))| inside(frame, *panel))
            .min_by(|(_, (a, _)), (_, (b, _))| (a[2] * a[3]).total_cmp(&(b[2] * b[3])))
            .map(|(index, _)| index);
        match holder {
            Some(index) => units[index].1.push((layer, frame)),
            None => units.push((frame, vec![(layer, frame)])),
        }
    }
    let mut order = Vec::new();
    let unit_frames = units.iter().map(|(frame, _)| *frame).collect::<Vec<_>>();
    for unit in manga_order(&unit_frames) {
        let members = &units[unit].1;
        let frames = members.iter().map(|(_, frame)| *frame).collect::<Vec<_>>();
        order.extend(
            manga_order(&frames)
                .into_iter()
                .map(|index| members[index].0),
        );
    }
    Ok(order)
}

/// Boxes in manga order: rows from the top, a box joining a row when it
/// overlaps the row's band by half its height, right to left inside a row.
fn manga_order(boxes: &[[f64; 4]]) -> Vec<usize> {
    let mut by_top = (0..boxes.len()).collect::<Vec<_>>();
    by_top.sort_by(|a, b| boxes[*a][1].total_cmp(&boxes[*b][1]));
    let mut rows: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for index in by_top {
        let [_, top, _, height] = boxes[index];
        let bottom = top + height;
        match rows.last_mut() {
            Some((row_top, row_bottom, members))
                if bottom.min(*row_bottom) - top.max(*row_top)
                    >= 0.5 * height.min(*row_bottom - *row_top) =>
            {
                *row_top = row_top.min(top);
                *row_bottom = row_bottom.max(bottom);
                members.push(index);
            }
            _ => rows.push((top, bottom, vec![index])),
        }
    }
    rows.into_iter()
        .flat_map(|(_, _, mut members)| {
            members.sort_by(|a, b| {
                (boxes[*b][0] + boxes[*b][2]).total_cmp(&(boxes[*a][0] + boxes[*a][2]))
            });
            members
        })
        .collect()
}

/// Drops `layer`'s own box and binds it to the balloon that holds it, so the
/// renderer lays it out in the balloon's shape.
fn balloon_flow_step(snapshot: &Snapshot, layer: EntityId) -> Result<EditStep> {
    let text = snapshot.text_layer(layer)?;
    let balloon = match text.balloon_target()? {
        Some(balloon) => balloon.id(),
        None => {
            let page = owning_page(snapshot, layer)?;
            let geometry = text
                .content()?
                .source_region()?
                .map(|region| region.geometry())
                .transpose()?
                .or(text.frame()?)
                .with_context(|| {
                    format!(
                        "{} has no box to find its balloon by",
                        named(snapshot, layer)
                    )
                })?;
            Project::containing_bubble(snapshot, page, &geometry)?.with_context(|| {
                format!(
                    "no detected balloon holds {}: give it a region instead of flow balloon",
                    named(snapshot, layer)
                )
            })?
        }
    };
    let bound = text.balloon_target()?.is_some();
    let fits = snapshot
        .relations_from_as::<koharu_scene::FitsTo>(layer)
        .map(|relation| relation.id())
        .collect::<Vec<_>>();
    let content = text.content()?.id();
    Ok(Box::new(move |edit| {
        edit.promote_entity_to_user(layer)?;
        edit.promote_entity_to_user(content)?;
        edit.remove::<Geometry>(layer)?;
        for relation in fits {
            edit.remove_relation(relation)?;
        }
        if !bound {
            edit.relate::<koharu_scene::FlowsIn>(layer, balloon)?;
        }
        Ok(())
    }))
}

/// A page or layer an agent names: its id, a page number (`45`), or a page
/// and layer number (`45#13`) as chapter_text and page_layers count them.
pub(super) fn address(snapshot: &Snapshot, value: &str) -> Result<EntityId> {
    if let Ok(id) = entity(value) {
        return Ok(id);
    }
    let (page, layer) = match value.trim().split_once('#') {
        Some((page, layer)) => (page, Some(layer)),
        None => (value.trim(), None),
    };
    let number = |text: &str| {
        text.trim()
            .parse::<usize>()
            .ok()
            .filter(|number| *number > 0)
            .with_context(|| format!("{value} is neither an id nor a page#layer address"))
    };
    let page = snapshot
        .pages()
        .nth(number(page)? - 1)
        .with_context(|| format!("the chapter has no page {page}"))?
        .id();
    let Some(layer) = layer else {
        return Ok(page);
    };
    let index = number(layer)? - 1;
    snapshot
        .page(page)?
        .text_group()?
        .and_then(|group| group.text_layers().ok()?.nth(index))
        .map(|layer| layer.id())
        .with_context(|| format!("page {value} has no such layer"))
}

/// The elements the changes name, in order.
pub(super) fn elements(snapshot: &Snapshot, changes: &[TrialChange]) -> Result<Vec<EntityId>> {
    changes
        .iter()
        .map(|change| address(snapshot, &change.element))
        .collect()
}

/// The changes as checked edit steps against `snapshot`: the same steps the
/// trial render previews and `apply_layers` commits as one edit.
pub(super) fn steps(
    snapshot: &Snapshot,
    changes: Vec<TrialChange>,
    sources: &HashMap<EntityId, RgbaImage>,
    laid_out_size: impl Fn(EntityId) -> Option<f32>,
) -> Result<Vec<EditStep>> {
    let mut steps = Vec::new();
    for change in changes {
        let layer = address(snapshot, &change.element)?;
        let name = named(snapshot, layer);
        let text = match (change.text, change.lines) {
            (Some(_), Some(_)) => bail!("give {name} either text or lines, not both"),
            (text, lines) => text.or_else(|| lines.map(|lines| lines.join("\n"))),
        };
        if let Some(text) = text {
            steps.push(Project::translation_step(snapshot, layer, Some(text))?);
        }
        if let Some(source) = change.source {
            steps.push(Project::source_step(snapshot, layer, source)?);
        } else if change.confirm_source {
            let source = snapshot
                .text_layer(layer)?
                .content()?
                .source()?
                .map(|source| source.text.value)
                .with_context(|| format!("{name} has no source text to confirm"))?;
            steps.push(Project::source_step(snapshot, layer, source)?);
        }
        if change.flow == Some(TrialFlow::Balloon) {
            if change.region.is_some() || change.corners.is_some() {
                bail!("give {name} either a region, corners or flow balloon, not one with another");
            }
            steps.push(balloon_flow_step(snapshot, layer)?);
        }
        if let Some(corners) = change.corners {
            if change.region.is_some() {
                bail!("give {name} either corners or a region, not both");
            }
            let corners = match (change.angle, change.orient) {
                (Some(_), Some(_)) => {
                    bail!("give {name}'s corners either an angle or orient, not both")
                }
                (Some(turn), None) => Surface::new(corners)
                    .with_context(|| format!("{name}'s corners make no convex quadrilateral"))?
                    .turned(f64::from(turn)),
                (None, Some(TrialOrient::AlongSource)) => {
                    let lines = sources
                        .get(&owning_page(snapshot, layer)?)
                        .and_then(|image| printed_lines(image, &corners))
                        .filter(|lines| lines.strength >= MINIMUM_STRENGTH)
                        .with_context(|| {
                            format!("no printed lines were measured inside {name}'s corners")
                        })?;
                    let surface = Surface::new(corners).with_context(|| {
                        format!("{name}'s corners make no convex quadrilateral")
                    })?;
                    surface.turned(surface.turn_to(lines.angle))
                }
                (None, None) => corners,
            };
            let (frame, unit) = frame_from_corners(corners)
                .with_context(|| format!("{name}'s corners make no convex quadrilateral"))?;
            let geometry = Project::geometry_from_frame(frame)?;
            let mut layout = snapshot
                .component::<SceneTextLayout>(layer)?
                .context("only text layers take corners")?;
            layout.origin = Origin::User;
            layout.angle_degrees = Some(frame.angle_degrees);
            layout.perspective = Some(unit);
            if !koharu_scene::valid_perspective(layout.perspective.as_deref().unwrap_or_default()) {
                bail!("{name}'s corners make no convex quadrilateral");
            }
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(layer)?;
                edit.set(layer, &geometry)?;
                edit.set(layer, &layout)
            }));
        }
        let (region, angle) = match change.orient {
            _ if change.corners.is_some() => (None, None),
            Some(TrialOrient::AlongSource) => {
                if change.region.is_some() || change.angle.is_some() {
                    bail!("give {name} either orient or a region and angle, not both");
                }
                let (region, angle) = along_source(snapshot, layer, sources)?;
                (Some(region), Some(angle))
            }
            None => (change.region, change.angle),
        };
        if let Some(angle) = angle {
            let [x, y, width, height] =
                region.with_context(|| format!("turning {name} needs its region"))?;
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
            layout.perspective = None;
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(layer)?;
                edit.set(layer, &geometry)?;
                edit.set(layer, &layout)
            }));
        } else if let Some([x, y, width, height]) = region {
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
            let typography = typography.onto(
                snapshot
                    .component::<koharu_scene::Typography>(layer)?
                    .as_ref(),
            );
            steps.push(Project::typography_step(
                snapshot,
                vec![TypographyUpdate { layer, typography }],
                &laid_out_size,
            )?);
        }
        if let Some(role) = change.role {
            let content = snapshot.text_layer(layer)?.content()?.id();
            steps.push(Box::new(move |edit| {
                edit.promote_entity_to_user(content)?;
                edit.set(
                    content,
                    &koharu_scene::TextRole {
                        origin: Origin::User,
                        role: role.kind().to_owned(),
                    },
                )
            }));
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
    full: bool,
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
        let clean = render_image(desktop, &without_text(snapshot, page)?, page).await?;
        let shapes = koharu_renderer::balloon_flow_shapes(snapshot, page)?;
        let numbers = numbering(snapshot, page)?;
        let source = base_image(snapshot, page).await?;
        for layer in members {
            let mut value = layer_value(snapshot, &frame, layer, numbers(layer))?;
            measure_balloon(&clean, &mut value, &shapes);
            measure_source_lines(snapshot, &source, layer, &mut value)?;
            if !full {
                value = compact_layer(&value);
            }
            value["page"] = json!(page);
            values.push(value);
        }
    }
    Ok(Value::Array(values))
}

pub(super) fn image_url(png: String) -> String {
    format!("data:image/png;base64,{png}")
}

/// The page's shown, translated text layers whose font and colour nobody
/// chose: typography from processing (or none), or a font and colour both
/// left unset. They draw in the kind's font and the colour sampled from the
/// original lettering, often grey; the export wants them set.
fn automatic_style_layers(snapshot: &Snapshot, page: EntityId) -> Result<Vec<EntityId>> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    if !shown(snapshot, group.id())? {
        return Ok(Vec::new());
    }
    let mut layers = Vec::new();
    for layer in group.text_layers()? {
        let translated = layer
            .content()?
            .translation()?
            .is_some_and(|translation| !translation.text.value.trim().is_empty());
        if !translated || !shown(snapshot, layer.id())? {
            continue;
        }
        let automatic = snapshot
            .component::<koharu_scene::Typography>(layer.id())?
            .is_none_or(|typography| {
                matches!(typography.origin, Origin::Generated(_))
                    || (typography.preferred_font.is_none() && typography.color.is_none())
            });
        if automatic {
            layers.push(layer.id());
        }
    }
    Ok(layers)
}

/// Every text layer's address, 45#13, by layer.
pub(super) fn addresses(snapshot: &Snapshot) -> Result<HashMap<EntityId, String>> {
    let mut addresses = HashMap::new();
    for (index, page) in snapshot.pages().enumerate() {
        if let Some(group) = page.text_group()? {
            for (number, layer) in group.text_layers()?.enumerate() {
                addresses.insert(layer.id(), format!("{}#{}", index + 1, number + 1));
            }
        }
    }
    Ok(addresses)
}

/// A layer's address as the tools take it, 45#13, for messages; its id when
/// it has none.
fn named(snapshot: &Snapshot, layer: EntityId) -> String {
    let address = || -> Result<String> {
        let page = owning_page(snapshot, layer)?;
        let page_number = snapshot
            .pages()
            .position(|candidate| candidate.id() == page)
            .context("the layer's page is not in the project")?
            + 1;
        Ok(format!(
            "{page_number}#{}",
            numbering(snapshot, page)?(layer)
        ))
    };
    address().unwrap_or_else(|_| layer.to_string())
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
    let angle = text
        .layout()?
        .angle_degrees
        .filter(|angle| angle.abs() > f32::EPSILON);
    // A turned box is reported as the box that was turned, with its angle,
    // so it can be sent back as it came.
    let frame_box = text
        .frame()?
        .as_ref()
        .and_then(|geometry| unturned(geometry, angle.unwrap_or(0.0)));
    let page_width = snapshot.page(owning_page(snapshot, layer)?)?.page()?.width;
    let ocr = match content.source_region()? {
        Some(region) => region.ocr()?.map(|analysis| {
            json!({
                "confidence": analysis.confidence,
                "unverified": koharu_pipeline::ocr_unverified(content).unwrap_or(false),
            })
        }),
        None => None,
    };
    Ok(json!({
        "id": layer,
        "number": number,
        "visible": visible,
        "hidden_by": if !visible {
            Some("layer")
        } else if text_group_hidden(snapshot, owning_page(snapshot, layer)?)? {
            Some("text_group")
        } else {
            None
        },
        "role": content.role()?.map(|role| role.role),
        "source_text": content.source()?.map(|source| source.text.value),
        "ocr": ocr,
        "translated_text": content.translation()?.map(|translation| translation.text.value),
        "box": frame_box,
        "angle": angle,
        "corners": text
            .layout()?
            .perspective
            .as_deref()
            .zip(frame_box)
            .map(|(unit, frame_box)| corners_on_page(frame_box, angle.unwrap_or(0.0), unit)),
        "flow": if snapshot.component::<Geometry>(layer)?.is_none() && balloon.is_some() {
            "balloon"
        } else {
            "box"
        },
        "balloon": balloon.as_ref().map(|geometry| json!({
            "box": extent(geometry),
            "inner_box": inner_box(geometry),
        })),
        "typography": typography,
        "rendered": rendered.map(|rendered| {
            let bounds = rendered.layout_bounds;
            json!({
                "font": rendered.post_script_fonts.first(),
                "font_size": (f64::from(rendered.font_size) * 10.0).round() / 10.0,
                "auto_fit_max": rendered.auto_fit_ceiling.map(|size| (f64::from(size) * 10.0).round() / 10.0),
                "fits_up_to": rendered.fits_up_to.map(|size| (f64::from(size) * 10.0).round() / 10.0),
                "small": f64::from(rendered.font_size) < page_width * SMALL_TEXT_SHARE,
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
                "readable_color": frame.diagnostics().iter().find_map(|diagnostic| match diagnostic {
                    RenderDiagnostic::TextLowContrast { entity, readable, .. } if *entity == layer => {
                        Some(*readable)
                    }
                    _ => None,
                }),
                "overflow": overflow(frame.diagnostics(), layer, angle.and(frame_box)),
            })
        }),
    }))
}

/// Text below this share of the page width is flagged `small`: 15 px on a
/// 1280 px page, under the owner's smallest hand-set dialogue.
const SMALL_TEXT_SHARE: f64 = 0.012;

/// The box `geometry` is once turned back by `angle` degrees about its centre.
/// The frame a surface's page `corners` (top-left, top-right, bottom-right,
/// bottom-left) are seen in, and the corners in units of that frame: the frame
/// is centred on the corners, turned like the surface's top and bottom edges
/// on average, and as large as its edges on average, so text laid out in it
/// keeps the surface's own proportions.
fn frame_from_corners(corners: [[f64; 2]; 4]) -> Option<(TextFrame, Vec<koharu_scene::Point>)> {
    let [top_left, top_right, bottom_right, bottom_left] = corners;
    let edge = |from: [f64; 2], to: [f64; 2]| [to[0] - from[0], to[1] - from[1]];
    let length = |[x, y]: [f64; 2]| x.hypot(y);
    let (top, bottom) = (edge(top_left, top_right), edge(bottom_left, bottom_right));
    let (left, right) = (edge(top_left, bottom_left), edge(top_right, bottom_right));
    let (top_length, bottom_length) = (length(top), length(bottom));
    if !(top_length > 0.0 && bottom_length > 0.0) {
        return None;
    }
    let direction = [
        top[0] / top_length + bottom[0] / bottom_length,
        top[1] / top_length + bottom[1] / bottom_length,
    ];
    let angle = direction[1].atan2(direction[0]);
    let width = (top_length + bottom_length) * 0.5;
    let height = (length(left) + length(right)) * 0.5;
    if !(width > 1.0 && height > 1.0) {
        return None;
    }
    let centre = [
        corners.iter().map(|point| point[0]).sum::<f64>() * 0.25,
        corners.iter().map(|point| point[1]).sum::<f64>() * 0.25,
    ];
    let (sin, cos) = angle.sin_cos();
    let unit = corners
        .iter()
        .map(|point| {
            let (x, y) = (point[0] - centre[0], point[1] - centre[1]);
            koharu_scene::Point {
                x: (x * cos + y * sin) / width + 0.5,
                y: (-x * sin + y * cos) / height + 0.5,
            }
        })
        .collect();
    let frame = TextFrame {
        x: (centre[0] - width * 0.5) as f32,
        y: (centre[1] - height * 0.5) as f32,
        width: width as f32,
        height: height as f32,
        angle_degrees: angle.to_degrees() as f32,
    };
    Some((frame, unit))
}

/// A layer's frame drawn in perspective onto page corners: where the frame's
/// points land on the page.
struct Surface {
    frame: TextFrame,
    perspective: koharu_renderer::Perspective,
}

impl Surface {
    fn new(corners: [[f64; 2]; 4]) -> Option<Self> {
        let (frame, unit) = frame_from_corners(corners)?;
        if !koharu_scene::valid_perspective(&unit) {
            return None;
        }
        let unit = [0, 1, 2, 3].map(|index| (unit[index].x as f32, unit[index].y as f32));
        Some(Self {
            perspective: koharu_renderer::Perspective::new(frame.width, frame.height, unit),
            frame,
        })
    }

    fn page(&self, x: f64, y: f64) -> [f64; 2] {
        let (width, height) = (f64::from(self.frame.width), f64::from(self.frame.height));
        let (x, y) = self.perspective.apply(x, y);
        let (dx, dy) = (x - width * 0.5, y - height * 0.5);
        let (sin, cos) = f64::from(self.frame.angle_degrees).to_radians().sin_cos();
        [
            f64::from(self.frame.x) + width * 0.5 + dx * cos - dy * sin,
            f64::from(self.frame.y) + height * 0.5 + dx * sin + dy * cos,
        ]
    }

    /// The page slant, degrees clockwise, of text lines turned by `turn`
    /// degrees on the surface, at its centre.
    fn slant(&self, turn: f64) -> f64 {
        let (cx, cy) = (
            f64::from(self.frame.width) * 0.5,
            f64::from(self.frame.height) * 0.5,
        );
        let (sin, cos) = turn.to_radians().sin_cos();
        let [x0, y0] = self.page(cx, cy);
        let [x1, y1] = self.page(cx + cos, cy + sin);
        (y1 - y0).atan2(x1 - x0).to_degrees()
    }

    /// The turn on the surface that sets text lines at page `slant`.
    fn turn_to(&self, slant: f64) -> f64 {
        let mut turn = 0.0;
        for _ in 0..24 {
            let miss = (slant - self.slant(turn) + 180.0).rem_euclid(360.0) - 180.0;
            if miss.abs() < 0.01 {
                break;
            }
            turn += miss;
        }
        turn
    }

    /// The page corners of the frame turned by `turn` degrees about its
    /// centre on the surface.
    fn turned(&self, turn: f64) -> [[f64; 2]; 4] {
        let (width, height) = (f64::from(self.frame.width), f64::from(self.frame.height));
        let (sin, cos) = turn.to_radians().sin_cos();
        [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(u, v)| {
            let (x, y) = (u * width, v * height);
            self.page(
                width * 0.5 + x * cos - y * sin,
                height * 0.5 + x * sin + y * cos,
            )
        })
    }
}

/// Records on `value` the slant of the lettering printed under `layer` on
/// `source` (inside its corners, or its detected source region), the slant
/// its text is drawn at and the difference, wrapped to a quarter turn.
fn measure_source_lines(
    snapshot: &Snapshot,
    source: &RgbaImage,
    layer: EntityId,
    value: &mut Value,
) -> Result<()> {
    let corners = serde_json::from_value::<[[f64; 2]; 4]>(value["corners"].clone()).ok();
    let polygon = match corners {
        Some(corners) => corners.to_vec(),
        None => match snapshot.text_layer(layer)?.content()?.source_region()? {
            Some(region) => region
                .geometry()?
                .points
                .iter()
                .map(|point| [point.x, point.y])
                .collect(),
            None => return Ok(()),
        },
    };
    let Some(lines) =
        printed_lines(source, &polygon).filter(|lines| lines.strength >= MINIMUM_STRENGTH)
    else {
        return Ok(());
    };
    let text = match corners.and_then(Surface::new) {
        Some(surface) => surface.slant(0.0),
        None => value["angle"].as_f64().unwrap_or(0.0),
    };
    let delta = (text - lines.angle + 45.0).rem_euclid(90.0) - 45.0;
    value["source_lines"] = json!({
        "angle": lines.angle,
        "along": match lines.along {
            Along::Rows => "rows",
            Along::Columns => "columns",
        },
        "text_angle": round_pixel(text),
        "delta": round_pixel(delta),
        // Below about 1.5 the print is faint or partly covered (a finger on
        // a phone, art across a sign): check the slant by eye.
        "strength": (lines.strength * 100.0).round() / 100.0,
    });
    Ok(())
}

/// The page corners of a layer drawn in perspective: `unit` corners placed
/// by its unturned `frame_box` turned by `angle` about its centre.
fn corners_on_page(frame_box: [f64; 4], angle: f32, unit: &[koharu_scene::Point]) -> Vec<[f64; 2]> {
    let [x, y, width, height] = frame_box;
    let centre = [x + width * 0.5, y + height * 0.5];
    let (sin, cos) = f64::from(angle).to_radians().sin_cos();
    unit.iter()
        .map(|point| {
            let (dx, dy) = ((point.x - 0.5) * width, (point.y - 0.5) * height);
            [
                round_pixel(centre[0] + dx * cos - dy * sin),
                round_pixel(centre[1] + dx * sin + dy * cos),
            ]
        })
        .collect()
}

fn round_pixel(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn unturned(geometry: &Geometry, angle: f32) -> Option<[f64; 4]> {
    let (sin, cos) = f64::from(angle).to_radians().sin_cos();
    let points = geometry
        .points
        .iter()
        .map(|point| {
            (
                point.x * cos + point.y * sin,
                -point.x * sin + point.y * cos,
            )
        })
        .collect::<Vec<_>>();
    let (left, top, right, bottom) = points.iter().fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(l, t, r, b), (x, y)| (l.min(*x), t.min(*y), r.max(*x), b.max(*y)),
    );
    if !(right > left && bottom > top) {
        return None;
    }
    let (cx, cy) = ((left + right) / 2.0, (top + bottom) / 2.0);
    let (width, height) = (right - left, bottom - top);
    let (x, y) = (cx * cos - cy * sin, cx * sin + cy * cos);
    Some([x - width / 2.0, y - height / 2.0, width, height].map(f64::round))
}

/// The short form of a layer value for apply and preview results: what was
/// drawn and what went wrong.
fn compact_layer(value: &Value) -> Value {
    let rendered = &value["rendered"];
    let mut compact = serde_json::Map::new();
    for (key, field) in [
        ("layer", &value["number"]),
        ("id", &value["id"]),
        ("box", &value["box"]),
        ("angle", &value["angle"]),
        ("size", &rendered["font_size"]),
        ("auto_fit_max", &rendered["auto_fit_max"]),
        ("warning", &rendered["warning"]),
        ("readable_color", &rendered["readable_color"]),
        ("overflow", &rendered["overflow"]),
        ("flow", &value["flow"]),
        ("inner_box", &value["balloon"]["inner_box"]),
        ("off_centre", &value["balloon"]["off_centre"]),
        ("edge_gap", &value["balloon"]["edge_gap"]),
        ("fits_up_to", &rendered["fits_up_to"]),
        ("corners", &value["corners"]),
        ("source_lines", &value["source_lines"]),
    ] {
        if !field.is_null() {
            compact.insert(key.to_owned(), field.clone());
        }
    }
    if rendered["small"] == json!(true) {
        compact.insert("small".to_owned(), json!(true));
    }
    if let Some(lines) = rendered["lines"].as_array() {
        compact.insert(
            "lines".to_owned(),
            Value::Array(lines.iter().map(|line| line["text"].clone()).collect()),
        );
    }
    if rendered["hyphenated"]
        .as_array()
        .is_some_and(|words| !words.is_empty())
    {
        compact.insert("hyphenated".to_owned(), rendered["hyphenated"].clone());
    }
    Value::Object(compact)
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

/// What overflowed: the room the layout had and the size it drew, both in
/// the text's own axes. A turned layer's room is its unturned box: the
/// renderer reports the turned room's page bounds.
fn overflow(
    diagnostics: &[RenderDiagnostic],
    layer: EntityId,
    turned_box: Option<[f64; 4]>,
) -> Option<Value> {
    diagnostics.iter().find_map(|diagnostic| match diagnostic {
        RenderDiagnostic::TextOverflow {
            entity,
            available,
            actual_width,
            actual_height,
            ..
        } if *entity == layer => Some(json!({
            "room": turned_box.map_or_else(
                || [available.width, available.height].map(|value| f64::from(value).round()),
                |[_, _, width, height]| [width, height],
            ),
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

/// The largest axis-aligned rectangle inside the balloon around the text, as
/// drawn on the clean page; see `balloon_interior`.
fn pixel_inner_box(clean: &RgbaImage, balloon: [f64; 4], text: [f64; 4]) -> Option<[f64; 4]> {
    balloon_interior(clean, balloon, text, None).map(|interior| interior.inner_box)
}

/// The inside of a balloon as drawn on the clean page, for the text whose
/// box is given.
struct Interior {
    /// Largest rectangle in the lobe holding its centre, kept a margin from
    /// the outline.
    inner_box: [f64; 4],
    /// Bounds and centroid of the lobe without its tail and bumps.
    bounds: [f64; 4],
    centre: [f64; 2],
    left: u32,
    top: u32,
    width: usize,
    lobe: Vec<bool>,
}

impl Interior {
    /// The lobe's leftmost and rightmost pixel on page row `y`.
    fn row(&self, y: f64) -> Option<(f64, f64)> {
        let row = (y - f64::from(self.top)).round();
        if row < 0.0 {
            return None;
        }
        let start = row as usize * self.width;
        let cells = self.lobe.get(start..start + self.width)?;
        let first = cells.iter().position(|inside| *inside)?;
        let last = cells.iter().rposition(|inside| *inside)?;
        Some((
            f64::from(self.left) + first as f64,
            f64::from(self.left) + last as f64 + 1.0,
        ))
    }
}

/// Whether the point lies inside the polygon (even-odd rule).
fn polygon_contains(polygon: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut inside = false;
    let Some(mut previous) = polygon.last().copied() else {
        return false;
    };
    for &current in polygon {
        if (current.1 > y) != (previous.1 > y)
            && x < (previous.0 - current.0) * (y - current.1) / (previous.1 - current.1) + current.0
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

/// Chamfer (3-4) distance, in thirds of a pixel, from each pixel to the
/// nearest `source` pixel.
fn chamfer(sources: &[bool], width: usize, height: usize) -> Vec<u32> {
    const FAR: u32 = u32::MAX / 2;
    let mut distance = sources
        .iter()
        .map(|source| if *source { 0 } else { FAR })
        .collect::<Vec<_>>();
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            let mut best = distance[at];
            if x > 0 {
                best = best.min(distance[at - 1] + 3);
            }
            if y > 0 {
                best = best.min(distance[at - width] + 3);
                if x > 0 {
                    best = best.min(distance[at - width - 1] + 4);
                }
                if x + 1 < width {
                    best = best.min(distance[at - width + 1] + 4);
                }
            }
            distance[at] = best;
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let at = y * width + x;
            let mut best = distance[at];
            if x + 1 < width {
                best = best.min(distance[at + 1] + 3);
            }
            if y + 1 < height {
                best = best.min(distance[at + width] + 3);
                if x + 1 < width {
                    best = best.min(distance[at + width + 1] + 4);
                }
                if x > 0 {
                    best = best.min(distance[at + width - 1] + 4);
                }
            }
            distance[at] = best;
        }
    }
    distance
}

/// The light area flooded from the light pixel nearest the text box's
/// centre, its holes filled. An opening by 15% of its shorter side takes off
/// the tail and small bumps and parts the lobes of a joined balloon at their
/// neck; the lobe holding (or nearest) the text is measured. `None` when the
/// flood leaks out of the balloon's box (an open balloon) or finds no light
/// pixel, so the detected outline decides instead.
fn balloon_interior(
    clean: &RgbaImage,
    balloon: [f64; 4],
    text: [f64; 4],
    shape: Option<&Geometry>,
) -> Option<Interior> {
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
    // Open the inside and keep the lobe of the text.
    let mut inside = outside.iter().map(|outside| !outside).collect::<Vec<_>>();
    if let Some(shape) = shape {
        let polygon = shape
            .points
            .iter()
            .map(|point| (point.x, point.y))
            .collect::<Vec<_>>();
        for (at, cell) in inside.iter_mut().enumerate() {
            if *cell {
                let (x, y) = (at % width, at / width);
                *cell = polygon_contains(
                    &polygon,
                    f64::from(left) + x as f64 + 0.5,
                    f64::from(top) + y as f64 + 0.5,
                );
            }
        }
    }
    let outside = inside.iter().map(|inside| !inside).collect::<Vec<_>>();
    let (inside_min_x, inside_max_x, inside_min_y, inside_max_y) = (0..width * height)
        .filter(|at| inside[*at])
        .fold((width, 0, height, 0), |(a, b, c, d), at| {
            let (x, y) = (at % width, at / width);
            (a.min(x), b.max(x), c.min(y), d.max(y))
        });
    if inside_max_x <= inside_min_x || inside_max_y <= inside_min_y {
        return None;
    }
    let radius = (((inside_max_x - inside_min_x).min(inside_max_y - inside_min_y) as f64 * 0.15)
        .max(2.0)
        * 3.0) as u32;
    let to_outside = chamfer(&outside, width, height);
    let eroded = to_outside
        .iter()
        .map(|distance| *distance > radius)
        .collect::<Vec<_>>();
    let to_eroded = chamfer(&eroded, width, height);
    let opened = (0..width * height)
        .map(|at| inside[at] && to_eroded[at] <= radius)
        .collect::<Vec<_>>();
    let mut component = vec![usize::MAX; width * height];
    let mut components = Vec::<Vec<usize>>::new();
    for start in 0..width * height {
        if !opened[start] || component[start] != usize::MAX {
            continue;
        }
        let index = components.len();
        let mut pixels = vec![start];
        component[start] = index;
        let mut cursor = 0;
        while cursor < pixels.len() {
            let at = pixels[cursor];
            cursor += 1;
            let (x, y) = (at % width, at / width);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < width && ny < height {
                    let next = ny * width + nx;
                    if opened[next] && component[next] == usize::MAX {
                        component[next] = index;
                        pixels.push(next);
                    }
                }
            }
        }
        components.push(pixels);
    }
    let seed_at = seed.1 * width + seed.0;
    let lobe_pixels = if components.is_empty() {
        (0..width * height)
            .filter(|at| inside[*at])
            .collect::<Vec<_>>()
    } else if component[seed_at] != usize::MAX {
        components.swap_remove(component[seed_at])
    } else {
        let nearest = components
            .iter()
            .enumerate()
            .min_by_key(|(_, pixels)| {
                pixels
                    .iter()
                    .map(|at| {
                        let (x, y) = (at % width, at / width);
                        x.abs_diff(seed.0).pow(2) + y.abs_diff(seed.1).pow(2)
                    })
                    .min()
                    .unwrap_or(usize::MAX)
            })
            .map(|(index, _)| index)?;
        components.swap_remove(nearest)
    };
    let mut lobe = vec![false; width * height];
    for at in &lobe_pixels {
        lobe[*at] = true;
    }
    let count = lobe_pixels.len().max(1) as f64;
    let (sum_x, sum_y) = lobe_pixels.iter().fold((0.0, 0.0), |(sx, sy), at| {
        (sx + (at % width) as f64, sy + (at / width) as f64)
    });
    let centroid = (
        (sum_x / count).round() as usize,
        (sum_y / count).round() as usize,
    );
    let outside = lobe.iter().map(|inside| !inside).collect::<Vec<_>>();
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
    // The largest rectangle holding the lobe's centroid, when one does: the
    // text centres on the lobe, not on where the source lettering sat.
    let mut heights = vec![0usize; width];
    let mut best = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut around_seed = (0usize, 0usize, 0usize, 0usize, 0usize);
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
                let rect = (area, start, y + 1 - heights[last], x - start, heights[last]);
                if area > best.0 {
                    best = rect;
                }
                let holds_seed = (start..x).contains(&centroid.0)
                    && (y + 1 - heights[last]..=y).contains(&centroid.1);
                if holds_seed && area > around_seed.0 {
                    around_seed = rect;
                }
            }
            stack.push(x);
        }
    }
    if around_seed.0 > 0 {
        best = around_seed;
    }
    let (area, x, y, rect_width, rect_height) = best;
    if area == 0 {
        return None;
    }
    let (left_f, top_f) = (f64::from(left), f64::from(top));
    Some(Interior {
        inner_box: [
            left_f + x as f64,
            top_f + y as f64,
            rect_width as f64,
            rect_height as f64,
        ],
        bounds: [
            left_f + min_x as f64,
            top_f + min_y as f64,
            (max_x - min_x + 1) as f64,
            (max_y - min_y + 1) as f64,
        ],
        centre: [
            (left_f + sum_x / count).round(),
            (top_f + sum_y / count).round(),
        ],
        left,
        top,
        width,
        lobe,
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

/// The page the cleaning worked on: its visible scan enhancement, or the
/// imported source. Comparing the clean page with it shows what the cleaning
/// removed, not what enhancement moved.
async fn base_image(snapshot: &Snapshot, page: EntityId) -> Result<RgbaImage> {
    let role = AssetRole::new("source")?;
    for entity in snapshot.children(page)? {
        let enhancement = snapshot
            .component::<koharu_scene::RasterLayer>(entity)?
            .is_some_and(|layer| layer.kind == koharu_scene::RasterLayerKind::Enhancement);
        if enhancement && shown(snapshot, entity)? {
            if let Some(asset) = snapshot.asset(entity, &role)? {
                return Ok(
                    image::load_from_memory(&snapshot.read_blob(asset.blob).await?)?.to_rgba8(),
                );
            }
        }
    }
    source_image(snapshot, page).await
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
const FLOW_COLOR: [u8; 4] = [0, 180, 60, 255];
const UNBOXED_COLOR: [u8; 4] = [255, 140, 0, 255];

/// Draws each of `layers`' box in magenta with its number, the inside of its
/// balloon as measured for `page_layers` in blue, and the shape a balloon
/// flow lays it out in (its lobe of a joined balloon) in green. Strokes and
/// digits are sized to stay legible at `scale`.
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
    let shapes = koharu_renderer::balloon_flow_shapes(snapshot, page)?;
    let stroke = (2.0 / scale).ceil().max(1.0) as u32;
    let cell = (3.0 / scale).ceil().max(2.0) as u32;
    for layer in layers {
        let number = numbers(*layer);
        let mut value = layer_value(snapshot, frame, *layer, number)?;
        measure_balloon(clean, &mut value, &shapes);
        if value["flow"] == "balloon"
            && let Some(shape) = shapes.get(layer)
        {
            draw_polygon(image, &shape.points, stroke, FLOW_COLOR);
        }
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

fn draw_polygon(
    image: &mut RgbaImage,
    points: &[koharu_scene::Point],
    stroke: u32,
    color: [u8; 4],
) {
    let (image_width, image_height) = image.dimensions();
    let half = f64::from(stroke) / 2.0;
    for (index, start) in points.iter().enumerate() {
        let end = points[(index + 1) % points.len()];
        let steps = (end.x - start.x)
            .abs()
            .max((end.y - start.y).abs())
            .ceil()
            .max(1.0) as usize;
        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let (x, y) = (
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
            );
            for py in (y - half).max(0.0) as u32..((y + half).ceil() as u32).min(image_height) {
                for px in (x - half).max(0.0) as u32..((x + half).ceil() as u32).min(image_width) {
                    image.put_pixel(px, py, image::Rgba(color));
                }
            }
        }
    }
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
    fn outlined_lettering_names_its_fill_and_outline() {
        // Dark strokes ringed in white over a mid-grey screentone, and the
        // same strokes plain on white paper; the cleaning leaves the ground.
        let page = |ground: u8, outline: bool| {
            RgbaImage::from_fn(120, 60, |x, y| {
                let stroke = (20..100).contains(&x) && (y % 20) >= 6 && (y % 20) < 14;
                let ring =
                    outline && (16..104).contains(&x) && (y % 20) >= 2 && (y % 20) < 18 && !stroke;
                image::Rgba(if stroke {
                    [10, 10, 10, 255]
                } else if ring {
                    [250, 250, 250, 255]
                } else {
                    [ground, ground, ground, 255]
                })
            })
        };
        let clean =
            |ground: u8| RgbaImage::from_pixel(120, 60, image::Rgba([ground, ground, ground, 255]));
        let outlined = lettering(&page(130, true), &clean(130), [0.0, 0.0, 120.0, 60.0]);
        assert_eq!(
            (outlined.tone, outlined.outline),
            (Some("dark"), Some("light"))
        );
        let plain = lettering(&page(255, false), &clean(255), [0.0, 0.0, 120.0, 60.0]);
        assert_eq!((plain.tone, plain.outline), (Some("dark"), None));

        // On a dotted screentone the cleaning redrew one dot over: the dots
        // are ground, not lettering, either way.
        let dots = |shift: u32| {
            move |x: u32, y: u32| {
                if (x + shift) % 2 == 0 && y % 2 == 0 {
                    [0, 0, 0, 255]
                } else {
                    [255, 255, 255, 255]
                }
            }
        };
        let toned = |outline: bool| {
            let ground = dots(0);
            RgbaImage::from_fn(120, 60, |x, y| {
                let stroke = (20..100).contains(&x) && (y % 20) >= 6 && (y % 20) < 14;
                let ring =
                    outline && (16..104).contains(&x) && (y % 20) >= 2 && (y % 20) < 18 && !stroke;
                image::Rgba(if stroke {
                    [10, 10, 10, 255]
                } else if ring {
                    [250, 250, 250, 255]
                } else {
                    ground(x, y)
                })
            })
        };
        let redrawn = RgbaImage::from_fn(120, 60, |x, y| image::Rgba(dots(1)(x, y)));
        let outlined = lettering(&toned(true), &redrawn, [0.0, 0.0, 120.0, 60.0]);
        assert_eq!(
            (outlined.tone, outlined.outline),
            (Some("dark"), Some("light"))
        );
        let plain = lettering(&toned(false), &redrawn, [0.0, 0.0, 120.0, 60.0]);
        assert_eq!((plain.tone, plain.outline), (Some("dark"), None));
    }

    #[test]
    fn lettering_cleaned_away_without_a_layer_is_found() {
        // Two words of dark strokes on white; the cleaning took both away.
        let source = RgbaImage::from_fn(200, 100, |x, y| {
            let word = ((20..60).contains(&x) || (120..170).contains(&x)) && (40..60).contains(&y);
            let value = if word && x % 4 < 2 { 20 } else { 250 };
            image::Rgba([value, value, value, 255])
        });
        let clean = RgbaImage::from_pixel(200, 100, image::Rgba([250, 250, 250, 255]));
        // A translated layer covers the right word only.
        let places = cleaned_without_text(&source, &clean, &[[110.0, 30.0, 70.0, 40.0]]);
        assert_eq!(places.len(), 1, "{places:?}");
        assert!(places[0][0] < 30.0 && places[0][0] + places[0][2] < 70.0);
    }

    #[test]
    fn manga_order_reads_rows_from_the_top_right_to_left() {
        // Two panels on top (right one first), one wide panel below, and a
        // small box sitting a little lower in the top row.
        let boxes = [
            [0.0, 0.0, 100.0, 100.0],
            [110.0, 0.0, 100.0, 100.0],
            [0.0, 120.0, 210.0, 80.0],
            [60.0, 20.0, 30.0, 90.0],
        ];
        assert_eq!(manga_order(&boxes), [1, 0, 3, 2]);
    }

    #[tokio::test]
    async fn an_address_names_a_page_and_its_numbered_layer() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let mut layers = Vec::new();
        let mut pages = Vec::new();
        for label in ["1", "2"] {
            let page = setup
                .add_page(
                    koharu_scene::PageDraft::new(label, 10.0, 10.0),
                    koharu_scene::At::End,
                )
                .unwrap();
            pages.push(page);
            for _ in 0..2 {
                let content = setup.add_text_content(page, koharu_scene::At::End).unwrap();
                layers.push(
                    setup
                        .add_text_layer(
                            page,
                            koharu_scene::At::End,
                            content,
                            &koharu_scene::TextLayout {
                                origin: Origin::User,
                                kind: koharu_scene::TextLayoutKind::Paragraph,
                                angle_degrees: None,
                                perspective: None,
                            },
                        )
                        .unwrap(),
                );
            }
        }
        session.commit(setup.finish().unwrap()).await.unwrap();
        let snapshot = session.snapshot();
        assert_eq!(address(&snapshot, "2").unwrap(), pages[1]);
        assert_eq!(address(&snapshot, "2#2").unwrap(), layers[3]);
        assert_eq!(
            address(&snapshot, &layers[0].to_string()).unwrap(),
            layers[0]
        );
        assert!(address(&snapshot, "3#1").is_err());
        assert!(address(&snapshot, "2#9").is_err());
    }

    #[test]
    fn text_turns_on_a_surface_to_the_printed_slant() {
        // A flat, upright surface: a turn on it is a turn on the page.
        let flat = Surface::new([
            [100.0, 100.0],
            [300.0, 100.0],
            [300.0, 200.0],
            [100.0, 200.0],
        ])
        .unwrap();
        assert!((flat.slant(0.0)).abs() < 1e-6);
        assert!((flat.turn_to(-12.0) + 12.0).abs() < 0.05);
        // A receding column: whatever the turn, the drawn lines reach the
        // slant asked for.
        let column = Surface::new([
            [410.0, 220.0],
            [612.0, 251.0],
            [598.0, 530.0],
            [380.0, 488.0],
        ])
        .unwrap();
        for slant in [-14.0, 0.0, 9.0] {
            let turn = column.turn_to(slant);
            assert!((column.slant(turn) - slant).abs() < 0.05, "{slant} {turn}");
            let turned = Surface::new(column.turned(turn)).unwrap();
            assert!((turned.slant(0.0) - slant).abs() < 0.5, "{slant}");
        }
    }

    #[test]
    fn surface_corners_come_back_from_the_frame_they_set() {
        // A newspaper column tilted and seen from its lower left.
        let corners = [
            [410.0, 220.0],
            [612.0, 251.0],
            [598.0, 530.0],
            [380.0, 488.0],
        ];
        let (frame, unit) = frame_from_corners(corners).unwrap();
        assert!(koharu_scene::valid_perspective(&unit));
        let frame_box = [
            f64::from(frame.x),
            f64::from(frame.y),
            f64::from(frame.width),
            f64::from(frame.height),
        ];
        for (back, corner) in corners_on_page(frame_box, frame.angle_degrees, &unit)
            .into_iter()
            .zip(corners)
        {
            assert!((back[0] - corner[0]).abs() < 0.2 && (back[1] - corner[1]).abs() < 0.2);
        }
        // Corners out of order cross and are refused.
        let crossed = [corners[0], corners[2], corners[1], corners[3]];
        assert!(
            frame_from_corners(crossed)
                .is_none_or(|(_, unit)| !koharu_scene::valid_perspective(&unit))
        );
    }

    #[test]
    fn balloon_issues_name_crossing_off_centre_and_undersized_text() {
        let layer = |size: f64, gap: f64, off: [f64; 2]| {
            json!({
                "visible": true,
                "translated_text": "Привет",
                "typography": { "writing_mode": null },
                "balloon": {
                    "measured": "pixels",
                    "interior": { "box": [0.0, 0.0, 200.0, 400.0] },
                    "off_centre": off,
                    "edge_gap": gap,
                },
                "rendered": { "font_size": size, "small": false },
            })
        };
        let hidden = json!({ "visible": false, "translated_text": "Скрыто" });
        let layers = [
            layer(25.0, 12.0, [3.0, 4.0]),
            layer(26.0, 10.0, [-2.0, 0.0]),
            layer(40.0, 8.0, [0.0, 0.0]),
            layer(20.0, -9.0, [30.0, 60.0]),
            hidden,
        ];
        let found = page_balloon_issues(&layers.iter().collect::<Vec<_>>());
        assert!(found[0].is_empty() && found[1].is_empty());
        // A larger shout is the letterer's choice.
        assert!(found[2].is_empty());
        assert_eq!(
            found[3],
            [
                "crosses=-9",
                "off_centre=30,60",
                "size=20 similar_median=26"
            ]
        );
        assert!(found[4].is_empty());

        // A narrow balloon is compared with narrow ones, and a measure a
        // panel line cut short reports no crossing.
        let narrow = |size: f64, measured: &str| {
            json!({
                "visible": true,
                "translated_text": "Привет",
                "typography": { "writing_mode": null },
                "balloon": {
                    "measured": measured,
                    "interior": { "box": [0.0, 0.0, 60.0, 400.0] },
                    "off_centre": [0.0, 0.0],
                    "edge_gap": -9.0,
                },
                "rendered": { "font_size": size, "small": false },
            })
        };
        let layers = [
            layer(26.0, 10.0, [0.0, 0.0]),
            layer(26.0, 10.0, [0.0, 0.0]),
            layer(26.0, 10.0, [0.0, 0.0]),
            narrow(16.0, "partial"),
            narrow(16.0, "partial"),
            narrow(15.0, "partial"),
        ];
        let found = page_balloon_issues(&layers.iter().collect::<Vec<_>>());
        assert!(found.iter().all(Vec::is_empty), "{found:?}");
    }

    #[tokio::test]
    async fn restyling_detected_typography_keeps_the_translation_horizontal() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("1", 100.0, 100.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let content = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                koharu_scene::At::End,
                content,
                &koharu_scene::TextLayout {
                    origin: Origin::User,
                    kind: koharu_scene::TextLayoutKind::Paragraph,
                    angle_degrees: None,
                    perspective: None,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        // Detection records the Japanese source's vertical direction.
        let detection = koharu_scene::Generation::new(
            koharu_scene::ProducerId::new("dev.koharu.pipeline.detection").unwrap(),
        );
        let mut detected = session.snapshot().edit_as(detection.clone());
        detected
            .set(
                layer,
                &koharu_scene::Typography {
                    origin: Origin::Generated(detection),
                    preferred_font: None,
                    font_weight: None,
                    font_style: None,
                    size: None,
                    auto_fit: true,
                    color: Some([0, 0, 0, 255]),
                    stroke_color: None,
                    stroke_width: None,
                    alignment: None,
                    writing_mode: Some(koharu_scene::WritingMode::Vertical),
                    extensions: Default::default(),
                    line_breaking: None,
                },
            )
            .unwrap();
        session.commit(detected.finish().unwrap()).await.unwrap();

        let change: TrialChange = serde_json::from_value(json!({
            "element": "1#1",
            "typography": { "preferred_font": "Arial", "line_breaking": "whole_words" },
        }))
        .unwrap();
        let restyled = steps(&session.snapshot(), vec![change], &HashMap::new(), |_| None).unwrap();
        let mut edit = session.snapshot().edit();
        for step in restyled {
            step(&mut edit).unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        let restyled = session
            .snapshot()
            .component::<koharu_scene::Typography>(layer)
            .unwrap()
            .unwrap();
        assert_eq!(restyled.writing_mode, None);
        assert_eq!(restyled.preferred_font.as_deref(), Some("Arial"));

        // Automatic line breaking can be asked for again.
        let change: TrialChange = serde_json::from_value(json!({
            "element": "1#1",
            "typography": { "line_breaking": "automatic", "size": 20.0 },
        }))
        .unwrap();
        let reset = steps(&session.snapshot(), vec![change], &HashMap::new(), |_| None).unwrap();
        let mut edit = session.snapshot().edit();
        for step in reset {
            step(&mut edit).unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        let reset = session
            .snapshot()
            .component::<koharu_scene::Typography>(layer)
            .unwrap()
            .unwrap();
        assert_eq!(reset.line_breaking, None);
        // A size given alone is drawn at that size.
        assert!(!reset.auto_fit);
        assert_eq!(reset.size, Some(20.0));
    }

    #[tokio::test]
    async fn a_text_change_keeps_the_layers_typography() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("1", 100.0, 100.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let content = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                koharu_scene::At::End,
                content,
                &koharu_scene::TextLayout {
                    origin: Origin::User,
                    kind: koharu_scene::TextLayoutKind::Paragraph,
                    angle_degrees: None,
                    perspective: None,
                },
            )
            .unwrap();
        let typography = koharu_scene::Typography {
            origin: Origin::User,
            preferred_font: Some("NM Shonen 2".to_owned()),
            font_weight: None,
            font_style: None,
            size: Some(30.0),
            auto_fit: false,
            color: Some([0, 0, 0, 255]),
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
            extensions: Default::default(),
            line_breaking: None,
        };
        setup.set(layer, &typography).unwrap();
        setup
            .set(
                content,
                &koharu_scene::SourceText {
                    text: koharu_scene::Authored::user("こんにちは".to_owned()),
                    language: None,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();

        let change: TrialChange =
            serde_json::from_value(json!({ "element": "1#1", "text": "Привет" })).unwrap();
        let retexted = steps(&session.snapshot(), vec![change], &HashMap::new(), |_| None).unwrap();
        let mut edit = session.snapshot().edit();
        for step in retexted {
            step(&mut edit).unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();

        let snapshot = session.snapshot();
        assert_eq!(
            snapshot
                .component::<koharu_scene::Typography>(layer)
                .unwrap()
                .unwrap(),
            typography
        );
        assert_eq!(
            snapshot
                .text_layer(layer)
                .unwrap()
                .content()
                .unwrap()
                .translation()
                .unwrap()
                .unwrap()
                .text
                .value,
            "Привет"
        );

        // A new size alone keeps the font, colour and the rest.
        let change: TrialChange =
            serde_json::from_value(json!({ "element": "1#1", "typography": { "size": 22.0 } }))
                .unwrap();
        let resized = steps(&session.snapshot(), vec![change], &HashMap::new(), |_| None).unwrap();
        let mut edit = session.snapshot().edit();
        for step in resized {
            step(&mut edit).unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        assert_eq!(
            session
                .snapshot()
                .component::<koharu_scene::Typography>(layer)
                .unwrap()
                .unwrap(),
            koharu_scene::Typography {
                size: Some(22.0),
                ..typography
            }
        );
    }

    #[tokio::test]
    async fn shown_translated_text_with_no_chosen_font_or_colour_is_listed() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("1", 100.0, 100.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let typography = |font: Option<&str>, color: Option<[u8; 4]>| koharu_scene::Typography {
            origin: Origin::User,
            preferred_font: font.map(str::to_owned),
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: None,
            extensions: Default::default(),
            line_breaking: None,
        };
        // No typography; both unset; a chosen font; untranslated; hidden.
        let cases = [
            (None, true, true),
            (Some(typography(None, None)), true, true),
            (Some(typography(Some("NM Shonen 2"), None)), true, true),
            (None, false, true),
            (None, true, false),
        ];
        let mut layers = Vec::new();
        for (style, translated, visible) in cases {
            let content = setup.add_text_content(page, koharu_scene::At::End).unwrap();
            let layer = setup
                .add_text_layer(
                    page,
                    koharu_scene::At::End,
                    content,
                    &koharu_scene::TextLayout {
                        origin: Origin::User,
                        kind: koharu_scene::TextLayoutKind::Paragraph,
                        angle_degrees: None,
                        perspective: None,
                    },
                )
                .unwrap();
            setup
                .set(
                    content,
                    &koharu_scene::SourceText {
                        text: koharu_scene::Authored::user("あ".to_owned()),
                        language: None,
                    },
                )
                .unwrap();
            if translated {
                setup
                    .set(
                        content,
                        &koharu_scene::Translation {
                            text: koharu_scene::Authored::user("А".to_owned()),
                            language: None,
                        },
                    )
                    .unwrap();
            }
            if let Some(style) = style {
                setup.set(layer, &style).unwrap();
            }
            if !visible {
                setup
                    .set(
                        layer,
                        &Visibility {
                            origin: Origin::User,
                            visible: false,
                            opacity: 1.0,
                        },
                    )
                    .unwrap();
            }
            layers.push(layer);
        }
        session.commit(setup.finish().unwrap()).await.unwrap();

        assert_eq!(
            automatic_style_layers(&session.snapshot(), page).unwrap(),
            vec![layers[0], layers[1]]
        );
    }

    #[tokio::test]
    async fn the_owner_cleaning_leaves_the_text_side_unchanged() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(
                koharu_scene::PageDraft::new("1", 10.0, 10.0),
                koharu_scene::At::End,
            )
            .unwrap();
        let content = setup.add_text_content(page, koharu_scene::At::End).unwrap();
        setup
            .add_text_layer(
                page,
                koharu_scene::At::End,
                content,
                &koharu_scene::TextLayout {
                    origin: Origin::User,
                    kind: koharu_scene::TextLayoutKind::Paragraph,
                    angle_degrees: None,
                    perspective: None,
                },
            )
            .unwrap();
        let cleanup = setup.add_entity(page, koharu_scene::At::End).unwrap();
        setup
            .set(
                cleanup,
                &koharu_scene::RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: koharu_scene::RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let before = text_signatures(&session.snapshot()).unwrap();

        let mut cleaning = session.snapshot().edit();
        cleaning
            .set(
                cleanup,
                &Visibility {
                    origin: Origin::User,
                    visible: false,
                    opacity: 1.0,
                },
            )
            .unwrap();
        session.commit(cleaning.finish().unwrap()).await.unwrap();
        assert_eq!(text_signatures(&session.snapshot()).unwrap(), before);

        let mut text = session.snapshot().edit();
        text.set(
            content,
            &koharu_scene::SourceText {
                text: koharu_scene::Authored::user("テスト".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(text.finish().unwrap()).await.unwrap();
        assert_ne!(text_signatures(&session.snapshot()).unwrap(), before);
    }

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
