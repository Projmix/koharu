use std::{
    cmp::Ordering,
    collections::{BTreeMap, VecDeque},
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{
    DynamicImage, ExtendedColorType, GrayImage, ImageEncoder as _, Luma, RgbImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use imageproc::{
    contours::{BorderType, find_contours_with_threshold},
    distance_transform::{Norm, distance_transform},
    geometry::{approximate_polygon_dp, arc_length, contour_area},
    morphology::{close, dilate},
};
use koharu_ml::koharu_layout_rfdetr_seg_2xl::{
    KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask, KoharuLayoutRFDetrSeg2XL,
    KoharuLayoutThresholds,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, DetectionAnalysis, DetectionLabel,
    EntityId, EntityOrigin, FitsTo, FlowsIn, Generation, Geometry, Inside, Origin, PanelRegion,
    Point, Presents, RecognizedFrom, Region, RegionKind, RegionSpec, RemovePolicy, TextContent,
    TextLayout, TextLayoutKind, TextRegion, TextRole, Typography, WritingMode,
};
use koharu_translator::Language;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{StageInput, StageProcessor, finish, generation};
use crate::{DetectionModel, ModelCell};

const OUTPUT_MODEL_ID: &str = "mayocream/koharu-layout-rfdetr-seg-2xl-1152#postprocess-6";
#[cfg(test)]
const LEGACY_OUTPUT_MODEL_ID: &str = "mayocream/koharu-layout-rfdetr-seg-2xl-1152";
const MODEL_NAME: &str = "koharu-layout-rfdetr-seg-2xl";
const PRODUCER: &str = "dev.koharu.pipeline.detection";
const ANGLE_SNAP_DEGREES: f32 = 3.0;
const ANGLE_SEARCH_HALF_STEPS: i32 = 90;
const ANGLE_SEARCH_STEP_DEGREES: f64 = 0.5;
// The background ring must clear outlines drawn past the glyph box (up to
// about 13 px on the owner's chapter); otherwise the outline itself is taken
// as the background and the stroke is lost.
const TYPOGRAPHY_SAMPLE_MARGIN: f32 = 14.0;
const COLOR_SNAP_DARK_LUMINANCE: u32 = 64 * 256;
const COLOR_SNAP_LIGHT_LUMINANCE: u32 = 191 * 256;
const COLOR_CLUSTER_MIN_DISTANCE_SQUARED: u32 = 32 * 32;
const COLOR_CLUSTER_COUNT: usize = 4;
const MIN_EXTREME_COLOR_PIXELS: u32 = 4;
const MIN_MEASURED_STROKE_WIDTH: u8 = 2;
const DIALOGUE_MASK_CONTAINMENT_THRESHOLD: f32 = 0.9;
const TEXT_DUPLICATE_IOU_THRESHOLD: f32 = 0.95;
const TEXT_DUPLICATE_AREA_RATIO_THRESHOLD: f32 = 0.9;
const NESTED_TEXT_CONTAINMENT_THRESHOLD: f32 = 0.8;
const SPURIOUS_ONOMATOPOEIA_MAX_SCORE: f32 = 0.5;
const SPURIOUS_ONOMATOPOEIA_CONTAINMENT: f32 = 0.5;
const SPURIOUS_ONOMATOPOEIA_PAGE_FRACTION: f32 = 0.2;
const PAGE_NUMBER_EDGE_FRACTION: f32 = 0.012;
const PAGE_NUMBER_MAX_WIDTH_FRACTION: f32 = 0.016;
const PAGE_NUMBER_MAX_HEIGHT_FRACTION: f32 = 0.035;
const PAGE_NUMBER_MARGIN_FRACTION: f32 = 0.15;
const DARK_BALLOON_LUMINANCE: u8 = 96;
const DARK_BALLOON_RING_FRACTION: f32 = 0.9;
const DARK_BALLOON_MIN_GROWTH: f32 = 1.15;
const FREE_TEXT_ROLE: &str = "dev.koharu.text.free-text";
const DIALOGUE_ROLE: &str = "dev.koharu.text.dialogue";
const ONOMATOPOEIA_ROLE: &str = "dev.koharu.text.onomatopoeia";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct KoharuLayoutRFDetrSeg2XLConfig {
    pub text_threshold: Option<f32>,
    pub bubble_threshold: Option<f32>,
    pub panel_threshold: Option<f32>,
}

pub(super) struct Processor {
    config: DetectionModel,
    source_language: Language,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(
        mut config: DetectionModel,
        source_language: Language,
        device: koharu_ml::Device,
    ) -> Self {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &mut config;
        for (name, value) in [
            ("text", &mut settings.text_threshold),
            ("bubble", &mut settings.bubble_threshold),
            ("panel", &mut settings.panel_threshold),
        ] {
            // A stored threshold is only a preference; refusing to start over one
            // leaves the application unusable until the file is edited by hand.
            if let Some(threshold) = *value
                && !(threshold.is_finite() && (0.0..=1.0).contains(&threshold))
            {
                tracing::warn!(
                    label = name,
                    threshold = %threshold,
                    "confidence threshold is not between zero and one; using the model default"
                );
                *value = None;
            }
        }

        Self {
            config,
            source_language,
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, _input: &StageInput) -> String {
        MODEL_NAME.to_owned()
    }

    fn recovers_from_memory_pressure(&self, _input: &StageInput) -> bool {
        true
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        let current_generation = generation(PRODUCER, OUTPUT_MODEL_ID)?;
        let mut cached_text_bounds = Vec::new();
        for entity in input.scene.descendants(input.page()?)? {
            let id = entity.id();
            if !input.contains_entity(id)?
                || !entity
                    .component::<Region>()?
                    .is_some_and(|region| region.kind == TextRegion::kind())
            {
                continue;
            }
            let Some(EntityOrigin {
                origin: Origin::Generated(owner),
            }) = entity.component::<EntityOrigin>()?
            else {
                return Ok(true);
            };
            if owner.producer == current_generation.producer
                && owner.model != current_generation.model
            {
                return Ok(false);
            }
            if input
                .scene
                .relations_to_as::<RecognizedFrom>(id)
                .next()
                .is_none()
            {
                return Ok(false);
            }
            let content = input
                .scene
                .relations_to_as::<RecognizedFrom>(id)
                .next()
                .expect("recognized text relation was checked above")
                .value()
                .source;
            if input.scene.component::<TextContent>(content)?.is_none() {
                return Ok(false);
            }
            let mut has_presentation = false;
            for relation in input.scene.relations_to_as::<Presents>(content) {
                let layer = relation.value().source;
                if input.contains_entity(layer)?
                    && input.scene.component::<TextLayout>(layer)?.is_some()
                {
                    has_presentation = true;
                    break;
                }
            }
            if !has_presentation {
                return Ok(false);
            }
            let Some(geometry) = entity.component::<Geometry>()? else {
                return Ok(false);
            };
            let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&geometry) else {
                return Ok(false);
            };
            let bounds = [left as f32, top as f32, right as f32, bottom as f32];
            if cached_text_bounds.iter().any(|existing| {
                intersection_over_union(*existing, bounds) >= TEXT_DUPLICATE_IOU_THRESHOLD
            }) {
                return Ok(false);
            }
            cached_text_bounds.push(bounds);
        }
        Ok(!cached_text_bounds.is_empty())
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self, _input: &StageInput) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("detection model is not loaded"))?
            .run(input, self.source_language)
            .await
    }
}

struct Model {
    network: Arc<Mutex<KoharuLayoutRFDetrSeg2XL>>,
    thresholds: KoharuLayoutThresholds,
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &DetectionModel) -> Result<Self> {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = config;
        let network = KoharuLayoutRFDetrSeg2XL::load(device).await?;
        let mut thresholds = network.recommended_thresholds();
        thresholds.text = config.text_threshold.unwrap_or(thresholds.text);
        thresholds.bubble = config.bubble_threshold.unwrap_or(thresholds.bubble);
        thresholds.panel = config.panel_threshold.unwrap_or(thresholds.panel);
        Ok(Self {
            network: Arc::new(Mutex::new(network)),
            thresholds,
        })
    }

    async fn run(
        &self,
        input: StageInput,
        source_language: Language,
    ) -> Result<koharu_scene::Patch> {
        let page = input.page()?;
        let image = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let output = self.detect(image.clone()).await?;
        build_patch(
            &input,
            &image,
            output,
            &generation(PRODUCER, OUTPUT_MODEL_ID)?,
            source_language,
        )
        .await
    }

    async fn detect(&self, image: Arc<DynamicImage>) -> Result<KoharuLayoutDetections> {
        let network = self.network.clone();
        let thresholds = self.thresholds;
        tokio_rayon::spawn(move || {
            let network = network
                .lock()
                .map_err(|_| anyhow!("layout model lock is poisoned"))?;
            network.inference_with_thresholds(&image, thresholds)
        })
        .await
    }
}

struct DetectedRegion<'a> {
    entity: EntityId,
    mask: &'a KoharuLayoutMask,
    area: u32,
}

struct DetectedText<'a> {
    entity: EntityId,
    mask: &'a KoharuLayoutMask,
    bounds: [f32; 4],
    content: EntityId,
    layer: EntityId,
    role: &'static str,
}

enum RegionOutput<'a> {
    Bubble(DetectedRegion<'a>),
    Text(DetectedText<'a>),
    Other,
}

#[derive(Default)]
struct PageRegions<'a> {
    bubbles: Vec<DetectedRegion<'a>>,
    texts: Vec<DetectedText<'a>>,
}

#[derive(Clone, Copy)]
struct ImageSize {
    width: u32,
    height: u32,
}

async fn build_patch(
    input: &StageInput,
    image: &DynamicImage,
    output: KoharuLayoutDetections,
    generation: &Generation,
    source_language: Language,
) -> Result<koharu_scene::Patch> {
    let page = input.page()?;
    let mut edit = input.scene.edit_as(generation.clone());
    edit.observe_subtree(page)?;
    remove_previous_regions(input, &mut edit, generation)
        .context("failed to replace the previous detection regions")?;
    write_page(
        input,
        &mut edit,
        page,
        image,
        output,
        generation,
        source_language,
    )
    .await
    .context("failed to write detection output")?;
    finish(edit)
}

fn remove_previous_regions(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    generation: &Generation,
) -> Result<()> {
    let mut remove = Vec::new();
    for entity in input.scene.descendants(input.page()?)? {
        let id = entity.id();
        if !input.contains_entity(id)? {
            continue;
        }
        let owned_region = entity
            .component::<EntityOrigin>()?
            .is_some_and(|origin| {
                matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
            })
            && entity.component::<Region>()?.is_some();
        if owned_region {
            remove.push(id);
        }
    }
    for entity in remove {
        if input.scene.entity(entity).is_ok() {
            edit.remove_entity(entity, RemovePolicy::Cascade)?;
        }
    }
    Ok(())
}

async fn write_page(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &DynamicImage,
    output: KoharuLayoutDetections,
    generation: &Generation,
    source_language: Language,
) -> Result<()> {
    let KoharuLayoutDetections {
        mut detections,
        image_width,
        image_height,
    } = output;
    let size = ImageSize {
        width: image_width,
        height: image_height,
    };
    if let Some(region) = input.region {
        detections.retain(|detection| intersects(detection.bbox, region));
    }
    clamp_to_image(&mut detections, size);
    non_maximum_suppression(&mut detections, 0.5);
    remove_spurious_onomatopoeia(&mut detections, size);
    remove_page_numbers(&mut detections, size);
    remove_nested_text_detections(&mut detections);
    let image = image.to_rgb8();
    let dark = dark_balloons(&image, &detections);
    detections.extend(dark);
    sort_by_layout(&mut detections, source_language);

    let regions = write_regions(&input.scene, edit, page, &image, &detections, generation)
        .context("failed to write detected regions")?;
    link_dialogue_regions(edit, &regions, generation)
        .context("failed to associate detected text with dialogue regions")?;
    write_masks(input, edit, page, &detections, size)
        .await
        .context("failed to write detection masks")
}

fn write_regions<'a>(
    snapshot: &koharu_scene::Snapshot,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &RgbImage,
    detections: &'a [KoharuLayoutDetection],
    generation: &Generation,
) -> Result<PageRegions<'a>> {
    let mut regions = PageRegions::default();
    let inferred = detections
        .par_iter()
        .map(|detection| {
            matches!(detection.label.as_str(), "text" | "onomatopoeia")
                .then(|| infer_typography(image, detection))
                .flatten()
        })
        .collect::<Vec<_>>();
    let text_group = snapshot.page(page)?.text_group()?;
    let managed_text_group = if let Some(group) = text_group {
        snapshot
            .component::<EntityOrigin>(group.id())?
            .is_some_and(|origin| origin.origin != Origin::User)
            .then_some(group.id())
    } else {
        None
    };
    for (index, (detection, inferred)) in detections.iter().zip(inferred).enumerate() {
        match write_region(edit, page, detection, inferred, generation)
            .with_context(|| format!("failed to write {} detection {index}", detection.label))?
        {
            RegionOutput::Bubble(bubble) => regions.bubbles.push(bubble),
            RegionOutput::Text(text) => {
                if let Some(group) = managed_text_group {
                    edit.move_entity(text.layer, Some(group), At::End)
                        .context("failed to place the detected text layer in its group")?;
                }
                regions.texts.push(text);
            }
            RegionOutput::Other => {}
        }
    }
    Ok(regions)
}

fn write_region<'a>(
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detection: &'a KoharuLayoutDetection,
    inferred: Option<InferredTypography>,
    generation: &Generation,
) -> Result<RegionOutput<'a>> {
    let text_role = match detection.label.as_str() {
        "text" => Some(FREE_TEXT_ROLE),
        "onomatopoeia" => Some(ONOMATOPOEIA_ROLE),
        _ => None,
    };
    let entity = edit
        .add_entity(page, At::End)
        .context("failed to create a detected region")?;
    let kind = region_kind(&detection.label)?;
    let geometry = if detection.label == "bubble" {
        mask_geometry(&detection.mask).unwrap_or_else(|| rectangle_geometry(detection.bbox))
    } else if text_role.is_some() {
        inferred.map_or_else(
            || rectangle_geometry(detection.bbox),
            |typography| {
                if typography.angle_degrees == 0.0 {
                    rectangle_geometry(detection.bbox)
                } else {
                    rotated_rectangle_geometry(typography.oriented_extent, typography.angle_degrees)
                }
            },
        )
    } else {
        rectangle_geometry(detection.bbox)
    };
    edit.set(entity, &geometry)
        .context("failed to set detected region geometry")?;
    edit.set(
        entity,
        &Region {
            origin: Origin::Generated(generation.clone()),
            kind: kind.clone(),
            label: Some(detection.label.clone()),
        },
    )
    .context("failed to set detected region metadata")?;
    edit.set(
        entity,
        &DetectionAnalysis {
            origin: Origin::Generated(generation.clone()),
            labels: vec![DetectionLabel {
                kind,
                confidence: detection.score,
            }],
        },
    )
    .context("failed to set detection analysis")?;
    let Some(text_role) = text_role else {
        return Ok(if detection.label == "bubble" {
            RegionOutput::Bubble(DetectedRegion {
                entity,
                mask: &detection.mask,
                area: detection.area,
            })
        } else {
            RegionOutput::Other
        });
    };

    let content = edit.add_text_content(page, At::End)?;
    let layer = edit.add_text_layer(
        page,
        At::End,
        content,
        &TextLayout {
            origin: Origin::Generated(generation.clone()),
            kind: TextLayoutKind::Paragraph,
            angle_degrees: None,
        },
    )?;
    write_text_role(edit, content, text_role, generation)
        .context("failed to set the detected text role")?;
    edit.set(
        layer,
        &Typography {
            origin: Origin::Generated(generation.clone()),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: inferred.map(|value| [value.color[0], value.color[1], value.color[2], u8::MAX]),
            stroke_color: inferred
                .and_then(|value| value.stroke_color)
                .map(|color| [color[0], color[1], color[2], u8::MAX]),
            stroke_width: inferred.and_then(|value| value.stroke_width),
            alignment: None,
            writing_mode: inferred.map(|value| value.writing_mode),
            extensions: Default::default(),
        },
    )
    .context("failed to set detected typography")?;
    edit.relate::<RecognizedFrom>(content, entity)
        .context("failed to associate detected text content with its region")?;

    Ok(RegionOutput::Text(DetectedText {
        entity,
        mask: &detection.mask,
        bounds: detection.bbox,
        content,
        layer,
        role: text_role,
    }))
}

fn link_dialogue_regions(
    edit: &mut koharu_scene::Edit,
    regions: &PageRegions<'_>,
    generation: &Generation,
) -> Result<()> {
    let containing = regions
        .texts
        .iter()
        .map(|text| containing_bubble(&regions.bubbles, text).map(|bubble| bubble.entity))
        .collect::<Vec<_>>();
    for (text, bubble_entity) in regions.texts.iter().zip(&containing) {
        if text.role == ONOMATOPOEIA_ROLE {
            // Fork divergence: a sound effect that is the only text in a balloon
            // (often a dark balloon with light lettering) lays out in that balloon,
            // keeping its role. One sharing a balloon with dialogue stays fitted to
            // its own region so it does not take a share of the speech.
            let alone = bubble_entity.is_some_and(|bubble| {
                containing
                    .iter()
                    .filter(|other| **other == Some(bubble))
                    .count()
                    == 1
            });
            match bubble_entity {
                Some(bubble) if alone => {
                    edit.relate::<Inside>(text.entity, *bubble)?;
                    edit.relate::<FlowsIn>(text.layer, *bubble)?;
                }
                _ => {
                    edit.relate::<FitsTo>(text.layer, text.entity)?;
                }
            }
            continue;
        }
        let bubble = containing_bubble(&regions.bubbles, text);
        if let Some(bubble) = bubble {
            edit.relate::<Inside>(text.entity, bubble.entity)?;
            edit.relate::<FlowsIn>(text.layer, bubble.entity)?;
            write_text_role(edit, text.content, DIALOGUE_ROLE, generation)?;
        } else {
            edit.relate::<FitsTo>(text.layer, text.entity)?;
        }
    }
    Ok(())
}

fn containing_bubble<'regions, 'detections>(
    bubbles: &'regions [DetectedRegion<'detections>],
    text: &DetectedText<'detections>,
) -> Option<&'regions DetectedRegion<'detections>> {
    bubbles
        .iter()
        .filter(|bubble| {
            mask_containment(bubble.mask, text.mask, text.bounds)
                >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
        })
        .min_by_key(|bubble| bubble.area)
}

fn write_text_role(
    edit: &mut koharu_scene::Edit,
    entity: EntityId,
    role: &str,
    generation: &Generation,
) -> Result<()> {
    edit.set(
        entity,
        &TextRole {
            origin: Origin::Generated(generation.clone()),
            role: role.to_owned(),
        },
    )?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InferredTypography {
    color: [u8; 3],
    stroke_color: Option<[u8; 3]>,
    stroke_width: Option<f32>,
    angle_degrees: f32,
    /// `[min_u, min_v, max_u, max_v]` of the mask in the frame rotated by
    /// `angle_degrees`, where u = x cos + y sin and v = y cos - x sin.
    oriented_extent: [f64; 4],
    writing_mode: WritingMode,
}

#[derive(Clone, Copy)]
struct MaskPoint {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy)]
struct MaskPixel {
    x: u32,
    y: u32,
    color: [u8; 3],
    inside_mask: bool,
}

// BallonsTranslator normalizes vertical-line angles relative to upright text:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/utils/textblock.py#L576-L608
// RF-DETR provides foreground pixels rather than line quadrilaterals, so
// projection-profile sharpness supplies the line axis. Whole-block PCA is
// deliberately avoided because a tall multiline horizontal block otherwise
// looks vertical.
fn infer_typography(
    image: &RgbImage,
    detection: &KoharuLayoutDetection,
) -> Option<InferredTypography> {
    let mask = &detection.mask;
    let width = image.width();
    let height = image.height();
    let [bbox_left, bbox_top, bbox_right, bbox_bottom] = detection.bbox;
    let [left, top, right, bottom] = mask_window(
        [
            bbox_left - TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_top - TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_right + TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_bottom + TYPOGRAPHY_SAMPLE_MARGIN,
        ],
        width,
        height,
    )?;
    let local_width = right - left + 2;
    let local_height = bottom - top + 2;
    let background_margin = TYPOGRAPHY_SAMPLE_MARGIN.ceil() as u32;
    let mut points = Vec::new();
    let mut pixels = Vec::new();
    let mut background = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let local_x = x - left + 1;
            let local_y = y - top + 1;
            let color = image.get_pixel(x, y).0;
            let inside_mask = mask.contains(x, y);
            pixels.push(MaskPixel {
                x: local_x,
                y: local_y,
                color,
                inside_mask,
            });
            if inside_mask {
                points.push(MaskPoint {
                    x: f64::from(x) + 0.5,
                    y: f64::from(y) + 0.5,
                });
            }
            if x < left + background_margin
                || x + background_margin >= right
                || y < top + background_margin
                || y + background_margin >= bottom
            {
                background.push(color);
            }
        }
    }
    if points.is_empty() {
        return None;
    }

    let background = if background.is_empty() {
        median_pixel_color(&pixels)
    } else {
        median_color(&background)
    };
    let (angle_degrees, vertical) = mask_angle(&points, detection.bbox);
    let oriented_extent = oriented_extent(&points, angle_degrees);
    let (color, stroke_color, stroke_width) =
        infer_text_paint(&pixels, background, local_width, local_height);
    Some(InferredTypography {
        color,
        stroke_color,
        stroke_width,
        angle_degrees,
        oriented_extent,
        writing_mode: if vertical {
            WritingMode::Vertical
        } else {
            WritingMode::Horizontal
        },
    })
}

fn mask_window([left, top, right, bottom]: [f32; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    if width == 0 || height == 0 {
        return None;
    }
    let left = left.floor().clamp(0.0, width as f32) as u32;
    let top = top.floor().clamp(0.0, height as f32) as u32;
    let right = right.ceil().clamp(0.0, width as f32) as u32;
    let bottom = bottom.ceil().clamp(0.0, height as f32) as u32;
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn mask_angle(points: &[MaskPoint], [left, top, right, bottom]: [f32; 4]) -> (f32, bool) {
    let first = points[0];
    let bounds = points[1..].iter().fold(
        [first.x, first.y, first.x, first.y],
        |[left, top, right, bottom], point| {
            [
                left.min(point.x),
                top.min(point.y),
                right.max(point.x),
                bottom.max(point.y),
            ]
        },
    );
    let mut horizontal = (f64::NEG_INFINITY, 0.0);
    let mut vertical = (f64::NEG_INFINITY, 0.0);
    for step in -ANGLE_SEARCH_HALF_STEPS..=ANGLE_SEARCH_HALF_STEPS {
        let angle_degrees = f64::from(step) * ANGLE_SEARCH_STEP_DEGREES;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let (horizontal_score, vertical_score) = projection_scores(points, bounds, sin, cos);
        if horizontal_score > horizontal.0 {
            horizontal = (horizontal_score, angle_degrees);
        }
        if vertical_score > vertical.0 {
            vertical = (vertical_score, angle_degrees);
        }
    }
    let maximum_score = horizontal.0.max(vertical.0);
    let scores_are_close = (horizontal.0 - vertical.0).abs() <= maximum_score * 0.02;
    let is_vertical = if scores_are_close {
        bottom - top > right - left
    } else {
        vertical.0 > horizontal.0
    };
    let mut angle = if is_vertical {
        vertical.1
    } else {
        horizontal.1
    } as f32;
    if angle.abs() < ANGLE_SNAP_DEGREES {
        angle = 0.0;
    }
    (angle, is_vertical)
}

fn projection_scores(points: &[MaskPoint], bounds: [f64; 4], sin: f64, cos: f64) -> (f64, f64) {
    let (horizontal_origin, horizontal_length) = projection_extent(bounds, -sin, cos);
    let (vertical_origin, vertical_length) = projection_extent(bounds, cos, sin);
    let mut horizontal = vec![0.0; horizontal_length];
    let mut vertical = vec![0.0; vertical_length];
    for point in points {
        let horizontal_projection = point.y * cos - point.x * sin - horizontal_origin;
        let horizontal_index = horizontal_projection.floor() as usize;
        let horizontal_fraction = horizontal_projection - horizontal_index as f64;
        horizontal[horizontal_index] += 1.0 - horizontal_fraction;
        horizontal[horizontal_index + 1] += horizontal_fraction;

        let vertical_projection = point.x * cos + point.y * sin - vertical_origin;
        let vertical_index = vertical_projection.floor() as usize;
        let vertical_fraction = vertical_projection - vertical_index as f64;
        vertical[vertical_index] += 1.0 - vertical_fraction;
        vertical[vertical_index + 1] += vertical_fraction;
    }
    let count = points.len() as f64;
    (
        horizontal.iter().map(|value| value * value).sum::<f64>() / count,
        vertical.iter().map(|value| value * value).sum::<f64>() / count,
    )
}

fn projection_extent(
    [left, top, right, bottom]: [f64; 4],
    axis_x: f64,
    axis_y: f64,
) -> (f64, usize) {
    let minimum_x = if axis_x >= 0.0 { left } else { right };
    let minimum_y = if axis_y >= 0.0 { top } else { bottom };
    let maximum_x = if axis_x >= 0.0 { right } else { left };
    let maximum_y = if axis_y >= 0.0 { bottom } else { top };
    let minimum = minimum_x * axis_x + minimum_y * axis_y;
    let maximum = maximum_x * axis_x + maximum_y * axis_y;
    let origin = minimum.floor();
    let length = (maximum.ceil() - origin).max(0.0) as usize + 2;
    (origin, length)
}

fn histogram_value_at(histogram: &[u32; 256], mut rank: usize) -> u8 {
    for (value, count) in histogram.iter().enumerate() {
        if rank < *count as usize {
            return value as u8;
        }
        rank -= *count as usize;
    }
    u8::MAX
}

fn histogram_median(histogram: &[u32; 256], count: usize) -> u8 {
    if count == 0 {
        return 0;
    }
    let lower = histogram_value_at(histogram, (count - 1) / 2);
    let upper = histogram_value_at(histogram, count / 2);
    ((u16::from(lower) + u16::from(upper)) / 2) as u8
}

// The detector mask may trace glyphs or cover a whole text region. Color alone
// is also insufficient because a black glyph core can be identical to artwork
// outside its white outline. A real outline is therefore identified by the
// topology of a color band enclosing another color, then measured across that
// exact band. Clustering remains the fallback for unoutlined text.
fn infer_text_paint(
    pixels: &[MaskPixel],
    background_seed: [u8; 3],
    width: u32,
    height: u32,
) -> ([u8; 3], Option<[u8; 3]>, Option<f32>) {
    let clusters = color_clusters(pixels, background_seed);
    if let Some(outline) = infer_outline(&clusters, background_seed, width, height) {
        return (
            outline.fill_color,
            Some(outline.stroke_color),
            Some(outline.stroke_width),
        );
    }

    let nonempty = clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| !cluster.pixels.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if nonempty.len() == 1 {
        return (
            normalize_text_color(clusters[nonempty[0]].color),
            None,
            None,
        );
    }

    let background_index = nonempty
        .iter()
        .copied()
        .min_by_key(|index| color_distance_squared(clusters[*index].color, background_seed))
        .unwrap();
    let fill = nonempty
        .iter()
        .copied()
        .filter(|index| *index != background_index)
        .max_by_key(|index| cluster_ink_score(&clusters[*index], background_seed));
    let color = fill
        .map(|index| representative_ink_color(&clusters[index]))
        .unwrap_or(clusters[background_index].color);
    (normalize_text_color(color), None, None)
}

#[derive(Clone, Copy)]
struct OutlinePaint {
    fill_color: [u8; 3],
    stroke_color: [u8; 3],
    stroke_width: f32,
    score: u64,
}

fn infer_outline(
    clusters: &[ColorCluster; COLOR_CLUSTER_COUNT],
    background: [u8; 3],
    width: u32,
    height: u32,
) -> Option<OutlinePaint> {
    let mut assignments = vec![usize::MAX; width as usize * height as usize];
    for (index, cluster) in clusters.iter().enumerate() {
        for pixel in &cluster.pixels {
            assignments[pixel.y as usize * width as usize + pixel.x as usize] = index;
        }
    }

    let background_index = clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| !cluster.pixels.is_empty())
        .min_by_key(|(_, cluster)| color_distance_squared(cluster.color, background))
        .map(|(index, _)| index)?;
    let mut best = None;
    for (stroke_index, stroke_cluster) in clusters.iter().enumerate() {
        if stroke_cluster.pixels.len() < 8 || stroke_index == background_index {
            continue;
        }
        let outside = reachable_without_cluster(&assignments, stroke_index, width, height);
        for (fill_index, fill_cluster) in clusters.iter().enumerate() {
            if fill_index == stroke_index || fill_cluster.pixels.is_empty() {
                continue;
            }
            let enclosed_fill = fill_cluster
                .pixels
                .iter()
                .copied()
                .filter(|pixel| !outside[pixel.y as usize * width as usize + pixel.x as usize])
                .collect::<Vec<_>>();
            if enclosed_fill.len() < 8 {
                continue;
            }
            let stroke_pixels = enclosing_cluster_pixels(
                &assignments,
                stroke_index,
                &stroke_cluster.pixels,
                &enclosed_fill,
                width,
                height,
            );
            if stroke_pixels.len() < 8 {
                continue;
            }

            let fill_color = median_pixel_color(&enclosed_fill);
            let stroke_color = median_pixel_color(&stroke_pixels);
            let contrast = color_distance_squared(fill_color, stroke_color);
            if contrast < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_distance_squared(fill_color, background)
                    < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_distance_squared(stroke_color, background)
                    < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_lies_between(stroke_color, fill_color, background)
                || color_lies_between(fill_color, stroke_color, background)
            {
                continue;
            }
            // A colour between the fill and another ink is anti-aliasing at
            // their edge, not a drawn outline (white glyphs on a dark balloon).
            if clusters.iter().enumerate().any(|(index, cluster)| {
                index != fill_index
                    && index != stroke_index
                    && !cluster.pixels.is_empty()
                    && (color_lies_between(stroke_color, fill_color, cluster.color)
                        || color_lies_between(fill_color, stroke_color, cluster.color))
            }) {
                continue;
            }
            let normalized_fill = normalize_text_color(fill_color);
            let normalized_stroke = normalize_text_color(stroke_color);
            if normalized_fill == normalized_stroke {
                continue;
            }

            let Some(stroke_width) =
                measured_stroke_width(&stroke_pixels, &enclosed_fill, &outside, width, height)
            else {
                continue;
            };
            let inside_fill = enclosed_fill
                .iter()
                .filter(|pixel| pixel.inside_mask)
                .count();
            let evidence = enclosed_fill.len() + inside_fill;
            let candidate = OutlinePaint {
                fill_color: normalized_fill,
                stroke_color: normalized_stroke,
                stroke_width,
                score: u64::from(contrast) * evidence.min(4096) as u64,
            };
            if best.is_none_or(|current: OutlinePaint| candidate.score > current.score) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn reachable_without_cluster(
    assignments: &[usize],
    barrier: usize,
    width: u32,
    height: u32,
) -> Vec<bool> {
    let mut reachable = vec![false; assignments.len()];
    let mut queue = VecDeque::new();
    for x in 0..width {
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            x,
            0,
            width,
        );
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            x,
            height - 1,
            width,
        );
    }
    for y in 0..height {
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            0,
            y,
            width,
        );
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            width - 1,
            y,
            width,
        );
    }
    while let Some((x, y)) = queue.pop_front() {
        if x > 0 {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x - 1,
                y,
                width,
            );
        }
        if x + 1 < width {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x + 1,
                y,
                width,
            );
        }
        if y > 0 {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x,
                y - 1,
                width,
            );
        }
        if y + 1 < height {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x,
                y + 1,
                width,
            );
        }
    }
    reachable
}

fn push_reachable(
    reachable: &mut [bool],
    queue: &mut VecDeque<(u32, u32)>,
    assignments: &[usize],
    barrier: usize,
    x: u32,
    y: u32,
    width: u32,
) {
    let index = y as usize * width as usize + x as usize;
    if !reachable[index] && assignments[index] != barrier {
        reachable[index] = true;
        queue.push_back((x, y));
    }
}

fn enclosing_cluster_pixels(
    assignments: &[usize],
    cluster: usize,
    cluster_pixels: &[MaskPixel],
    enclosed_fill: &[MaskPixel],
    width: u32,
    height: u32,
) -> Vec<MaskPixel> {
    let mut selected = vec![false; assignments.len()];
    let mut queue = VecDeque::new();
    for fill in enclosed_fill {
        for y in fill.y.saturating_sub(1)..=(fill.y + 1).min(height - 1) {
            for x in fill.x.saturating_sub(1)..=(fill.x + 1).min(width - 1) {
                let index = y as usize * width as usize + x as usize;
                if assignments[index] == cluster && !selected[index] {
                    selected[index] = true;
                    queue.push_back((x, y));
                }
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        for next_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for next_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let index = next_y as usize * width as usize + next_x as usize;
                if assignments[index] == cluster && !selected[index] {
                    selected[index] = true;
                    queue.push_back((next_x, next_y));
                }
            }
        }
    }
    cluster_pixels
        .iter()
        .copied()
        .filter(|pixel| selected[pixel.y as usize * width as usize + pixel.x as usize])
        .collect()
}

fn cluster_ink_score(cluster: &ColorCluster, background: [u8; 3]) -> u64 {
    let inside = cluster
        .pixels
        .iter()
        .filter(|pixel| pixel.inside_mask)
        .count();
    let evidence = if inside >= 8 {
        inside + cluster.pixels.len()
    } else {
        cluster.pixels.len()
    };
    u64::from(color_distance_squared(cluster.color, background)) * evidence.min(4096) as u64
}

fn representative_ink_color(cluster: &ColorCluster) -> [u8; 3] {
    let inside = cluster
        .pixels
        .iter()
        .copied()
        .filter(|pixel| pixel.inside_mask)
        .collect::<Vec<_>>();
    if inside.len() >= 8 {
        median_pixel_color(&inside)
    } else {
        median_pixel_color(&cluster.pixels)
    }
}

struct ColorCluster {
    color: [u8; 3],
    pixels: Vec<MaskPixel>,
}

fn color_clusters(
    pixels: &[MaskPixel],
    background: [u8; 3],
) -> [ColorCluster; COLOR_CLUSTER_COUNT] {
    let palette = color_palette(pixels);
    let darkest = extreme_palette_color(&palette, true);
    let lightest = extreme_palette_color(&palette, false);
    let distant = distant_palette_color(&palette, &[background, darkest, lightest]);
    let mut centers = [background, darkest, lightest, distant];
    for _ in 0..4 {
        let mut accumulators = [ColorAccumulator::default(); COLOR_CLUSTER_COUNT];
        for entry in &palette {
            let color = entry.color();
            let index = centers
                .iter()
                .enumerate()
                .min_by_key(|(_, center)| color_distance_squared(color, **center))
                .map(|(index, _)| index)
                .unwrap();
            accumulators[index].add(entry);
        }
        for (center, accumulator) in centers.iter_mut().zip(accumulators) {
            if let Some(color) = accumulator.color() {
                *center = color;
            }
        }
    }
    let mut groups: [Vec<MaskPixel>; COLOR_CLUSTER_COUNT] = std::array::from_fn(|_| Vec::new());
    for pixel in pixels {
        let index = centers
            .iter()
            .enumerate()
            .min_by_key(|(_, center)| color_distance_squared(pixel.color, **center))
            .map(|(index, _)| index)
            .unwrap();
        groups[index].push(*pixel);
    }
    std::array::from_fn(|index| ColorCluster {
        color: centers[index],
        pixels: std::mem::take(&mut groups[index]),
    })
}

fn extreme_palette_color(palette: &[ColorBin], darkest: bool) -> [u8; 3] {
    let select = |significant_only: bool| {
        let candidates = palette
            .iter()
            .filter(|bin| !significant_only || bin.count >= MIN_EXTREME_COLOR_PIXELS);
        if darkest {
            candidates.min_by_key(|bin| color_luminance(bin.color()))
        } else {
            candidates.max_by_key(|bin| color_luminance(bin.color()))
        }
    };
    select(true)
        .or_else(|| select(false))
        .map(ColorBin::color)
        .unwrap_or_default()
}

#[derive(Clone, Copy, Default)]
struct ColorBin {
    count: u32,
    sums: [u64; 3],
}

impl ColorBin {
    fn add(&mut self, color: [u8; 3]) {
        self.count += 1;
        for (sum, channel) in self.sums.iter_mut().zip(color) {
            *sum += u64::from(channel);
        }
    }

    fn color(&self) -> [u8; 3] {
        std::array::from_fn(|channel| {
            ((self.sums[channel] + u64::from(self.count / 2)) / u64::from(self.count)) as u8
        })
    }
}

#[derive(Clone, Copy, Default)]
struct ColorAccumulator {
    count: u64,
    sums: [u64; 3],
}

impl ColorAccumulator {
    fn add(&mut self, bin: &ColorBin) {
        self.count += u64::from(bin.count);
        for (sum, value) in self.sums.iter_mut().zip(bin.sums) {
            *sum += value;
        }
    }

    fn color(self) -> Option<[u8; 3]> {
        (self.count != 0).then(|| {
            std::array::from_fn(|channel| {
                ((self.sums[channel] + self.count / 2) / self.count) as u8
            })
        })
    }
}

fn color_palette(pixels: &[MaskPixel]) -> Vec<ColorBin> {
    let mut bins = vec![ColorBin::default(); 16 * 16 * 16];
    for pixel in pixels {
        let [red, green, blue] = pixel.color.map(|channel| usize::from(channel >> 4));
        bins[(red << 8) | (green << 4) | blue].add(pixel.color);
    }
    bins.retain(|bin| bin.count != 0);
    bins
}

fn distant_palette_color(palette: &[ColorBin], centers: &[[u8; 3]]) -> [u8; 3] {
    palette
        .iter()
        .max_by_key(|bin| {
            u64::from(minimum_color_distance(bin.color(), centers)) * u64::from(bin.count.min(64))
        })
        .map(ColorBin::color)
        .unwrap_or_default()
}

fn minimum_color_distance(color: [u8; 3], centers: &[[u8; 3]]) -> u32 {
    centers
        .iter()
        .map(|center| color_distance_squared(color, *center))
        .min()
        .unwrap_or_default()
}

fn median_pixel_color(pixels: &[MaskPixel]) -> [u8; 3] {
    let mut histograms = [[0_u32; 256]; 3];
    for pixel in pixels {
        for (histogram, channel) in histograms.iter_mut().zip(pixel.color) {
            histogram[usize::from(channel)] += 1;
        }
    }
    std::array::from_fn(|channel| histogram_median(&histograms[channel], pixels.len()))
}

fn measured_stroke_width(
    stroke_pixels: &[MaskPixel],
    fill_pixels: &[MaskPixel],
    outside: &[bool],
    width: u32,
    height: u32,
) -> Option<f32> {
    if stroke_pixels.len() < 8 || fill_pixels.len() < 8 {
        return None;
    }
    let mut fill_mask = GrayImage::from_pixel(width, height, Luma([0]));
    for pixel in fill_pixels {
        fill_mask.put_pixel(pixel.x, pixel.y, Luma([u8::MAX]));
    }
    let fill_distances = distance_transform(&fill_mask, Norm::L2);
    let mut histogram = [0_u32; 256];
    let mut count = 0_usize;
    for pixel in stroke_pixels {
        let mut touches_outside = false;
        for y in pixel.y.saturating_sub(1)..=(pixel.y + 1).min(height - 1) {
            for x in pixel.x.saturating_sub(1)..=(pixel.x + 1).min(width - 1) {
                if outside[y as usize * width as usize + x as usize] {
                    touches_outside = true;
                    break;
                }
            }
            if touches_outside {
                break;
            }
        }
        if !touches_outside {
            continue;
        }
        let distance = fill_distances.get_pixel(pixel.x, pixel.y).0[0];
        if distance == 0 {
            continue;
        }
        histogram[usize::from(distance)] += 1;
        count += 1;
    }
    if count < 8 {
        return None;
    }
    let width = histogram_median(&histogram, count);
    (width >= MIN_MEASURED_STROKE_WIDTH).then_some(f32::from(width))
}

fn color_lies_between(candidate: [u8; 3], start: [u8; 3], end: [u8; 3]) -> bool {
    let start = start.map(f64::from);
    let end = end.map(f64::from);
    let candidate = candidate.map(f64::from);
    let direction = std::array::from_fn::<_, 3, _>(|channel| end[channel] - start[channel]);
    let length_squared = direction.iter().map(|value| value * value).sum::<f64>();
    if length_squared <= f64::EPSILON {
        return false;
    }
    let projection = candidate
        .iter()
        .zip(start)
        .zip(direction)
        .map(|((&candidate, start), direction)| (candidate - start) * direction)
        .sum::<f64>()
        / length_squared;
    if !(0.0..=1.0).contains(&projection) {
        return false;
    }
    candidate
        .iter()
        .zip(start)
        .zip(direction)
        .map(|((&candidate, start), direction)| {
            let difference = candidate - (start + direction * projection);
            difference * difference
        })
        .sum::<f64>()
        <= 24.0_f64.powi(2) * 3.0
}

fn median_color(colors: &[[u8; 3]]) -> [u8; 3] {
    let mut histograms = [[0_u32; 256]; 3];
    for color in colors {
        for (histogram, channel) in histograms.iter_mut().zip(*color) {
            histogram[usize::from(channel)] += 1;
        }
    }
    std::array::from_fn(|channel| histogram_median(&histograms[channel], colors.len()))
}

fn color_distance_squared(left: [u8; 3], right: [u8; 3]) -> u32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| i32::from(left) - i32::from(right))
        .map(|difference| difference.unsigned_abs().pow(2))
        .sum()
}

fn normalize_text_color(color: [u8; 3]) -> [u8; 3] {
    let luminance = color_luminance(color);
    if luminance <= COLOR_SNAP_DARK_LUMINANCE {
        [0, 0, 0]
    } else if luminance >= COLOR_SNAP_LIGHT_LUMINANCE {
        [u8::MAX; 3]
    } else {
        color
    }
}

fn color_luminance(color: [u8; 3]) -> u32 {
    u32::from(color[0]) * 54 + u32::from(color[1]) * 183 + u32::from(color[2]) * 19
}

fn rectangle_geometry([left, top, right, bottom]: [f32; 4]) -> Geometry {
    Geometry::rectangle(
        f64::from(left),
        f64::from(top),
        f64::from((right - left).max(1.0)),
        f64::from((bottom - top).max(1.0)),
    )
}

/// The tightest rectangle at `angle_degrees` around the mask pixels. Rotating
/// the axis-aligned box instead would grow it past the text by up to its full
/// diagonal, past the page edge, and over neighboring dialogue.
fn oriented_extent(points: &[MaskPoint], angle_degrees: f32) -> [f64; 4] {
    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
    points.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |[min_u, min_v, max_u, max_v], point| {
            let u = point.x * cos + point.y * sin;
            let v = point.y * cos - point.x * sin;
            // Points are pixel centers; extend by half a pixel to cover them.
            [
                min_u.min(u - 0.5),
                min_v.min(v - 0.5),
                max_u.max(u + 0.5),
                max_v.max(v + 0.5),
            ]
        },
    )
}

fn rotated_rectangle_geometry(
    [min_u, min_v, max_u, max_v]: [f64; 4],
    angle_degrees: f32,
) -> Geometry {
    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
    Geometry {
        origin: Origin::User,
        points: [
            (min_u, min_v),
            (max_u, min_v),
            (max_u, max_v),
            (min_u, max_v),
        ]
        .map(|(u, v)| Point {
            x: u * cos - v * sin,
            y: u * sin + v * cos,
        })
        .into(),
    }
}

fn mask_geometry(mask: &KoharuLayoutMask) -> Option<Geometry> {
    let origin = (mask.x, mask.y);
    let mask = GrayImage::from_raw(mask.width, mask.height, mask.pixels.clone())?;
    let mut padded = GrayImage::new(mask.width() + 2, mask.height() + 2);
    image::imageops::replace(&mut padded, &mask, 1, 1);
    let contours = find_contours_with_threshold::<i32>(&padded, 0);
    let contour = contours
        .iter()
        .filter(|contour| contour.border_type == BorderType::Outer)
        .max_by(|left, right| {
            contour_area(&left.points)
                .partial_cmp(&contour_area(&right.points))
                .unwrap_or(Ordering::Equal)
        })?;
    if contour.points.len() < 3 {
        return None;
    }

    let epsilon = (arc_length(&contour.points, true) * 0.001).max(f64::EPSILON);
    let points = approximate_polygon_dp(&contour.points, epsilon, true)
        .into_iter()
        .map(|point| Point {
            x: f64::from(point.x - 1) + f64::from(origin.0),
            y: f64::from(point.y - 1) + f64::from(origin.1),
        })
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(Geometry {
        origin: Origin::User,
        points,
    })
}

async fn write_masks(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detections: &[KoharuLayoutDetection],
    size: ImageSize,
) -> Result<()> {
    write_mask(input, edit, page, detections, size).await?;
    edit.remove_asset(page, &AssetRole::new("bubble-mask")?)?;
    Ok(())
}

async fn write_mask(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detections: &[KoharuLayoutDetection],
    size: ImageSize,
) -> Result<()> {
    let mut mask = if size.width > 0 && size.height > 0 {
        let radius = ((size.width.max(size.height) as f32 / 1024.0) * 6.0)
            .round()
            .clamp(1.0, 255.0) as u8;
        closed_mask_for(detections, "text", size, radius)
    } else {
        mask_for(detections, "text", size)
    };
    if let Some(bounds) = input.region {
        preserve_mask_outside_region(input, page, "text-mask", bounds, &mut mask).await?;
    }
    let mut bytes = Cursor::new(Vec::new());
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, FilterType::NoFilter)
        .write_image(
            mask.as_raw(),
            mask.width(),
            mask.height(),
            ExtendedColorType::L8,
        )?;
    edit.set_asset(
        page,
        &AssetRole::new("text-mask")?,
        AssetInput::new(
            Arc::<[u8]>::from(bytes.into_inner()),
            "image/png",
            AssetMetadata {
                width: Some(size.width),
                height: Some(size.height),
                attributes: BTreeMap::new(),
            },
        ),
    )?;
    Ok(())
}

async fn preserve_mask_outside_region(
    input: &StageInput,
    page: EntityId,
    role: &str,
    bounds: crate::Bounds,
    mask: &mut GrayImage,
) -> Result<()> {
    let previous = input
        .images
        .get(&input.scene, page, role)
        .await?
        .map(|image| image.to_luma8());
    if previous
        .as_ref()
        .is_some_and(|image| image.dimensions() != mask.dimensions())
    {
        bail!("existing {role} dimensions do not match page {page}");
    }
    for (x, y, pixel) in mask.enumerate_pixels_mut() {
        if f64::from(x + 1) <= bounds.x
            || f64::from(y + 1) <= bounds.y
            || f64::from(x) >= bounds.x + bounds.width
            || f64::from(y) >= bounds.y + bounds.height
        {
            *pixel = previous
                .as_ref()
                .map_or(Luma([0]), |image| *image.get_pixel(x, y));
        }
    }
    Ok(())
}

fn region_kind(label: &str) -> Result<RegionKind> {
    RegionKind::new(match label {
        "text" | "onomatopoeia" => TextRegion::KIND,
        "bubble" => BubbleRegion::KIND,
        "panel" => PanelRegion::KIND,
        _ => "dev.koharu.region.unknown",
    })
    .map_err(Into::into)
}

fn mask_for(detections: &[KoharuLayoutDetection], label: &str, size: ImageSize) -> GrayImage {
    let mut mask = GrayImage::new(size.width, size.height);
    for detection in detections
        .iter()
        .filter(|detection| matches_mask_label(detection, label))
    {
        stamp_mask(&mut mask, &detection.mask, 0, 0);
    }
    mask
}

fn matches_mask_label(detection: &KoharuLayoutDetection, label: &str) -> bool {
    if label == "text" {
        is_text_detection(detection)
    } else {
        detection.label == label
    }
}

fn closed_mask_for(
    detections: &[KoharuLayoutDetection],
    label: &str,
    size: ImageSize,
    radius: u8,
) -> GrayImage {
    let masks = detections
        .iter()
        .filter(|detection| matches_mask_label(detection, label))
        .map(|detection| &detection.mask)
        .filter(|mask| valid_mask(mask) && mask.width > 0 && mask.height > 0)
        .collect::<Vec<_>>();
    let mut output = GrayImage::new(size.width, size.height);
    if masks.is_empty() {
        return output;
    }

    // Two dilations can make instances interact before the final erosion.
    // Conservatively group their expanded bounds, then retain a third-radius
    // halo so the local erosion observes the same zero/page-edge neighborhood
    // as a full-page operation.
    let interaction_radius = u32::from(radius) * 2;
    let mut parents = (0..masks.len()).collect::<Vec<_>>();
    for left in 0..masks.len() {
        for right in left + 1..masks.len() {
            if rectangles_touch(
                expand_mask_bounds(masks[left], interaction_radius, size),
                expand_mask_bounds(masks[right], interaction_radius, size),
            ) {
                union(&mut parents, left, right);
            }
        }
    }

    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..masks.len() {
        let root = find(&mut parents, index);
        groups.entry(root).or_default().push(index);
    }
    let local_masks = groups
        .into_values()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|group| {
            let mut bounds = mask_bounds(masks[group[0]], size);
            for &index in &group[1..] {
                bounds = union_bounds(bounds, mask_bounds(masks[index], size));
            }
            bounds = expand_bounds(bounds, u32::from(radius) * 3, size);
            let [left, top, right, bottom] = bounds;
            if left >= right || top >= bottom {
                return None;
            }
            let mut local = GrayImage::new(right - left, bottom - top);
            for &index in &group {
                stamp_mask(&mut local, masks[index], left, top);
            }
            let local = close(&dilate(&local, Norm::L2, radius), Norm::L2, radius);
            Some((left, top, local))
        })
        .collect::<Vec<_>>();
    for (left, top, local) in local_masks {
        for (x, y, pixel) in local.enumerate_pixels() {
            if pixel.0[0] != 0 {
                output.put_pixel(left + x, top + y, Luma([u8::MAX]));
            }
        }
    }
    output
}

/// Balloons with a dark fill and light lettering that the layout model did
/// not report. A text box whose surrounding ring is dark is grown by flood fill
/// over dark pixels; the fill must close inside a window around the text, or
/// it is page background rather than a balloon. Light balloons are left to the
/// model, since a light fill cannot be told from the page's own paper.
fn dark_balloons(
    image: &RgbImage,
    detections: &[KoharuLayoutDetection],
) -> Vec<KoharuLayoutDetection> {
    let bubbles = detections
        .iter()
        .filter(|detection| detection.label == "bubble")
        .collect::<Vec<_>>();
    let mut found = Vec::<KoharuLayoutDetection>::new();
    for text in detections
        .iter()
        .filter(|detection| matches!(detection.label.as_str(), "text" | "onomatopoeia"))
    {
        let inside = |bubble: &KoharuLayoutDetection| {
            mask_containment(&bubble.mask, &text.mask, text.bbox)
                >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
        };
        if bubbles.iter().any(|bubble| inside(bubble)) || found.iter().any(inside) {
            continue;
        }
        if let Some(balloon) = dark_balloon(image, text.bbox) {
            found.push(balloon);
        }
    }
    found
}

fn dark_balloon(image: &RgbImage, bbox: [f32; 4]) -> Option<KoharuLayoutDetection> {
    let (width, height) = image.dimensions();
    let [left, top, right, bottom] = mask_window(bbox, width, height)?;
    let side = (right - left).max(bottom - top);
    let ring = ((right - left).min(bottom - top) / 10).max(2);
    let reach = side + side / 2;
    let window = [
        left.saturating_sub(reach),
        top.saturating_sub(reach),
        right.saturating_add(reach).min(width),
        bottom.saturating_add(reach).min(height),
    ];
    let [window_left, window_top, window_right, window_bottom] = window;
    let local_width = window_right - window_left;
    let local_height = window_bottom - window_top;
    let dark = |x: u32, y: u32| {
        let [red, green, blue] = image.get_pixel(x, y).0;
        let luminance =
            (u32::from(red) * 299 + u32::from(green) * 587 + u32::from(blue) * 114) / 1000;
        luminance < u32::from(DARK_BALLOON_LUMINANCE)
    };
    let in_box = |x: u32, y: u32| x >= left && x < right && y >= top && y < bottom;

    let mut seeds = Vec::new();
    let mut ring_pixels = 0_u32;
    for y in top.saturating_sub(ring)..(bottom + ring).min(height) {
        for x in left.saturating_sub(ring)..(right + ring).min(width) {
            if in_box(x, y) {
                continue;
            }
            ring_pixels += 1;
            if dark(x, y) {
                seeds.push((x, y));
            }
        }
    }
    if ring_pixels == 0 {
        return None;
    }
    // The share of dark ring pixels stands in for a model confidence.
    let score = seeds.len() as f32 / ring_pixels as f32;
    if score < DARK_BALLOON_RING_FRACTION {
        return None;
    }

    let index = |x: u32, y: u32| ((y - window_top) * local_width + (x - window_left)) as usize;
    let mut filled = vec![0_u8; (local_width * local_height) as usize];
    for y in top..bottom {
        for x in left..right {
            filled[index(x, y)] = 1;
        }
    }
    let mut stack = seeds;
    for &(x, y) in &stack {
        filled[index(x, y)] = 1;
    }
    while let Some((x, y)) = stack.pop() {
        // Reaching the window edge means the dark area keeps going: it is a
        // dark panel or background, not a closed balloon.
        if (x == window_left && x > 0)
            || (y == window_top && y > 0)
            || (x + 1 == window_right && window_right < width)
            || (y + 1 == window_bottom && window_bottom < height)
            || x == 0
            || y == 0
            || x + 1 == width
            || y + 1 == height
        {
            return None;
        }
        for (next_x, next_y) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
            let slot = index(next_x, next_y);
            if filled[slot] == 0 && dark(next_x, next_y) {
                filled[slot] = 1;
                stack.push((next_x, next_y));
            }
        }
    }

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0, 0);
    let mut area = 0_u32;
    for local_y in 0..local_height {
        for local_x in 0..local_width {
            if filled[(local_y * local_width + local_x) as usize] != 0 {
                area += 1;
                min_x = min_x.min(local_x);
                min_y = min_y.min(local_y);
                max_x = max_x.max(local_x);
                max_y = max_y.max(local_y);
            }
        }
    }
    let text_area = (right - left) * (bottom - top);
    if (area as f32) < text_area as f32 * DARK_BALLOON_MIN_GROWTH {
        return None;
    }
    let mask_width = max_x - min_x + 1;
    let mask_height = max_y - min_y + 1;
    let pixels = (min_y..=max_y)
        .flat_map(|local_y| {
            let filled = &filled;
            (min_x..=max_x).map(move |local_x| filled[(local_y * local_width + local_x) as usize])
        })
        .collect();
    let x = window_left + min_x;
    let y = window_top + min_y;
    Some(KoharuLayoutDetection {
        label_id: 0,
        label: "bubble".to_owned(),
        score,
        bbox: [
            x as f32,
            y as f32,
            (x + mask_width) as f32,
            (y + mask_height) as f32,
        ],
        area,
        mask: KoharuLayoutMask {
            x,
            y,
            width: mask_width,
            height: mask_height,
            pixels,
        },
    })
}

fn valid_mask(mask: &KoharuLayoutMask) -> bool {
    mask.width
        .checked_mul(mask.height)
        .and_then(|count| usize::try_from(count).ok())
        == Some(mask.pixels.len())
}

fn stamp_mask(target: &mut GrayImage, mask: &KoharuLayoutMask, origin_x: u32, origin_y: u32) {
    if !valid_mask(mask) {
        return;
    }
    for local_y in 0..mask.height {
        let Some(page_y) = mask.y.checked_add(local_y) else {
            continue;
        };
        let Some(target_y) = page_y.checked_sub(origin_y) else {
            continue;
        };
        if target_y >= target.height() {
            continue;
        }
        let row = local_y as usize * mask.width as usize;
        for local_x in 0..mask.width {
            let Some(page_x) = mask.x.checked_add(local_x) else {
                continue;
            };
            let Some(target_x) = page_x.checked_sub(origin_x) else {
                continue;
            };
            if target_x < target.width() && mask.pixels[row + local_x as usize] != 0 {
                target.put_pixel(target_x, target_y, Luma([u8::MAX]));
            }
        }
    }
}

fn mask_bounds(mask: &KoharuLayoutMask, size: ImageSize) -> [u32; 4] {
    [
        mask.x.min(size.width),
        mask.y.min(size.height),
        mask.x.saturating_add(mask.width).min(size.width),
        mask.y.saturating_add(mask.height).min(size.height),
    ]
}

fn expand_mask_bounds(mask: &KoharuLayoutMask, radius: u32, size: ImageSize) -> [u32; 4] {
    expand_bounds(mask_bounds(mask, size), radius, size)
}

fn expand_bounds([left, top, right, bottom]: [u32; 4], radius: u32, size: ImageSize) -> [u32; 4] {
    [
        left.saturating_sub(radius),
        top.saturating_sub(radius),
        right.saturating_add(radius).min(size.width),
        bottom.saturating_add(radius).min(size.height),
    ]
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn rectangles_touch(left: [u32; 4], right: [u32; 4]) -> bool {
    left[0] <= right[2] && right[0] <= left[2] && left[1] <= right[3] && right[1] <= left[3]
}

fn find(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left = find(parents, left);
    let right = find(parents, right);
    if left != right {
        parents[right] = left;
    }
}

fn intersects([left, top, right, bottom]: [f32; 4], region: crate::Bounds) -> bool {
    left < (region.x + region.width) as f32
        && right > region.x as f32
        && top < (region.y + region.height) as f32
        && bottom > region.y as f32
}

fn detection_order(left: &KoharuLayoutDetection, right: &KoharuLayoutDetection) -> Ordering {
    left.bbox[1]
        .total_cmp(&right.bbox[1])
        .then_with(|| right.bbox[0].total_cmp(&left.bbox[0]))
        .then_with(|| left.label.cmp(&right.label))
}

fn non_maximum_suppression(detections: &mut Vec<KoharuLayoutDetection>, threshold: f32) {
    detections.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| detection_order(left, right))
    });
    let mut kept = Vec::with_capacity(detections.len());
    for candidate in detections.drain(..) {
        let suppressed = kept.iter().any(|existing: &KoharuLayoutDetection| {
            let same_label = existing.label == candidate.label;
            if is_text_detection(existing) && is_text_detection(&candidate) {
                let smaller = area(existing.bbox).min(area(candidate.bbox));
                let larger = area(existing.bbox).max(area(candidate.bbox));
                if larger <= 0.0 || smaller / larger < TEXT_DUPLICATE_AREA_RATIO_THRESHOLD {
                    return false;
                }
            } else if !same_label {
                return false;
            }
            let threshold = if same_label {
                threshold
            } else {
                TEXT_DUPLICATE_IOU_THRESHOLD
            };
            intersection_over_union(existing.bbox, candidate.bbox) >= threshold
        });
        if !suppressed {
            kept.push(candidate);
        }
    }
    *detections = kept;
}

/// The detector can return boxes that extend past the page. Clip them so that
/// region geometry, OCR crops and inpainting stay on the image, and drop boxes
/// that lie entirely outside it.
fn clamp_to_image(detections: &mut Vec<KoharuLayoutDetection>, size: ImageSize) {
    let (width, height) = (size.width as f32, size.height as f32);
    for detection in detections.iter_mut() {
        let [left, top, right, bottom] = detection.bbox;
        detection.bbox = [
            left.clamp(0.0, width),
            top.clamp(0.0, height),
            right.clamp(0.0, width),
            bottom.clamp(0.0, height),
        ];
    }
    detections.retain(|detection| area(detection.bbox) > 0.0);
}

/// On real pages the detector returns low-confidence onomatopoeia boxes that
/// span several lines of dialogue, several smaller effects, or a large part of
/// the page. Below `SPURIOUS_ONOMATOPOEIA_MAX_SCORE`, a box is dropped when it
/// encloses at least half of two text or onomatopoeia boxes scored at least as
/// high as itself, or when it covers more than
/// `SPURIOUS_ONOMATOPOEIA_PAGE_FRACTION` of the page and encloses one. Box
/// containment is used, not masks: such a box's mask covers its own strokes,
/// not the dialogue it spans. A confident effect, or a large one that covers
/// no other text, such as a title, is kept.
fn remove_spurious_onomatopoeia(detections: &mut Vec<KoharuLayoutDetection>, size: ImageSize) {
    let page_area = size.width as f32 * size.height as f32;
    let spurious = detections
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            if candidate.label != "onomatopoeia"
                || candidate.score >= SPURIOUS_ONOMATOPOEIA_MAX_SCORE
            {
                return false;
            }
            let enclosed = detections
                .iter()
                .enumerate()
                .filter(|&(other_index, other)| {
                    let other_area = area(other.bbox);
                    other_index != index
                        && is_text_detection(other)
                        && other.score >= candidate.score
                        && other_area > 0.0
                        && intersection_area(candidate.bbox, other.bbox) / other_area
                            >= SPURIOUS_ONOMATOPOEIA_CONTAINMENT
                })
                .count();
            enclosed >= 2
                || enclosed >= 1
                    && area(candidate.bbox) > page_area * SPURIOUS_ONOMATOPOEIA_PAGE_FRACTION
        })
        .collect::<Vec<_>>();
    let mut index = 0usize;
    detections.retain(|_| {
        let keep = !spurious[index];
        index += 1;
        keep
    });
}

/// Scanned chapters print the page number as a narrow column of digits
/// against the left or right page edge, in the top or bottom margin, and the
/// detector reports it as text. It is not part of the work, so a text box that
/// touches a side edge within `PAGE_NUMBER_EDGE_FRACTION` of the page width,
/// is narrower than `PAGE_NUMBER_MAX_WIDTH_FRACTION` of it, is shorter than
/// `PAGE_NUMBER_MAX_HEIGHT_FRACTION` of the page height, and sits in the outer
/// `PAGE_NUMBER_MARGIN_FRACTION` of the height is dropped.
fn remove_page_numbers(detections: &mut Vec<KoharuLayoutDetection>, size: ImageSize) {
    let (width, height) = (size.width as f32, size.height as f32);
    detections.retain(|detection| {
        let [left, top, right, bottom] = detection.bbox;
        let center_y = (top + bottom) / 2.0;
        let page_number = detection.label == "text"
            && (left <= width * PAGE_NUMBER_EDGE_FRACTION
                || right >= width * (1.0 - PAGE_NUMBER_EDGE_FRACTION))
            && right - left < width * PAGE_NUMBER_MAX_WIDTH_FRACTION
            && bottom - top < height * PAGE_NUMBER_MAX_HEIGHT_FRACTION
            && (center_y < height * PAGE_NUMBER_MARGIN_FRACTION
                || center_y > height * (1.0 - PAGE_NUMBER_MARGIN_FRACTION));
        !page_number
    });
}

fn remove_nested_text_detections(detections: &mut Vec<KoharuLayoutDetection>) {
    let nested = detections
        .iter()
        .enumerate()
        .map(|(child_index, child)| {
            let child_area = area(child.bbox);
            is_text_detection(child)
                && child_area > 0.0
                && detections.iter().enumerate().any(|(parent_index, parent)| {
                    parent_index != child_index
                        && is_text_detection(parent)
                        && area(parent.bbox) > child_area
                        && mask_containment(&parent.mask, &child.mask, child.bbox)
                            >= NESTED_TEXT_CONTAINMENT_THRESHOLD
                })
        })
        .collect::<Vec<_>>();
    let mut index = 0usize;
    detections.retain(|_| {
        let keep = !nested[index];
        index += 1;
        keep
    });
}

fn is_text_detection(detection: &KoharuLayoutDetection) -> bool {
    matches!(detection.label.as_str(), "text" | "onomatopoeia")
}

#[derive(Clone, Copy)]
enum ReadingOrder {
    RowMajor,
    ColumnMajor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LayoutKind {
    Panel,
    Bubble,
    Text,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LayoutItem {
    pub kind: LayoutKind,
    pub bounds: [f64; 4],
}

impl ReadingOrder {
    fn for_language(language: Language) -> Self {
        match language {
            Language::Japanese | Language::ChineseSimplified | Language::ChineseTraditional => {
                Self::ColumnMajor
            }
            _ => Self::RowMajor,
        }
    }
}

fn sort_by_layout(detections: &mut Vec<KoharuLayoutDetection>, source_language: Language) {
    let order = layout_order(detections, source_language);
    let mut values = std::mem::take(detections)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    *detections = order
        .into_iter()
        .map(|index| {
            values[index]
                .take()
                .expect("layout order contains each detection once")
        })
        .collect();
}

fn layout_order(detections: &[KoharuLayoutDetection], source_language: Language) -> Vec<usize> {
    let items = detections
        .iter()
        .map(|detection| LayoutItem {
            kind: match detection.label.as_str() {
                "panel" => LayoutKind::Panel,
                "bubble" => LayoutKind::Bubble,
                "text" | "onomatopoeia" => LayoutKind::Text,
                _ => LayoutKind::Other,
            },
            bounds: detection.bbox.map(f64::from),
        })
        .collect::<Vec<_>>();
    reading_order(&items, source_language)
}

pub(super) fn reading_order(items: &[LayoutItem], source_language: Language) -> Vec<usize> {
    let reading_order = ReadingOrder::for_language(source_language);
    let panels = indices_with_kind(items, LayoutKind::Panel);
    let bubbles = indices_with_kind(items, LayoutKind::Bubble);
    let texts = indices_with_kind(items, LayoutKind::Text);

    let mut panel_for_bubble = vec![None; items.len()];
    for &bubble in &bubbles {
        panel_for_bubble[bubble] = best_container(items, bubble, &panels);
    }
    let mut bubble_for_text = vec![None; items.len()];
    let mut panel_for_text = vec![None; items.len()];
    for &text in &texts {
        let bubble = best_container(items, text, &bubbles);
        bubble_for_text[text] = bubble;
        panel_for_text[text] = bubble
            .and_then(|bubble| panel_for_bubble[bubble])
            .or_else(|| best_container(items, text, &panels));
    }

    let mut parent = vec![None; items.len()];
    for bubble in bubbles {
        parent[bubble] = panel_for_bubble[bubble];
    }
    for text in texts {
        parent[text] = bubble_for_text[text].or(panel_for_text[text]);
    }

    let mut roots = Vec::new();
    let mut children = vec![Vec::new(); items.len()];
    for (index, parent) in parent.into_iter().enumerate() {
        if let Some(parent) = parent {
            children[parent].push(index);
        } else {
            roots.push(index);
        }
    }
    sort_spatial(items, &mut roots, reading_order);
    for siblings in &mut children {
        sort_spatial(items, siblings, reading_order);
    }

    let mut order = Vec::with_capacity(items.len());
    for root in roots {
        append_layout_subtree(root, &children, &mut order);
    }
    order
}

pub fn text_layer_placement(
    snapshot: &koharu_scene::Snapshot,
    page: EntityId,
    geometry: &Geometry,
    source_language: Language,
) -> Result<At> {
    let Some(group) = snapshot.page(page)?.text_group()? else {
        return Ok(At::End);
    };
    let mut items = Vec::new();
    let mut layers = Vec::new();

    for entity in snapshot.descendants(page)? {
        let Some(region) = snapshot.component::<Region>(entity.id())? else {
            continue;
        };
        let kind = if region.kind == PanelRegion::kind() {
            LayoutKind::Panel
        } else if region.kind == BubbleRegion::kind() {
            LayoutKind::Bubble
        } else {
            continue;
        };
        let Some(geometry) = snapshot.component::<Geometry>(entity.id())? else {
            continue;
        };
        let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&geometry) else {
            continue;
        };
        items.push(LayoutItem {
            kind,
            bounds: [left, top, right, bottom],
        });
        layers.push(None);
    }

    for layer in group.text_layers()? {
        let Some(region) = layer.content()?.source_region()? else {
            continue;
        };
        let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&region.geometry()?)
        else {
            continue;
        };
        items.push(LayoutItem {
            kind: LayoutKind::Text,
            bounds: [left, top, right, bottom],
        });
        layers.push(Some(layer.id()));
    }

    let Some((left, top, right, bottom)) = crate::scope::geometry_extents(geometry) else {
        return Ok(At::End);
    };
    let inserted = items.len();
    items.push(LayoutItem {
        kind: LayoutKind::Text,
        bounds: [left, top, right, bottom],
    });
    layers.push(None);

    let mut after_inserted = false;
    for index in reading_order(&items, source_language) {
        if index == inserted {
            after_inserted = true;
        } else if after_inserted
            && items[index].kind == LayoutKind::Text
            && let Some(layer) = layers[index]
        {
            return Ok(At::Before(layer));
        }
    }
    Ok(At::End)
}

fn indices_with_kind(items: &[LayoutItem], kind: LayoutKind) -> Vec<usize> {
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (item.kind == kind).then_some(index))
        .collect()
}

fn append_layout_subtree(index: usize, children: &[Vec<usize>], order: &mut Vec<usize>) {
    order.push(index);
    for &child in &children[index] {
        append_layout_subtree(child, children, order);
    }
}

fn sort_spatial(items: &[LayoutItem], indices: &mut [usize], reading_order: ReadingOrder) {
    if indices.len() < 2 {
        return;
    }

    let ordered_panels = {
        let mut panels = indices
            .iter()
            .copied()
            .filter(|&index| items[index].kind == LayoutKind::Panel)
            .collect::<Vec<_>>();
        if panels.len() > 1 && panels.len() < indices.len() {
            sort_spatial(items, &mut panels, reading_order);
            Some(panels)
        } else {
            None
        }
    };

    if indices
        .iter()
        .all(|&index| items[index].kind == LayoutKind::Panel)
        && let Some(groups) = panel_gap_groups(items, indices, reading_order)
    {
        let mut result = Vec::with_capacity(indices.len());
        for mut group in groups {
            sort_spatial(items, &mut group, reading_order);
            result.extend(group);
        }
        indices.copy_from_slice(&result);
        return;
    }

    let axis = match reading_order {
        ReadingOrder::ColumnMajor => SpatialAxis::X,
        ReadingOrder::RowMajor => SpatialAxis::Y,
    };
    let mut ordered = indices.to_vec();
    ordered.sort_by(|&left, &right| {
        let left_center = axis_center(&items[left], axis);
        let right_center = axis_center(&items[right], axis);
        let center_order = match reading_order {
            ReadingOrder::ColumnMajor => right_center.total_cmp(&left_center),
            ReadingOrder::RowMajor => left_center.total_cmp(&right_center),
        };
        center_order
            .then_with(|| {
                let left_cross = cross_axis_start(&items[left], axis);
                let right_cross = cross_axis_start(&items[right], axis);
                left_cross.total_cmp(&right_cross)
            })
            .then_with(|| left.cmp(&right))
    });

    let tolerance = grouping_tolerance(items, &ordered, axis);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for index in ordered {
        let center = axis_center(&items[index], axis);
        let group = groups.iter().position(|group| {
            (center - group_center(items, group, axis)).abs() <= tolerance
                || group.iter().any(|&item| {
                    ranges_overlap(
                        axis_bounds(&items[index], axis),
                        axis_bounds(&items[item], axis),
                    )
                })
        });
        if let Some(group) = group {
            groups[group].push(index);
        } else {
            groups.push(vec![index]);
        }
    }

    groups.sort_by(|left, right| {
        let left_center = group_center(items, left, axis);
        let right_center = group_center(items, right, axis);
        let center_order = match reading_order {
            ReadingOrder::ColumnMajor => right_center.total_cmp(&left_center),
            ReadingOrder::RowMajor => left_center.total_cmp(&right_center),
        };
        center_order.then_with(|| left.iter().min().cmp(&right.iter().min()))
    });

    let mut result = Vec::with_capacity(indices.len());
    for group in &mut groups {
        group.sort_by(|&left, &right| {
            let left_cross = cross_axis_start(&items[left], axis);
            let right_cross = cross_axis_start(&items[right], axis);
            left_cross
                .total_cmp(&right_cross)
                .then_with(|| {
                    axis_start(&items[left], axis).total_cmp(&axis_start(&items[right], axis))
                })
                .then_with(|| left.cmp(&right))
        });
        result.extend(group.iter().copied());
    }
    if let Some(panels) = ordered_panels {
        let mut panels = panels.into_iter();
        for index in &mut result {
            if items[*index].kind == LayoutKind::Panel {
                *index = panels.next().expect("panel order has matching length");
            }
        }
    }
    indices.copy_from_slice(&result);
}

#[derive(Clone, Copy)]
enum SpatialAxis {
    X,
    Y,
}

fn panel_gap_groups(
    items: &[LayoutItem],
    indices: &[usize],
    reading_order: ReadingOrder,
) -> Option<Vec<Vec<usize>>> {
    let axes = match reading_order {
        ReadingOrder::ColumnMajor => [SpatialAxis::X, SpatialAxis::Y],
        ReadingOrder::RowMajor => [SpatialAxis::Y, SpatialAxis::X],
    };
    for axis in axes {
        let mut ordered = indices.to_vec();
        ordered.sort_by(|&left, &right| {
            axis_start(&items[left], axis)
                .total_cmp(&axis_start(&items[right], axis))
                .then_with(|| left.cmp(&right))
        });

        let mut groups = Vec::<Vec<usize>>::new();
        let mut group_start = f64::NEG_INFINITY;
        let mut end = f64::NEG_INFINITY;
        for index in ordered {
            let (start, item_end) = axis_bounds(&items[index], axis);
            // Detector boxes can bleed a few pixels across an otherwise visible panel gutter.
            let seam_tolerance = (end - group_start).min(item_end - start).max(0.0) * 0.02;
            let starts_new_group =
                groups.is_empty() || start >= end || end - start <= seam_tolerance;
            if starts_new_group {
                groups.push(Vec::new());
                group_start = start;
                end = item_end;
            } else {
                end = end.max(item_end);
            }
            groups.last_mut().expect("panel group exists").push(index);
        }
        if groups.len() > 1 {
            if matches!(
                (reading_order, axis),
                (ReadingOrder::ColumnMajor, SpatialAxis::X)
            ) {
                groups.reverse();
            }
            return Some(groups);
        }
    }
    None
}

fn axis_bounds(item: &LayoutItem, axis: SpatialAxis) -> (f64, f64) {
    match axis {
        SpatialAxis::X => (item.bounds[0], item.bounds[2]),
        SpatialAxis::Y => (item.bounds[1], item.bounds[3]),
    }
}

fn axis_start(item: &LayoutItem, axis: SpatialAxis) -> f64 {
    axis_bounds(item, axis).0
}

fn axis_center(item: &LayoutItem, axis: SpatialAxis) -> f64 {
    let (start, end) = axis_bounds(item, axis);
    (start + end) * 0.5
}

fn cross_axis_start(item: &LayoutItem, axis: SpatialAxis) -> f64 {
    match axis {
        SpatialAxis::X => item.bounds[1],
        SpatialAxis::Y => item.bounds[0],
    }
}

fn grouping_tolerance(items: &[LayoutItem], indices: &[usize], axis: SpatialAxis) -> f64 {
    let average_span = indices
        .iter()
        .map(|&index| {
            let (start, end) = axis_bounds(&items[index], axis);
            (end - start).max(1.0)
        })
        .sum::<f64>()
        / indices.len() as f64;
    (average_span * 0.85).max(24.0)
}

fn group_center(items: &[LayoutItem], group: &[usize], axis: SpatialAxis) -> f64 {
    group
        .iter()
        .map(|&index| axis_center(&items[index], axis))
        .sum::<f64>()
        / group.len().max(1) as f64
}

fn ranges_overlap(left: (f64, f64), right: (f64, f64)) -> bool {
    left.1.min(right.1) > left.0.max(right.0)
}

fn best_container(items: &[LayoutItem], value: usize, candidates: &[usize]) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .filter_map(|candidate| {
            let containment = containment(items[candidate].bounds, items[value].bounds);
            (containment >= 0.5).then_some((candidate, containment))
        })
        .min_by(|&(left, left_containment), &(right, right_containment)| {
            right_containment
                .total_cmp(&left_containment)
                .then_with(|| {
                    layout_area(items[left].bounds).total_cmp(&layout_area(items[right].bounds))
                })
                .then_with(|| left.cmp(&right))
        })
        .map(|(candidate, _)| candidate)
}

fn containment(container: [f64; 4], value: [f64; 4]) -> f64 {
    let value_area = layout_area(value);
    if value_area <= 0.0 {
        return 0.0;
    }
    layout_intersection_area(container, value) / value_area
}

fn layout_intersection_area(left: [f64; 4], right: [f64; 4]) -> f64 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0)
}

fn layout_area(bounds: [f64; 4]) -> f64 {
    (bounds[2] - bounds[0]).max(0.0) * (bounds[3] - bounds[1]).max(0.0)
}

fn mask_containment(
    container: &KoharuLayoutMask,
    value: &KoharuLayoutMask,
    bounds: [f32; 4],
) -> f32 {
    if !valid_mask(container) || !valid_mask(value) {
        return 0.0;
    }

    if !bounds.iter().all(|value| value.is_finite()) {
        return 0.0;
    }
    let value_right = value.x.saturating_add(value.width);
    let value_bottom = value.y.saturating_add(value.height);
    let left = bounds[0]
        .floor()
        .max(value.x as f32)
        .min(value_right as f32) as u32;
    let top = bounds[1]
        .floor()
        .max(value.y as f32)
        .min(value_bottom as f32) as u32;
    let right = bounds[2].ceil().max(value.x as f32).min(value_right as f32) as u32;
    let bottom = bounds[3]
        .ceil()
        .max(value.y as f32)
        .min(value_bottom as f32) as u32;
    if left >= right || top >= bottom {
        return 0.0;
    }

    let mut value_area = 0usize;
    let mut intersection = 0usize;
    for y in top..bottom {
        for x in left..right {
            if !value.contains(x, y) {
                continue;
            }
            value_area += 1;
            intersection += usize::from(container.contains(x, y));
        }
    }
    if value_area == 0 {
        0.0
    } else {
        intersection as f32 / value_area as f32
    }
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection = intersection_area(left, right);
    let union = area(left) + area(right) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn intersection_area(left: [f32; 4], right: [f32; 4]) -> f32 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0)
}

fn area(bounds: [f32; 4]) -> f32 {
    (bounds[2] - bounds[0]).max(0.0) * (bounds[3] - bounds[1]).max(0.0)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use imageproc::{
        distance_transform::Norm,
        morphology::{close, dilate},
    };
    use koharu_ml::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};
    use koharu_scene::{
        At, BubbleRegion, FitsTo, FlowsIn, Geometry, Inside, Origin, PageDraft, Region, RegionSpec,
        Session, TextLayout, TextLayoutKind, TextRegion, TextRole, Typography, WritingMode,
    };
    use koharu_translator::Language;

    use super::{
        DIALOGUE_MASK_CONTAINMENT_THRESHOLD, DetectedRegion, DetectedText, DetectionModel,
        FREE_TEXT_ROLE, ImageSize, KoharuLayoutRFDetrSeg2XLConfig, LayoutItem, LayoutKind,
        MaskPixel, ONOMATOPOEIA_ROLE, PageRegions, Processor, RecognizedFrom, RegionOutput,
        StageInput, StageProcessor, best_container, clamp_to_image, closed_mask_for, color_palette,
        dark_balloons, generation, infer_typography, layout_order, link_dialogue_regions,
        mask_containment, mask_for, mask_geometry, non_maximum_suppression, normalize_text_color,
        remove_nested_text_detections, remove_page_numbers, remove_spurious_onomatopoeia,
        write_region,
    };

    #[test]
    fn out_of_range_thresholds_fall_back_to_the_model_defaults() {
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(15.0),
                bubble_threshold: Some(f32::NAN),
                panel_threshold: Some(0.55),
            }),
            Language::English,
            koharu_ml::Device::cpu(),
        );

        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &processor.config;
        assert_eq!(settings.text_threshold, None);
        assert_eq!(settings.bubble_threshold, None);
        assert_eq!(settings.panel_threshold, Some(0.55));
    }

    #[tokio::test]
    async fn detection_skips_a_page_with_existing_text() {
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                edit.add_analysis_region::<TextRegion>(
                    id,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
                    None,
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            std::sync::Arc::new(crate::ImageCache::default()),
            None,
        );
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            Language::English,
            koharu_ml::Device::cpu(),
        );

        assert!(processor.skip(&input).unwrap());
    }

    #[tokio::test]
    async fn detection_does_not_skip_an_orphaned_generated_text_region() {
        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let owner = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
        let mut edit = session.snapshot().edit_as(owner);
        edit.add_analysis_region::<TextRegion>(
            page,
            At::End,
            &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
            None,
        )
        .unwrap();
        let snapshot = session
            .commit(edit.finish().unwrap())
            .await
            .unwrap()
            .snapshot;
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            std::sync::Arc::new(crate::ImageCache::default()),
            None,
        );
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            Language::English,
            koharu_ml::Device::cpu(),
        );

        assert!(!processor.skip(&input).unwrap());
    }

    #[tokio::test]
    async fn detection_does_not_skip_generated_region_without_text_presentation() {
        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let owner = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
        let mut edit = session.snapshot().edit_as(owner);
        let region = edit
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
                None,
            )
            .unwrap();
        let content = edit.add_text_content(page, At::End).unwrap();
        edit.relate::<RecognizedFrom>(content, region).unwrap();
        let snapshot = session
            .commit(edit.finish().unwrap())
            .await
            .unwrap()
            .snapshot;
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            std::sync::Arc::new(crate::ImageCache::default()),
            None,
        );
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            Language::English,
            koharu_ml::Device::cpu(),
        );

        assert!(!processor.skip(&input).unwrap());
    }

    #[tokio::test]
    async fn detection_rebuilds_cached_exact_text_duplicates() {
        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let owner = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
        let detected = |label: &str, score| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score,
            bbox: [10.0, 10.0, 30.0, 30.0],
            area: 400,
            mask: KoharuLayoutMask {
                x: 10,
                y: 10,
                width: 20,
                height: 20,
                pixels: vec![u8::MAX; 400],
            },
        };
        let mut edit = session.snapshot().edit_as(owner.clone());
        write_region(&mut edit, page, &detected("text", 0.9), None, &owner).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            Language::Japanese,
            koharu_ml::Device::cpu(),
        );
        let input = |session: &Session| {
            StageInput::new(
                session.snapshot(),
                page,
                None,
                None,
                std::sync::Arc::new(crate::ImageCache::default()),
                None,
            )
        };
        assert!(processor.skip(&input(&session)).unwrap());

        let mut edit = session.snapshot().edit_as(owner.clone());
        write_region(
            &mut edit,
            page,
            &detected("onomatopoeia", 0.8),
            None,
            &owner,
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert!(!processor.skip(&input(&session)).unwrap());
    }

    #[tokio::test]
    async fn detection_rebuilds_results_from_the_older_nested_region_postprocessor() {
        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let owner = generation(super::PRODUCER, super::LEGACY_OUTPUT_MODEL_ID).unwrap();
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 0.9,
            bbox: [10.0, 10.0, 30.0, 30.0],
            area: 400,
            mask: KoharuLayoutMask {
                x: 10,
                y: 10,
                width: 20,
                height: 20,
                pixels: vec![u8::MAX; 400],
            },
        };
        let mut edit = session.snapshot().edit_as(owner.clone());
        write_region(&mut edit, page, &detection, None, &owner).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            Language::Japanese,
            koharu_ml::Device::cpu(),
        );
        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            std::sync::Arc::new(crate::ImageCache::default()),
            None,
        );

        assert!(!processor.skip(&input).unwrap());
    }

    #[tokio::test]
    async fn joined_text_uses_balloon_flow_semantics() {
        let mut session = Session::memory().await.unwrap();
        let bubble_mask = KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            pixels: vec![u8::MAX],
        };
        let text_mask = bubble_mask.clone();
        let generation = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                let bubble = edit.add_analysis_region::<BubbleRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, 80.0, 80.0),
                    None,
                )?;
                let region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(30.0, 30.0, 20.0, 20.0),
                    None,
                )?;
                let content = edit.add_text_content(page, At::End)?;
                let layer = edit.add_text_layer(
                    page,
                    At::End,
                    content,
                    &TextLayout {
                        origin: Origin::User,
                        kind: TextLayoutKind::Paragraph,
                        angle_degrees: None,
                    },
                )?;
                link_dialogue_regions(
                    edit,
                    &PageRegions {
                        bubbles: vec![DetectedRegion {
                            entity: bubble,
                            mask: &bubble_mask,
                            area: 1,
                        }],
                        texts: vec![DetectedText {
                            entity: region,
                            mask: &text_mask,
                            bounds: [0.0, 0.0, 1.0, 1.0],
                            content,
                            layer,
                            role: FREE_TEXT_ROLE,
                        }],
                    },
                    &generation,
                )
                .unwrap();
                ids = Some((bubble, region, layer));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let (bubble, region, layer) = ids.unwrap();

        assert_eq!(
            snapshot
                .relation_from::<FlowsIn>(layer)
                .unwrap()
                .unwrap()
                .value()
                .target,
            bubble
        );
        assert!(snapshot.relation_from::<FitsTo>(layer).unwrap().is_none());
        assert_eq!(
            snapshot
                .relations_from_as::<Inside>(region)
                .next()
                .unwrap()
                .value()
                .target,
            bubble
        );
    }

    #[tokio::test]
    async fn onomatopoeia_flows_in_a_balloon_only_when_it_is_the_balloons_only_text() {
        for with_dialogue in [false, true] {
            let mut session = Session::memory().await.unwrap();
            let detected = |label: &str| KoharuLayoutDetection {
                label_id: 0,
                label: label.to_owned(),
                score: 1.0,
                bbox: [0.0, 0.0, 1.0, 1.0],
                area: 1,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    pixels: vec![u8::MAX],
                },
            };
            let bubble_detection = detected("bubble");
            let sound_effect_detection = detected("onomatopoeia");
            let dialogue_detection = detected("text");
            let generation = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
            let mut ids = None;
            let patch = session
                .snapshot()
                .patch(|edit| {
                    let page = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                    let RegionOutput::Bubble(bubble) =
                        write_region(edit, page, &bubble_detection, None, &generation).unwrap()
                    else {
                        panic!("expected a bubble region");
                    };
                    let RegionOutput::Text(text) =
                        write_region(edit, page, &sound_effect_detection, None, &generation)
                            .unwrap()
                    else {
                        panic!("expected an OCR text region");
                    };
                    ids = Some((text.entity, text.content, text.layer, bubble.entity));
                    let mut texts = vec![text];
                    if with_dialogue {
                        let RegionOutput::Text(dialogue) =
                            write_region(edit, page, &dialogue_detection, None, &generation)
                                .unwrap()
                        else {
                            panic!("expected an OCR text region");
                        };
                        texts.push(dialogue);
                    }
                    link_dialogue_regions(
                        edit,
                        &PageRegions {
                            bubbles: vec![bubble],
                            texts,
                        },
                        &generation,
                    )
                    .unwrap();
                    Ok(())
                })
                .unwrap();
            let snapshot = session.commit(patch).await.unwrap().snapshot;
            let (region, content, layer, bubble) = ids.unwrap();

            assert_eq!(
                snapshot
                    .component::<TextRole>(content)
                    .unwrap()
                    .unwrap()
                    .role,
                ONOMATOPOEIA_ROLE
            );
            let flows_in = snapshot
                .relation_from::<FlowsIn>(layer)
                .unwrap()
                .map(|relation| relation.value().target);
            let fits_to = snapshot
                .relation_from::<FitsTo>(layer)
                .unwrap()
                .map(|relation| relation.value().target);
            if with_dialogue {
                assert_eq!((flows_in, fits_to), (None, Some(region)));
            } else {
                assert_eq!((flows_in, fits_to), (Some(bubble), None));
            }
        }
    }

    fn detection(label: &str, score: f32, bbox: [f32; 4]) -> KoharuLayoutDetection {
        KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score,
            bbox,
            area: 0,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                pixels: vec![0],
            },
        }
    }

    #[test]
    fn bubble_geometry_follows_the_instance_mask_polygon() {
        let width = 9;
        let height = 9;
        let mut pixels = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                if x.abs_diff(4) + y.abs_diff(4) <= 4 {
                    pixels[y * width + x] = u8::MAX;
                }
            }
        }

        let geometry = mask_geometry(&KoharuLayoutMask {
            x: 10,
            y: 20,
            width: width as u32,
            height: height as u32,
            pixels,
        })
        .unwrap();

        assert!(geometry.points.len() >= 4);
        assert!(
            geometry
                .points
                .iter()
                .all(|point| !(point.x == 10.0 && point.y == 20.0))
        );
        assert!(
            geometry
                .points
                .iter()
                .any(|point| point.x == 14.0 && point.y == 20.0)
        );
    }

    fn masked_text(
        local_width: f64,
        local_height: f64,
        angle_degrees: f64,
        color: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let center_x = f64::from(width) * 0.5;
        let center_y = f64::from(height) * 0.5;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([200, 180, 160]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - center_x;
                let dy = f64::from(y) + 0.5 - center_y;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                if local_x.abs() <= local_width * 0.5 && local_y.abs() <= local_height * 0.5 {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb(color));
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [0.0, 0.0, width as f32, height as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    fn outlined_text(
        stroke_width: u32,
        fill: [u8; 3],
        stroke: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        outlined_text_on_background(stroke_width, fill, stroke, [16, 24, 88])
    }

    fn outlined_text_on_background(
        stroke_width: u32,
        fill: [u8; 3],
        stroke: [u8; 3],
        background: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 38, 76, 58];
        let mut image = RgbImage::from_pixel(width, height, Rgb(background));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                let is_fill = x >= left + stroke_width
                    && x < right - stroke_width
                    && y >= top + stroke_width
                    && y < bottom - stroke_width;
                image.put_pixel(x, y, if is_fill { Rgb(fill) } else { Rgb(stroke) });
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    fn region_masked_text(fill: [u8; 3], background: [u8; 3]) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 20, 76, 76];
        let mut image = RgbImage::from_pixel(width, height, Rgb(background));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        for x in [27, 35, 43, 51, 59, 67] {
            for y in 28..68 {
                for ink_x in x..x + 3 {
                    image.put_pixel(ink_x, y, Rgb(fill));
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    fn outlined_text_on_textured_background() -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 20, 76, 76];
        let mut image = RgbImage::from_pixel(width, height, Rgb([24, 32, 96]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                let background = match (x / 7 + y / 5) % 3 {
                    0 => [0, 0, 0],
                    1 => [38, 48, 128],
                    _ => [24, 32, 96],
                };
                image.put_pixel(x, y, Rgb(background));
            }
        }
        for glyph_left in [26, 38, 50, 62] {
            for y in 28..68 {
                for x in glyph_left..glyph_left + 9 {
                    let is_fill =
                        x >= glyph_left + 3 && x < glyph_left + 6 && (31..65).contains(&y);
                    image.put_pixel(
                        x,
                        y,
                        if is_fill {
                            Rgb([0, 0, 0])
                        } else {
                            Rgb([255, 255, 255])
                        },
                    );
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                },
            },
        )
    }

    #[tokio::test]
    async fn detected_typography_preserves_the_measured_outline() {
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                page = Some(edit.add_page(PageDraft::new("page", 96.0, 96.0), At::End)?);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(create).await.unwrap().snapshot;
        let page = page.unwrap();
        let (image, detection) = outlined_text(3, [0, 0, 0], [255, 255, 255]);
        let inferred = infer_typography(&image, &detection);
        let generation = generation(super::PRODUCER, super::OUTPUT_MODEL_ID).unwrap();
        let mut layer = None;
        let patch = snapshot
            .patch(|edit| {
                let output = write_region(edit, page, &detection, inferred, &generation).unwrap();
                let RegionOutput::Text(text) = output else {
                    panic!("expected a text region");
                };
                layer = Some(text.layer);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let typography = snapshot
            .component::<Typography>(layer.unwrap())
            .unwrap()
            .unwrap();

        assert_eq!(typography.color, Some([0, 0, 0, 255]));
        assert_eq!(typography.stroke_color, Some([255, 255, 255, 255]));
        assert_eq!(typography.stroke_width, Some(3.0));
    }

    #[test]
    fn nms_removes_lower_scored_overlapping_regions_per_class() {
        let mut detections = vec![
            detection("text", 0.8, [5.0, 5.0, 105.0, 105.0]),
            detection("bubble", 0.7, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.9, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.5, [20.0, 20.0, 80.0, 80.0]),
            detection("text", 0.6, [200.0, 0.0, 250.0, 50.0]),
        ];

        non_maximum_suppression(&mut detections, 0.5);

        let text_scores = detections
            .iter()
            .filter(|detection| detection.label == "text")
            .map(|detection| detection.score)
            .collect::<Vec<_>>();
        assert_eq!(text_scores, [0.9, 0.6, 0.5]);
        assert!(
            detections
                .iter()
                .any(|detection| detection.label == "bubble")
        );
    }

    #[test]
    fn nms_merges_exact_text_and_onomatopoeia_duplicates() {
        let mut detections = vec![
            detection("text", 0.7, [10.0, 20.0, 40.0, 60.0]),
            detection("onomatopoeia", 0.9, [10.0, 20.0, 40.0, 60.0]),
            detection("bubble", 0.8, [10.0, 20.0, 40.0, 60.0]),
        ];

        non_maximum_suppression(&mut detections, 0.5);

        assert_eq!(detections.len(), 2);
        assert!(
            detections
                .iter()
                .any(|detection| detection.label == "onomatopoeia")
        );
        assert!(
            detections
                .iter()
                .any(|detection| detection.label == "bubble")
        );
    }

    #[test]
    fn nms_preserves_nested_cross_class_text_regions() {
        let mut detections = vec![
            detection("text", 0.9, [0.0, 0.0, 100.0, 100.0]),
            detection("onomatopoeia", 0.8, [10.0, 10.0, 90.0, 90.0]),
        ];

        non_maximum_suppression(&mut detections, 0.5);

        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn boxes_are_clipped_to_the_page_and_outside_boxes_dropped() {
        let mut detections = vec![
            detection("onomatopoeia", 0.4, [-40.0, -238.0, 1325.0, 816.0]),
            detection("text", 0.9, [1300.0, 10.0, 1320.0, 40.0]),
        ];

        clamp_to_image(
            &mut detections,
            ImageSize {
                width: 1280,
                height: 1280,
            },
        );

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].bbox, [0.0, 0.0, 1280.0, 816.0]);
    }

    #[test]
    fn spurious_onomatopoeia_boxes_are_removed() {
        // Raw boxes from real 1280-pixel-wide pages: a 0.25 effect over a line
        // of dialogue and two better-scored effects, a 0.38 effect over 42% of
        // a page and a line of dialogue, a 0.81 effect that overlaps one line,
        // a 0.37 effect around two weaker fragments, and a 0.23 title over 25%
        // of a short page that covers nothing else. The last three must stay.
        let page = ImageSize {
            width: 1280,
            height: 1280,
        };
        let mut detections = vec![
            detection("onomatopoeia", 0.25, [626.0, 380.0, 849.0, 608.0]),
            detection("text", 0.9, [828.0, 366.0, 859.0, 426.0]),
            detection("onomatopoeia", 0.48, [706.0, 416.0, 727.0, 460.0]),
            detection("onomatopoeia", 0.33, [770.0, 490.0, 786.0, 541.0]),
            detection("onomatopoeia", 0.38, [18.0, 12.0, 1267.0, 566.0]),
            detection("text", 0.9, [100.0, 100.0, 140.0, 300.0]),
            detection("onomatopoeia", 0.81, [433.0, 735.0, 712.0, 967.0]),
            detection("text", 0.9, [624.0, 709.0, 659.0, 792.0]),
            detection("onomatopoeia", 0.37, [1121.0, 831.0, 1154.0, 889.0]),
            detection("onomatopoeia", 0.2, [1125.0, 835.0, 1150.0, 850.0]),
            detection("onomatopoeia", 0.2, [1125.0, 860.0, 1150.0, 880.0]),
        ];

        remove_spurious_onomatopoeia(&mut detections, page);

        let scores = detections
            .iter()
            .filter(|detection| detection.label == "onomatopoeia")
            .map(|detection| detection.score)
            .collect::<Vec<_>>();
        assert_eq!(scores, [0.48, 0.33, 0.81, 0.37, 0.2, 0.2]);

        let mut title = vec![detection(
            "onomatopoeia",
            0.23,
            [203.0, 123.0, 1250.0, 300.0],
        )];
        remove_spurious_onomatopoeia(
            &mut title,
            ImageSize {
                width: 1280,
                height: 568,
            },
        );
        assert_eq!(title.len(), 1);
    }

    #[test]
    fn page_numbers_at_the_side_edge_are_removed() {
        // Page numbers from a real 1280-pixel chapter (already clipped to the
        // page), then a short line of text at the edge mid-page, a wider
        // caption in the bottom margin, and an effect where a number would be.
        let page = ImageSize {
            width: 1280,
            height: 1280,
        };
        let mut detections = vec![
            detection("text", 0.45, [0.0, 1115.0, 10.0, 1147.0]),
            detection("text", 0.61, [1268.0, 1114.0, 1280.0, 1146.0]),
            detection("text", 0.25, [1269.0, 1115.0, 1280.0, 1146.0]),
            detection("text", 0.9, [2.0, 600.0, 18.0, 640.0]),
            detection("text", 0.9, [1180.0, 1110.0, 1278.0, 1150.0]),
            detection("onomatopoeia", 0.4, [1268.0, 1114.0, 1280.0, 1146.0]),
        ];

        remove_page_numbers(&mut detections, page);

        assert_eq!(
            detections
                .iter()
                .map(|detection| detection.bbox[1])
                .collect::<Vec<_>>(),
            [600.0, 1110.0, 1114.0]
        );
    }

    #[test]
    fn nms_preserves_nested_same_class_text_regions() {
        let mut detections = vec![
            detection("text", 0.9, [0.0, 0.0, 100.0, 100.0]),
            detection("text", 0.8, [10.0, 10.0, 90.0, 90.0]),
        ];

        non_maximum_suppression(&mut detections, 0.5);

        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn nested_text_detection_is_removed_from_its_parent_region() {
        let mut parent = detection("text", 0.8, [0.0, 0.0, 6.0, 6.0]);
        parent.area = 36;
        parent.mask = KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 6,
            height: 6,
            pixels: vec![u8::MAX; 36],
        };
        let mut child = detection("onomatopoeia", 0.9, [2.0, 2.0, 4.0, 4.0]);
        child.area = 4;
        child.mask = KoharuLayoutMask {
            x: 2,
            y: 2,
            width: 2,
            height: 2,
            pixels: vec![u8::MAX; 4],
        };
        let mut detections = vec![parent, child];

        remove_nested_text_detections(&mut detections);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "text");
        assert_eq!(detections[0].area, 36);
        assert!(detections[0].mask.pixels.iter().all(|pixel| *pixel != 0));
    }

    #[test]
    fn text_detection_contained_by_eighty_percent_is_removed() {
        let mut parent = detection("text", 0.8, [0.0, 0.0, 10.0, 10.0]);
        parent.area = 100;
        parent.mask = KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
            pixels: vec![u8::MAX; 100],
        };
        let mut child = detection("onomatopoeia", 0.9, [6.0, 2.0, 11.0, 7.0]);
        child.area = 25;
        child.mask = KoharuLayoutMask {
            x: 6,
            y: 2,
            width: 5,
            height: 5,
            pixels: vec![u8::MAX; 25],
        };
        let mut detections = vec![parent, child];

        remove_nested_text_detections(&mut detections);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "text");
        assert_eq!(detections[0].area, 100);
    }

    #[test]
    fn overlapping_masks_are_not_nested_from_bounding_boxes_alone() {
        let mut parent = detection("text", 0.8, [0.0, 0.0, 6.0, 6.0]);
        let mut parent_pixels = vec![0; 36];
        for y in 0..6 {
            parent_pixels[y * 6] = u8::MAX;
        }
        parent_pixels[7] = u8::MAX;
        parent.area = 7;
        parent.mask = KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 6,
            height: 6,
            pixels: parent_pixels,
        };
        let mut child = detection("onomatopoeia", 0.9, [1.0, 1.0, 5.0, 5.0]);
        child.area = 16;
        child.mask = KoharuLayoutMask {
            x: 1,
            y: 1,
            width: 4,
            height: 4,
            pixels: vec![u8::MAX; 16],
        };
        let mut detections = vec![parent, child];

        remove_nested_text_detections(&mut detections);

        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].area, 7);
        assert!(detections[0].mask.contains(1, 1));
    }

    #[test]
    fn nested_bubbles_do_not_remove_text_detections() {
        let mut text = detection("text", 0.8, [0.0, 0.0, 4.0, 4.0]);
        text.area = 16;
        text.mask = KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
            pixels: vec![u8::MAX; 16],
        };
        let mut bubble = detection("bubble", 0.9, [1.0, 1.0, 3.0, 3.0]);
        bubble.area = 4;
        bubble.mask = KoharuLayoutMask {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
            pixels: vec![u8::MAX; 4],
        };
        let mut detections = vec![text, bubble];

        remove_nested_text_detections(&mut detections);

        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].area, 16);
        assert!(detections[0].mask.pixels.iter().all(|pixel| *pixel != 0));
    }

    #[test]
    fn almost_contained_real_page_text_removes_the_inner_detection() {
        let mut parent = detection("text", 0.8, [394.544, 925.362, 955.329, 1119.854]);
        parent.area = 1;
        parent.mask = KoharuLayoutMask {
            x: 400,
            y: 930,
            width: 1,
            height: 1,
            pixels: vec![u8::MAX],
        };
        let mut child = detection("onomatopoeia", 0.9, [384.731, 923.990, 593.175, 1119.575]);
        child.area = 1;
        child.mask = parent.mask.clone();
        let mut detections = vec![parent, child];

        remove_nested_text_detections(&mut detections);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "text");
    }

    #[test]
    fn text_mask_includes_onomatopoeia() {
        let detection = |label: &str, x: u32, value: u8| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 1.0,
            bbox: [x as f32, 0.0, x as f32 + 1.0, 1.0],
            area: u32::from(value != 0),
            mask: KoharuLayoutMask {
                x,
                y: 0,
                width: 1,
                height: 1,
                pixels: vec![value],
            },
        };
        let detections = vec![
            detection("bubble", 0, 0),
            detection("onomatopoeia", 0, 255),
            detection("text", 1, 255),
            detection("onomatopoeia", 3, 255),
        ];

        let mask = mask_for(
            &detections,
            "text",
            ImageSize {
                width: 4,
                height: 1,
            },
        );

        assert_eq!(mask.as_raw(), &[255, 255, 0, 255]);
    }

    #[test]
    fn bubble_membership_uses_detected_ink_instead_of_text_box_corners() {
        let mask = |pixels| KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
            pixels,
        };
        let bubble = mask(vec![
            0, 0, 1, 0, 0, //
            0, 1, 1, 1, 0, //
            1, 1, 1, 1, 1, //
            0, 1, 1, 1, 0, //
            0, 0, 1, 0, 0,
        ]);
        let dialogue = mask(vec![
            0, 0, 0, 0, 0, //
            0, 0, 1, 0, 0, //
            0, 1, 0, 1, 0, //
            0, 0, 1, 0, 0, //
            0, 0, 0, 0, 0,
        ]);
        let crossing = mask(vec![
            0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, //
            0, 0, 0, 1, 1, //
            0, 0, 0, 1, 1, //
            0, 0, 0, 1, 1,
        ]);

        let bounds = [0.0, 0.0, 5.0, 5.0];
        assert_eq!(mask_containment(&bubble, &dialogue, bounds), 1.0);
        assert!(mask_containment(&bubble, &crossing, bounds) < DIALOGUE_MASK_CONTAINMENT_THRESHOLD);
    }

    #[test]
    fn mask_containment_accepts_independent_local_extents() {
        let valid = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 2,
            height: 2,
            pixels: vec![1; 4],
        };
        let malformed = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 2,
            height: 2,
            pixels: vec![1; 3],
        };
        let different_size = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 1,
            height: 1,
            pixels: vec![1],
        };

        let bounds = [5.0, 5.0, 7.0, 7.0];
        assert_eq!(mask_containment(&valid, &malformed, bounds), 0.0);
        assert_eq!(mask_containment(&valid, &different_size, bounds), 1.0);
    }

    #[test]
    fn local_mask_morphology_matches_the_full_page_operation() {
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let local = |x, y, width, height, pixels: Vec<u8>| KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [x as f32, y as f32, (x + width) as f32, (y + height) as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                x,
                y,
                width,
                height,
                pixels,
            },
        };
        let detections = vec![
            local(0, 0, 4, 4, vec![255; 16]),
            local(10, 8, 4, 4, vec![255; 16]),
            local(19, 10, 3, 5, vec![255; 15]),
            local(52, 40, 4, 3, vec![255; 12]),
        ];
        let radius = 3;
        let page = mask_for(&detections, "text", size);
        let expected = close(&dilate(&page, Norm::L2, radius), Norm::L2, radius);

        assert_eq!(closed_mask_for(&detections, "text", size, radius), expected);
    }

    #[test]
    fn local_mask_morphology_matches_varied_sparse_instances() {
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let mut state = 0x7f4a_7c15_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for case in 0..24 {
            let mut detections = Vec::new();
            for _ in 0..8 {
                let width = next() % 8 + 1;
                let height = next() % 8 + 1;
                let x = next() % (size.width - width + 1);
                let y = next() % (size.height - height + 1);
                let pixels = (0..width * height)
                    .map(|_| if next() % 3 == 0 { 0 } else { u8::MAX })
                    .collect::<Vec<_>>();
                detections.push(KoharuLayoutDetection {
                    label_id: 0,
                    label: "text".to_owned(),
                    score: 1.0,
                    bbox: [x as f32, y as f32, (x + width) as f32, (y + height) as f32],
                    area: pixels.iter().filter(|pixel| **pixel != 0).count() as u32,
                    mask: KoharuLayoutMask {
                        x,
                        y,
                        width,
                        height,
                        pixels,
                    },
                });
            }
            let radius = (case % 5 + 1) as u8;
            let page = mask_for(&detections, "text", size);
            let expected = close(&dilate(&page, Norm::L2, radius), Norm::L2, radius);
            assert_eq!(
                closed_mask_for(&detections, "text", size, radius),
                expected,
                "case {case}, radius {radius}"
            );
        }
    }

    #[test]
    fn layout_order_follows_ltr_panels_then_bubbles_then_their_text() {
        let detections = vec![
            detection("text", 0.63, [20.0, 30.0, 70.0, 70.0]),
            detection("bubble", 0.7, [10.0, 20.0, 80.0, 80.0]),
            detection("panel", 0.9, [100.0, 0.0, 200.0, 200.0]),
            detection("text", 0.62, [130.0, 110.0, 180.0, 150.0]),
            detection("bubble", 0.8, [120.0, 20.0, 190.0, 80.0]),
            detection("panel", 0.9, [0.0, 0.0, 95.0, 200.0]),
            detection("text", 0.61, [130.0, 30.0, 180.0, 70.0]),
            detection("bubble", 0.7, [120.0, 100.0, 190.0, 160.0]),
        ];

        let text_scores = layout_order(&detections, Language::English)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.63, 0.61, 0.62]);
    }

    #[test]
    fn layout_order_reads_japanese_columns_right_to_left_then_top_to_bottom() {
        let detections = vec![
            detection("text", 0.4, [20.0, 100.0, 70.0, 140.0]),
            detection("text", 0.2, [120.0, 120.0, 170.0, 160.0]),
            detection("text", 0.3, [20.0, 0.0, 70.0, 40.0]),
            detection("text", 0.1, [120.0, 20.0, 170.0, 60.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .map(|index| detections[index].score)
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn layout_order_merges_staggered_overlapping_fragments_in_one_cjk_column() {
        let detections = vec![
            detection("text", 0.1, [20.0, 0.0, 120.0, 40.0]),
            detection("text", 0.2, [100.0, 100.0, 200.0, 140.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .map(|index| detections[index].score)
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2]);
    }

    #[test]
    fn layout_order_preserves_panel_hierarchy_for_cjk_columns() {
        let detections = vec![
            detection("text", 0.63, [20.0, 30.0, 70.0, 70.0]),
            detection("bubble", 0.7, [10.0, 20.0, 80.0, 80.0]),
            detection("panel", 0.9, [100.0, 0.0, 200.0, 200.0]),
            detection("text", 0.62, [130.0, 110.0, 180.0, 150.0]),
            detection("bubble", 0.8, [120.0, 20.0, 190.0, 80.0]),
            detection("panel", 0.9, [0.0, 0.0, 95.0, 200.0]),
            detection("text", 0.61, [130.0, 30.0, 180.0, 70.0]),
            detection("bubble", 0.7, [120.0, 100.0, 190.0, 160.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.61, 0.62, 0.63]);
    }

    #[test]
    fn layout_order_reads_top_spanning_panel_before_cjk_columns_below() {
        let detections = vec![
            detection("panel", 0.9, [0.0, 0.0, 200.0, 80.0]),
            detection("text", 0.1, [80.0, 20.0, 120.0, 60.0]),
            detection("panel", 0.9, [0.0, 100.0, 90.0, 200.0]),
            detection("text", 0.3, [20.0, 120.0, 70.0, 160.0]),
            detection("panel", 0.9, [110.0, 100.0, 200.0, 200.0]),
            detection("text", 0.2, [130.0, 120.0, 180.0, 160.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn uncontained_sfx_does_not_change_japanese_t_panel_order() {
        let detections = vec![
            detection("panel", 0.9, [0.0, 0.0, 200.0, 80.0]),
            detection("text", 0.1, [80.0, 20.0, 120.0, 60.0]),
            detection("panel", 0.9, [0.0, 100.0, 90.0, 200.0]),
            detection("text", 0.3, [20.0, 120.0, 70.0, 160.0]),
            detection("panel", 0.9, [110.0, 100.0, 200.0, 200.0]),
            detection("text", 0.2, [130.0, 120.0, 180.0, 160.0]),
            detection("onomatopoeia", 0.4, [210.0, 20.0, 240.0, 60.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn uncontained_sfx_preserves_real_right_left_then_spanning_panel_order() {
        let detections = vec![
            detection("panel", 0.9, [103.53, 0.86, 680.85, 757.97]),
            detection("text", 0.2, [537.63, 430.70, 587.76, 575.53]),
            detection("panel", 0.9, [693.52, 1.93, 1280.82, 757.30]),
            detection("text", 0.1, [1106.99, 70.47, 1189.22, 218.39]),
            detection("panel", 0.9, [4.24, 752.93, 1278.80, 1211.98]),
            detection("text", 0.3, [560.91, 875.24, 613.39, 945.51]),
            detection("onomatopoeia", 0.4, [1300.0, 800.0, 1360.0, 900.0]),
        ];

        let text_scores = layout_order(&detections, Language::Japanese)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn best_container_prefers_containment_before_area() {
        let items = [
            LayoutItem {
                kind: LayoutKind::Panel,
                bounds: [40.0, 0.0, 75.0, 100.0],
            },
            LayoutItem {
                kind: LayoutKind::Panel,
                bounds: [0.0, 0.0, 120.0, 100.0],
            },
            LayoutItem {
                kind: LayoutKind::Text,
                bounds: [40.0, 0.0, 100.0, 100.0],
            },
        ];

        assert_eq!(best_container(&items, 2, &[0, 1]), Some(1));
    }

    #[test]
    fn layout_order_reads_western_rows_top_to_bottom() {
        let detections = vec![
            detection("text", 0.4, [20.0, 100.0, 70.0, 140.0]),
            detection("text", 0.2, [120.0, 120.0, 170.0, 160.0]),
            detection("text", 0.3, [20.0, 0.0, 70.0, 40.0]),
            detection("text", 0.1, [120.0, 20.0, 170.0, 60.0]),
        ];

        let text_scores = layout_order(&detections, Language::English)
            .into_iter()
            .map(|index| detections[index].score)
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.3, 0.1, 0.4, 0.2]);
    }

    #[test]
    fn layout_order_merges_containers_and_text_at_each_spatial_level() {
        let detections = vec![
            detection("panel", 0.9, [0.0, 40.0, 200.0, 240.0]),
            detection("bubble", 0.8, [100.0, 120.0, 190.0, 210.0]),
            detection("text", 0.3, [120.0, 140.0, 180.0, 190.0]),
            detection("text", 0.2, [20.0, 60.0, 180.0, 90.0]),
            detection("text", 0.1, [20.0, 0.0, 180.0, 20.0]),
        ];

        let text_scores = layout_order(&detections, Language::English)
            .into_iter()
            .filter_map(|index| {
                (detections[index].label == "text").then_some(detections[index].score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn antialiased_edges_of_light_text_are_not_an_outline() {
        // White strokes on a dark balloon: a grey blend rings the dark core of
        // the gap between strokes, but it is anti-aliasing, not an outline.
        let width = 120;
        let height = 120;
        let [left, top, right, bottom] = [36, 30, 84, 90];
        let mut image = RgbImage::from_pixel(width, height, Rgb([24, 32, 32]));
        for y in 33..87 {
            for x in 39..81 {
                image.put_pixel(x, y, Rgb([170, 177, 177]));
            }
        }
        for y in 34..86 {
            for x in (40..60).chain(64..80) {
                image.put_pixel(x, y, Rgb([250, 252, 252]));
            }
        }
        for y in 36..84 {
            for x in 61..63 {
                image.put_pixel(x, y, Rgb([6, 12, 12]));
            }
        }
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [255, 255, 255]);
        assert_eq!(inferred.stroke_color, None);
    }

    fn text_detection(bbox: [u32; 4]) -> KoharuLayoutDetection {
        let [left, top, right, bottom] = bbox;
        KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            area: (right - left) * (bottom - top),
            mask: KoharuLayoutMask {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top,
                pixels: vec![1; ((right - left) * (bottom - top)) as usize],
            },
        }
    }

    #[test]
    fn dark_balloon_with_light_text_becomes_a_bubble() {
        // A black ellipse with a white rim on white paper, white text inside.
        let mut image = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
        for y in 0..200_u32 {
            for x in 0..200_u32 {
                let dx = (x as f32 - 100.0) / 60.0;
                let dy = (y as f32 - 100.0) / 40.0;
                if dx * dx + dy * dy <= 1.0 {
                    image.put_pixel(x, y, Rgb([20, 20, 20]));
                }
            }
        }
        for x in (80..120).step_by(8) {
            for y in 90..110 {
                image.put_pixel(x, y, Rgb([250, 250, 250]));
            }
        }
        let balloons = dark_balloons(&image, &[text_detection([78, 88, 122, 112])]);
        assert_eq!(balloons.len(), 1);
        let [left, top, right, bottom] = balloons[0].bbox;
        assert!((left - 40.0).abs() <= 2.0 && (right - 160.0).abs() <= 2.0);
        assert!((top - 60.0).abs() <= 2.0 && (bottom - 140.0).abs() <= 2.0);
        assert!(balloons[0].mask.contains(100, 70));
        assert!(!balloons[0].mask.contains(45, 65));
    }

    #[test]
    fn dark_background_and_light_paper_are_not_balloons() {
        let night = RgbImage::from_pixel(200, 200, Rgb([10, 10, 10]));
        assert!(dark_balloons(&night, &[text_detection([80, 90, 120, 110])]).is_empty());
        let paper = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
        assert!(dark_balloons(&paper, &[text_detection([80, 90, 120, 110])]).is_empty());
    }

    #[test]
    fn dark_text_with_white_outline_on_halftone_keeps_the_outline() {
        // Segmentation boxes the glyphs, so the white outline extends past the
        // box into the screentone, as on real pages.
        let width = 120;
        let height = 120;
        let [left, top, right, bottom] = [30, 30, 90, 90];
        let outline = 6;
        let mut image = RgbImage::from_fn(width, height, |x, y| {
            if x % 4 < 2 && y % 4 < 2 {
                Rgb([0, 0, 0])
            } else {
                Rgb([150, 150, 150])
            }
        });
        for y in top - outline..bottom + outline {
            for x in left - outline..right + outline {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        for bar in [left, left + 18, left + 36, left + 54] {
            for y in top..bottom {
                for x in bar..bar + 6 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [left as f32, top as f32, right as f32, bottom as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
    }

    #[test]
    fn typography_comes_from_horizontal_text_mask() {
        let (image, detection) = masked_text(52.0, 12.0, 12.0, [24, 80, 160]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 12.0).abs() < 1.0);
        assert_eq!(inferred.color, [24, 80, 160]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
    }

    #[test]
    fn vertical_text_angle_is_relative_to_upright_vertical() {
        let (image, detection) = masked_text(12.0, 52.0, 9.0, [120, 80, 40]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 9.0).abs() < 1.0);
        assert_eq!(inferred.writing_mode, WritingMode::Vertical);
    }

    #[test]
    fn tall_multiline_mask_uses_text_lines_instead_of_block_aspect() {
        let width = 96;
        let height = 96;
        let angle_degrees = 8.0_f64;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([240, 240, 240]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - f64::from(width) * 0.5;
                let dy = f64::from(y) + 0.5 - f64::from(height) * 0.5;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                let inside = [-24.0, 0.0, 24.0]
                    .into_iter()
                    .any(|line_y| local_x.abs() <= 15.0 && (local_y - line_y).abs() <= 2.0);
                if inside {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb([8, 8, 8]));
                }
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            },
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
        assert!((inferred.angle_degrees - 8.0).abs() < 1.0);

        // The region hugs the rotated lines instead of rotating the full box,
        // which would push its corners off the image.
        let geometry =
            super::rotated_rectangle_geometry(inferred.oriented_extent, inferred.angle_degrees);
        for point in &geometry.points {
            assert!((0.0..=f64::from(width)).contains(&point.x), "{point:?}");
            assert!((0.0..=f64::from(height)).contains(&point.y), "{point:?}");
        }
        let [min_u, min_v, max_u, max_v] = inferred.oriented_extent;
        assert!((max_u - min_u - 31.0).abs() < 3.0);
        assert!((max_v - min_v - 53.0).abs() < 3.0);
    }

    #[test]
    fn antialiased_dark_text_uses_the_high_contrast_core() {
        let (mut image, detection) = masked_text(52.0, 12.0, 0.0, [96, 94, 92]);
        for (index, &mask) in detection.mask.pixels.iter().enumerate() {
            if mask != 0 && index.is_multiple_of(3) {
                let x = index as u32 % image.width();
                let y = index as u32 / image.width();
                image.put_pixel(x, y, Rgb([8, 9, 7]));
            }
        }

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
    }

    #[test]
    fn text_colors_snap_only_at_luminance_extremes() {
        assert_eq!(normalize_text_color([20, 31, 24]), [0, 0, 0]);
        assert_eq!(normalize_text_color([230, 240, 250]), [255, 255, 255]);
        assert_eq!(normalize_text_color([255, 0, 0]), [0, 0, 0]);
        assert_eq!(normalize_text_color([0, 255, 0]), [0, 255, 0]);
        assert_eq!(normalize_text_color([20, 115, 235]), [20, 115, 235]);
        assert_eq!(normalize_text_color([40, 70, 40]), [0, 0, 0]);
    }

    #[test]
    fn outlined_text_uses_the_deep_fill_and_measures_the_border() {
        for stroke_width in [2, 3, 5] {
            let (image, detection) = outlined_text(stroke_width, [0, 0, 0], [255, 255, 255]);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, [0, 0, 0]);
            assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
            assert_eq!(inferred.stroke_width, Some(stroke_width as f32));
        }
    }

    #[test]
    fn outlined_text_roles_do_not_flip_with_inverse_luminance() {
        let (image, detection) = outlined_text(3, [255, 255, 255], [0, 0, 0]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [255, 255, 255]);
        assert_eq!(inferred.stroke_color, Some([0, 0, 0]));
        assert_eq!(inferred.stroke_width, Some(3.0));
    }

    #[test]
    fn glyph_holes_matching_the_background_do_not_invert_the_paint() {
        let (image, detection) =
            outlined_text_on_background(3, [255, 255, 255], [0, 0, 0], [255, 255, 255]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }

    #[test]
    fn outlined_text_is_separated_from_matching_dark_artwork() {
        let (image, detection) = outlined_text_on_textured_background();

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
        assert_eq!(inferred.stroke_width, Some(3.0));
    }

    #[test]
    fn colors_in_the_same_luminance_class_do_not_create_a_border() {
        let (image, detection) = outlined_text(3, [0, 0, 0], [48, 48, 48]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }

    #[test]
    fn wide_antialias_bands_do_not_create_borders() {
        for (fill, antialias, background) in [
            ([20, 115, 235], [133, 180, 240], [245, 245, 245]),
            ([0, 0, 0], [96, 96, 96], [245, 245, 245]),
        ] {
            let (image, detection) = outlined_text_on_background(4, fill, antialias, background);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, fill);
            assert_eq!(inferred.stroke_color, None);
            assert_eq!(inferred.stroke_width, None);
        }
    }

    #[test]
    fn text_region_background_is_not_mistaken_for_the_fill() {
        for (fill, background, expected) in [
            ([24, 40, 80], [225, 130, 175], [0, 0, 0]),
            ([220, 240, 255], [16, 24, 88], [255, 255, 255]),
            ([20, 115, 235], [245, 245, 245], [20, 115, 235]),
        ] {
            let (image, detection) = region_masked_text(fill, background);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, expected);
            assert_eq!(inferred.stroke_color, None);
            assert_eq!(inferred.stroke_width, None);
        }
    }

    #[test]
    fn color_palette_is_bounded_independently_of_crop_area() {
        let pixels = (0..65_536_u32)
            .map(|index| MaskPixel {
                x: 0,
                y: 0,
                color: [
                    index as u8,
                    (index >> 8) as u8,
                    index.wrapping_mul(31) as u8,
                ],
                inside_mask: true,
            })
            .collect::<Vec<_>>();

        assert!(color_palette(&pixels).len() <= 16 * 16 * 16);
    }
}
