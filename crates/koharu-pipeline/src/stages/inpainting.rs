use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageBuffer, ImageFormat, Luma, Rgb, RgbImage,
    Rgba, RgbaImage,
};
use imageproc::region_labelling::{Connectivity, connected_components};
use koharu_ml::{
    aot_inpainting::AotInpainting,
    flux2_klein::{Flux2KleinInpaint, Flux2KleinInpaintOptions},
    lama::{InpaintRequest, LaMa},
    rorem_mixed::{DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, RoremMixed, RoremMixedOptions},
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityId, Geometry, Origin,
    RasterLayer, RasterLayerKind, RecognizedFrom, Region, RegionSpec, TextRegion, TextRole,
    Visibility,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{
    StageInput, StageProcessor,
    fal::{DEFAULT_CLEANUP_PROMPT, Fal, FalModel},
    finish, generation,
    openrouter_image::OpenRouterImage,
    replicate::{self, Replicate},
};
use crate::{
    ImageEditConfig, InpaintingApplyMode, InpaintingIntent, ModelCell, config::InpaintingModel,
};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

mod screentone;

#[cfg(test)]
mod preview_tests;
#[cfg(test)]
mod removal_tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2KleinConfig {
    pub prompt: String,
}

impl Default for Flux2KleinConfig {
    fn default() -> Self {
        Self {
            prompt: "Remove the text and reconstruct the background.".to_owned(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct RoremMixedConfig {
    pub prompt: String,
    pub negative_prompt: String,
}

impl Default for RoremMixedConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_owned(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_owned(),
        }
    }
}

pub(super) struct Processor {
    config: InpaintingModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(config: InpaintingModel, device: koharu_ml::Device) -> Result<Self> {
        match &config {
            InpaintingModel::LaMa {}
            | InpaintingModel::AotInpainting {}
            | InpaintingModel::BalloonFill {} => {}
            InpaintingModel::Flux2Klein(settings) => {
                ensure!(
                    !settings.prompt.contains('\0'),
                    "FLUX.2 prompt contains NUL"
                );
            }
            InpaintingModel::RoremMixed(settings) => {
                ensure!(
                    !settings.prompt.contains('\0') && !settings.negative_prompt.contains('\0'),
                    "RORem prompt contains NUL"
                );
            }
            InpaintingModel::FalImageEdit {
                config: settings, ..
            } => {
                ensure!(
                    !settings.prompt.contains('\0'),
                    "Fal.ai prompt contains NUL"
                );
            }
            InpaintingModel::OpenRouterImageEdit { config, .. } => {
                ensure!(
                    !config.prompt.contains('\0'),
                    "OpenRouter prompt contains NUL"
                );
            }
            InpaintingModel::ReplicateImageEdit { config, .. } => {
                ensure!(
                    !config.prompt.contains('\0'),
                    "Replicate prompt contains NUL"
                );
            }
        }

        Ok(Self {
            config,
            device,
            model: ModelCell::new(),
        })
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, _input: &StageInput) -> String {
        match &self.config {
            InpaintingModel::LaMa {} => "lama",
            InpaintingModel::AotInpainting {} => "aot-inpainting",
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
            InpaintingModel::RoremMixed(_) => "rorem-mixed",
            InpaintingModel::BalloonFill {} => BALLOON_FILL,
            InpaintingModel::FalImageEdit { model, .. } => model.id(),
            InpaintingModel::OpenRouterImageEdit { model, .. } => model,
            InpaintingModel::ReplicateImageEdit { model, .. } => model.id(),
        }
        .to_owned()
    }

    fn recovers_from_memory_pressure(&self, _input: &StageInput) -> bool {
        matches!(
            &self.config,
            InpaintingModel::LaMa {}
                | InpaintingModel::AotInpainting {}
                | InpaintingModel::Flux2Klein(_)
                | InpaintingModel::RoremMixed(_)
        )
    }

    /// A page the user marked as cleaned is left as it is by automatic
    /// cleaning; a drawn mask (removal, restoration) still runs.
    fn skip(&self, input: &StageInput) -> Result<bool> {
        Ok(input.inpainting_intent().is_none()
            && input
                .scene
                .component::<koharu_scene::CleanupDone>(input.page()?)?
                .is_some())
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
        let model = self
            .model
            .lock()
            .await
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("inpainting model is not loaded"))?;
        model.run(input).await
    }
}

#[derive(Clone)]
enum Model {
    BalloonFill,
    LaMa(Arc<Mutex<LaMa>>),
    Aot(Arc<Mutex<AotInpainting>>),
    Flux {
        model: Arc<Mutex<Flux2KleinInpaint>>,
        config: Flux2KleinConfig,
    },
    Rorem {
        model: Arc<Mutex<RoremMixed>>,
        config: RoremMixedConfig,
    },
    Fal {
        client: Fal,
        model: FalModel,
        config: ImageEditConfig,
    },
    OpenRouter {
        client: OpenRouterImage,
        model: String,
        config: ImageEditConfig,
    },
    Replicate {
        client: Replicate,
        model: replicate::ReplicateModel,
        config: ImageEditConfig,
    },
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
            InpaintingModel::BalloonFill {} => Ok(Self::BalloonFill),
            InpaintingModel::LaMa {} => {
                Ok(Self::LaMa(Arc::new(Mutex::new(LaMa::load(device).await?))))
            }
            InpaintingModel::AotInpainting {} => Ok(Self::Aot(Arc::new(Mutex::new(
                AotInpainting::load(device).await?,
            )))),
            InpaintingModel::Flux2Klein(config) => Ok(Self::Flux {
                model: Arc::new(Mutex::new(Flux2KleinInpaint::load(device).await?)),
                config: config.clone(),
            }),
            InpaintingModel::RoremMixed(config) => Ok(Self::Rorem {
                model: Arc::new(Mutex::new(RoremMixed::load(device).await?)),
                config: config.clone(),
            }),
            InpaintingModel::FalImageEdit { model, config } => Ok(Self::Fal {
                client: Fal::new()?,
                model: *model,
                config: config.clone(),
            }),
            InpaintingModel::OpenRouterImageEdit { model, config } => Ok(Self::OpenRouter {
                client: OpenRouterImage::new()?,
                model: model.clone(),
                config: config.clone(),
            }),
            InpaintingModel::ReplicateImageEdit { model, config } => Ok(Self::Replicate {
                client: Replicate::new()?,
                model: *model,
                config: config.clone(),
            }),
        }
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let intent = input.inpainting_mask.as_ref().map(|mask| mask.intent);
        let remove = intent == Some(InpaintingIntent::Remove);
        let restore = intent == Some(InpaintingIntent::Restore);
        let full_page = intent.is_none()
            && matches!(
                self,
                Self::Fal { config, .. } | Self::OpenRouter { config, .. } | Self::Replicate { config, .. }
                    if config.apply_mode == InpaintingApplyMode::FullPage
            );
        // The owner's paint is part of the page every model sees, and is
        // kept out of the cleanup layer so the result never covers it.
        let include_paint = !remove;
        // Automatic local cleaning leaves sound effects for the owner.
        let keep_sound_effects = intent.is_none()
            && !matches!(
                self,
                Self::Fal { .. } | Self::OpenRouter { .. } | Self::Replicate { .. }
            );
        let mut prepared = prepare(&input, include_paint, restore, keep_sound_effects).await?;
        if !full_page && prepared.mask.as_raw().iter().all(|value| *value == 0) {
            return finish(input.scene.edit());
        }
        let target = CleanupTarget {
            dimensions: prepared.original.dimensions(),
            mask: prepared.mask.clone(),
            paint: prepared.paint.take(),
            restore_bounds: prepared.restore_bounds,
            flat_fill: std::mem::take(&mut prepared.flat_fill),
        };
        if let Self::BalloonFill = self {
            ensure!(!restore, "balloon fill does not restore artwork");
            // The fill alone, as its own reviewable layer. A later cloud edit
            // recomputes the same fill from the source.
            if target.flat_fill.is_empty() {
                return finish(input.scene.edit());
            }
            let (width, height) = target.dimensions;
            let mut filled = GrayImage::new(width, height);
            for &(x, y, _) in &target.flat_fill {
                filled.put_pixel(x, y, Luma([u8::MAX]));
            }
            let target = CleanupTarget {
                mask: filled,
                ..target
            };
            return target.patch(&input, BALLOON_FILL, &prepared.image.to_rgba8(), false, &[]);
        }
        let sent = prepared.image.clone();
        let pending = prepared.unfilled_mask.clone();
        let (model_name, image) = match self {
            Self::BalloonFill => unreachable!("balloon fill returns above"),
            Self::LaMa(model) => {
                let model = model.clone();
                (
                    "lama",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        inpaint_tiled(&prepared.image, &prepared.unfilled_mask, |image, mask| {
                            Ok(DynamicImage::ImageRgb8(model.inference(
                                image,
                                mask,
                                &InpaintRequest::default(),
                            )?))
                        })
                    })
                    .await?,
                )
            }
            Self::Aot(model) => {
                let model = model.clone();
                (
                    "aot-inpainting",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("AOT model lock is poisoned"))?;
                        inpaint_tiled(&prepared.image, &prepared.unfilled_mask, |image, mask| {
                            Ok(DynamicImage::ImageRgb8(model.inference(image, mask)?))
                        })
                    })
                    .await?,
                )
            }
            Self::Flux { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "flux2-klein",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                        inpaint_tiled(&prepared.image, &prepared.unfilled_mask, |image, mask| {
                            model.inference(
                                &config.prompt,
                                image,
                                None,
                                &DynamicImage::ImageLuma8(mask.clone()),
                                &Flux2KleinInpaintOptions::default(),
                            )
                        })
                    })
                    .await?,
                )
            }
            Self::Rorem { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "rorem-mixed",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("RORem model lock is poisoned"))?;
                        inpaint_tiled(&prepared.image, &prepared.unfilled_mask, |image, mask| {
                            Ok(DynamicImage::ImageRgb8(model.inference(
                                image,
                                mask,
                                &config.prompt,
                                &config.negative_prompt,
                                &RoremMixedOptions::default(),
                            )?))
                        })
                    })
                    .await?,
                )
            }
            Self::Fal {
                client,
                model,
                config,
                ..
            } => {
                let mut config = if intent.is_none() {
                    protected_cleanup_config(config)
                } else {
                    config.clone()
                };
                let Some(prompt) = reviewed_prompt(
                    &input,
                    crate::InpaintingProvider::Fal,
                    model.id(),
                    &config.prompt,
                    &prepared.image,
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                config.prompt = prompt;
                let Some(image) = client
                    .edit(*model, &config, &prepared.image, &input.stop)
                    .await?
                else {
                    return finish(input.scene.edit());
                };
                (model.id(), image)
            }
            Self::OpenRouter {
                client,
                model,
                config,
                ..
            } => {
                let Some(prompt) = reviewed_prompt(
                    &input,
                    crate::InpaintingProvider::OpenRouter,
                    model,
                    &config.prompt,
                    &prepared.image,
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                let Some(image) = client
                    .edit(
                        model,
                        &prompt,
                        &prepared.image,
                        cleanup_layers(&input)?,
                        &input.stop,
                    )
                    .await?
                else {
                    return finish(input.scene.edit());
                };
                (model.as_str(), image)
            }
            Self::Replicate {
                client,
                model,
                config,
                ..
            } => {
                let Some(prompt) = reviewed_prompt(
                    &input,
                    crate::InpaintingProvider::Replicate,
                    model.id(),
                    &config.prompt,
                    &prepared.image,
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                let Some(image) = client
                    .edit(*model, &prompt, &prepared.image, &input.stop)
                    .await?
                else {
                    return finish(input.scene.edit());
                };
                (model.id(), image)
            }
        };
        // Local models leave screentone as grey mush; redraw its dots.
        let image = if !restore
            && matches!(
                self,
                Self::LaMa(_) | Self::Aot(_) | Self::Flux { .. } | Self::Rorem { .. }
            ) {
            let source = sent.to_rgb8();
            tokio_rayon::spawn(move || {
                let mut image = image.to_rgb8();
                screentone::rescreen(&source, &mut image, &pending);
                DynamicImage::ImageRgb8(image)
            })
            .await
        } else {
            image
        };
        let residual = if intent.is_none()
            && matches!(
                self,
                Self::Fal { .. } | Self::OpenRouter { .. } | Self::Replicate { .. }
            ) {
            residual_regions(&input, &sent, &image)?
        } else {
            Vec::new()
        };
        target.patch(&input, model_name, &image.to_rgba8(), full_page, &residual)
    }
}

struct CleanupTarget {
    dimensions: (u32, u32),
    mask: GrayImage,
    paint: Option<RgbaImage>,
    restore_bounds: Option<[u32; 4]>,
    flat_fill: Vec<(u32, u32, Rgb<u8>)>,
}

impl CleanupTarget {
    fn patch(
        &self,
        input: &StageInput,
        model: &str,
        image: &RgbaImage,
        full_page: bool,
        residual: &[EntityId],
    ) -> Result<koharu_scene::Patch> {
        let page = input.page()?;
        let intent = input.inpainting_intent();
        let mut edit = if intent == Some(InpaintingIntent::Remove) {
            input.scene.edit()
        } else {
            input.scene.edit_as(generation(PRODUCER, model)?)
        };
        edit.observe_assets(page)?;
        if intent == Some(InpaintingIntent::Remove) {
            edit.observe_subtree(page)?;
        }
        let expected_dimensions = self
            .restore_bounds
            .map_or(self.dimensions, |[left, top, right, bottom]| {
                (right - left, bottom - top)
            });
        if image.dimensions() != expected_dimensions || self.mask.dimensions() != self.dimensions {
            bail!("inpainted image dimensions do not match page {page}");
        }
        let mut overlay = cleanup_overlay(image, &self.mask, full_page, self.restore_bounds);
        lock_flat_fill(&mut overlay, &self.mask, full_page, &self.flat_fill);
        exclude_paint_from_cleanup(&mut overlay, self.paint.as_ref());
        let mut bytes = Cursor::new(Vec::new());
        let width = overlay.width();
        let height = overlay.height();
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
        let at = match intent {
            Some(InpaintingIntent::Remove) | None => {
                // Keep editable text separate; cover every raster that supplied the model input.
                let mut at = At::Start;
                for child in input.scene.children(page)? {
                    for entity in input.scene.subtree(child)? {
                        if input.scene.component::<RasterLayer>(entity.id())?.is_some() {
                            at = At::After(child);
                            break;
                        }
                    }
                }
                at
            }
            Some(InpaintingIntent::Restore) => restoration_layer_placement(&input.scene, page)?,
        };
        let entity = edit.add_entity(page, at)?;
        edit.set(
            entity,
            &RasterLayer {
                origin: Origin::User,
                name: "Cleanup".to_owned(),
                kind: RasterLayerKind::Cleanup,
            },
        )?;
        edit.set_asset(
            entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: if residual.is_empty() {
                        BTreeMap::new()
                    } else {
                        BTreeMap::from([(
                            RESIDUAL_TEXT_ATTRIBUTE.to_owned(),
                            residual
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        )])
                    },
                },
            ),
        )?;
        finish(edit)
    }
}

async fn reviewed_prompt(
    input: &StageInput,
    provider: crate::InpaintingProvider,
    model: &str,
    prompt: &str,
    image: &Arc<DynamicImage>,
) -> Result<Option<String>> {
    if input.stop.stopped() {
        return Ok(None);
    }
    let Some(reviewer) = &input.image_edit_reviewer else {
        return Ok(Some(prompt.to_owned()));
    };
    let page = input.page()?;
    let request = crate::ImageEditReview {
        page,
        page_label: input.scene.page(page)?.page()?.label,
        provider,
        model: model.to_owned(),
        prompt: prompt.to_owned(),
        image: image.clone(),
        restoration: input.inpainting_intent() == Some(InpaintingIntent::Restore),
    };
    let result = tokio::select! {
        biased;
        () = input.stop.cancelled() => return Ok(None),
        result = reviewer.review(request, &input.stop) => result,
    };
    if input.stop.stopped() {
        return Ok(None);
    }
    let Some(prompt) = result? else {
        input.stop.stop();
        return Ok(None);
    };
    ensure!(!prompt.trim().is_empty(), "image edit prompt is empty");
    ensure!(!prompt.contains('\0'), "image edit prompt contains NUL");
    Ok(Some(prompt))
}

fn cleanup_overlay(
    image: &RgbaImage,
    mask: &GrayImage,
    full_page: bool,
    restore_bounds: Option<[u32; 4]>,
) -> RgbaImage {
    let mut overlay = RgbaImage::new(mask.width(), mask.height());
    let [left, top, _, _] = restore_bounds.unwrap_or([0, 0, mask.width(), mask.height()]);
    for (x, y, target) in overlay.enumerate_pixels_mut() {
        if !full_page && mask.get_pixel(x, y)[0] < 127 {
            continue;
        }
        let generated = image.get_pixel(x - left, y - top);
        // Opaque cleanup texels prevent linear canvas sampling from revealing
        // the original lettering at edited edges.
        *target = Rgba([generated[0], generated[1], generated[2], 255]);
    }
    overlay
}

// Image-edit models see the whole page and may repaint a balloon they were
// told to keep, e.g. turn a dark balloon white. Uniform balloons were already
// cleaned without a model, so their pixels override whatever the model returned.
fn lock_flat_fill(
    cleanup: &mut RgbaImage,
    mask: &GrayImage,
    full_page: bool,
    flat_fill: &[(u32, u32, Rgb<u8>)],
) {
    for &(x, y, color) in flat_fill {
        if full_page || mask.get_pixel(x, y)[0] >= 127 {
            cleanup.put_pixel(x, y, Rgba([color[0], color[1], color[2], 255]));
        }
    }
}

fn exclude_paint_from_cleanup(cleanup: &mut RgbaImage, paint: Option<&RgbaImage>) {
    let Some(paint) = paint else {
        return;
    };
    for (target, painted) in cleanup.pixels_mut().zip(paint.pixels()) {
        if painted[3] != 0 {
            target[3] = 0;
        }
    }
}

fn protected_cleanup_config(config: &ImageEditConfig) -> ImageEditConfig {
    let mut config = config.clone();
    config.prompt = match config.prompt.trim() {
        DEFAULT_CLEANUP_PROMPT => DEFAULT_CLEANUP_PROMPT.to_owned(),
        prompt => format!("{prompt}\n\nSafety boundary: {DEFAULT_CLEANUP_PROMPT}"),
    };
    config
}

fn mask_bounds(mask: &GrayImage) -> Option<[u32; 4]> {
    let mut bounds = None::<[u32; 4]>;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] == 0 {
            continue;
        }
        bounds = Some(match bounds {
            Some([left, top, right, bottom]) => {
                [left.min(x), top.min(y), right.max(x + 1), bottom.max(y + 1)]
            }
            None => [x, y, x + 1, y + 1],
        });
    }
    bounds
}

/// Whether a page already has a cleanup layer, from a run or a manual edit.
pub(crate) fn cleaned(
    scene: &koharu_scene::Snapshot,
    page: koharu_scene::EntityId,
) -> Result<bool> {
    if scene
        .component::<koharu_scene::CleanupDone>(page)?
        .is_some()
    {
        return Ok(true);
    }
    for child in scene.children(page)? {
        if scene
            .component::<RasterLayer>(child)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn restoration_layer_placement(
    scene: &koharu_scene::Snapshot,
    page: koharu_scene::EntityId,
) -> Result<At> {
    if let Some(paint) = scene.children(page)?.find(|entity| {
        scene
            .component::<RasterLayer>(*entity)
            .ok()
            .flatten()
            .is_some_and(|layer| layer.kind == RasterLayerKind::Paint)
    }) {
        return Ok(At::Before(paint));
    }
    Ok(scene
        .page(page)?
        .text_group()?
        .map_or(At::End, |group| At::Before(group.id())))
}

#[derive(Clone, Debug)]
struct FlatFillRegion {
    bounds: [u32; 4],
    polygon: Vec<(f32, f32)>,
}

/// Removes from the mask each lettering blob lying mostly inside a text region
/// whose text is a sound effect, unless that region sits in a detected
/// balloon: detection calls small shouts in balloons ("HUH?") sound effects,
/// and a balloon cleans like any other.
fn drop_sound_effects(
    input: &StageInput,
    mask: &mut GrayImage,
    text_mask: &mut GrayImage,
) -> Result<()> {
    let (width, height) = mask.dimensions();
    let balloons = flat_fill_regions(input, width, height)?
        .into_iter()
        .map(|region| {
            let inside = polygon_interior(&region);
            (region, inside)
        })
        .collect::<Vec<_>>();
    let in_balloon = |x: f64, y: f64| {
        let (x, y) = (x.floor() as i64, y.floor() as i64);
        balloons.iter().any(|(region, inside)| {
            let [left, top, right, bottom] = region.bounds;
            x >= i64::from(left)
                && y >= i64::from(top)
                && x < i64::from(right)
                && y < i64::from(bottom)
                && inside.contains(x as u32, y as u32)
        })
    };
    let mut effect = vec![false; width as usize * height as usize];
    let mut any = false;
    for entity in input.scene.descendants(input.page()?)? {
        let content = entity.id();
        if !input
            .scene
            .component::<TextRole>(content)?
            .is_some_and(|role| role.role == super::detection::ONOMATOPOEIA_ROLE)
        {
            continue;
        }
        let Some(relation) = input.scene.relation_from::<RecognizedFrom>(content)? else {
            continue;
        };
        let Some(geometry) = input.scene.component::<Geometry>(relation.value().target)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&geometry) else {
            continue;
        };
        if in_balloon((left + right) / 2.0, (top + bottom) / 2.0) {
            continue;
        }
        let bounds = [
            left.floor().clamp(0.0, f64::from(width)) as u32,
            top.floor().clamp(0.0, f64::from(height)) as u32,
            right.ceil().clamp(0.0, f64::from(width)) as u32,
            bottom.ceil().clamp(0.0, f64::from(height)) as u32,
        ];
        if polygon.len() < 3 || bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
            continue;
        }
        let inside = polygon_interior(&FlatFillRegion { bounds, polygon });
        for y in bounds[1]..bounds[3] {
            for x in bounds[0]..bounds[2] {
                if inside.contains(x, y) {
                    effect[y as usize * width as usize + x as usize] = true;
                    any = true;
                }
            }
        }
    }
    if !any {
        return Ok(());
    }
    let (labels, components) = mask_components(mask);
    let mut counts = vec![[0u32; 2]; components.len() + 1];
    for (at, label) in labels.as_raw().iter().enumerate() {
        if *label != 0 {
            counts[*label as usize][usize::from(effect[at])] += 1;
        }
    }
    for (at, label) in labels.as_raw().iter().enumerate() {
        let [outside, inside] = counts[*label as usize];
        if *label != 0 && inside > outside {
            let (x, y) = (at as u32 % width, at as u32 / width);
            mask.put_pixel(x, y, Luma([0]));
            text_mask.put_pixel(x, y, Luma([0]));
        }
    }
    Ok(())
}

fn flat_fill_regions(input: &StageInput, width: u32, height: u32) -> Result<Vec<FlatFillRegion>> {
    let mut regions = Vec::new();
    for entity in input.scene.descendants(input.page()?)? {
        let id = entity.id();
        let is_bubble = input
            .scene
            .component::<Region>(id)?
            .is_some_and(|region| region.kind == BubbleRegion::kind());
        if !is_bubble {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        if polygon.len() < 3 {
            continue;
        }
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let bounds = [
            left.floor().clamp(0.0, width as f32) as u32,
            top.floor().clamp(0.0, height as f32) as u32,
            right.ceil().clamp(0.0, width as f32) as u32,
            bottom.ceil().clamp(0.0, height as f32) as u32,
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push(FlatFillRegion { bounds, polygon });
        }
    }
    Ok(regions)
}

struct InpaintInput {
    image: Arc<DynamicImage>,
    original: Arc<DynamicImage>,
    paint: Option<RgbaImage>,
    mask: GrayImage,
    /// `mask` minus the pixels already cleaned by uniform balloon fill.
    unfilled_mask: GrayImage,
    /// Page-space balloon pixels a model result must not change.
    flat_fill: Vec<(u32, u32, Rgb<u8>)>,
    restore_bounds: Option<[u32; 4]>,
}

/// `keep_sound_effects` leaves lettering of sound-effect regions out of the
/// mask: a local fill or model makes a ragged half-erased effect of it, and
/// the owner cleans those by hand or through an image-edit model.
async fn prepare(
    input: &StageInput,
    include_paint: bool,
    include_cleanup: bool,
    keep_sound_effects: bool,
) -> Result<InpaintInput> {
    let page = input.page()?;
    let original = input
        .images
        .page_image(&input.scene, page)
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let remove = input.inpainting_intent() == Some(InpaintingIntent::Remove);
    let automatic = input.inpainting_intent().is_none();
    let mut source = original.clone();
    let mut paint = None;
    if remove || include_paint || include_cleanup || automatic {
        let mut composite = original.to_rgba8();
        for entity in input.scene.descendants(page)? {
            let entity = entity.id();
            let Some(raster) = input.scene.component::<RasterLayer>(entity)? else {
                continue;
            };
            // The enhancement layer is already the base image.
            if raster.kind == RasterLayerKind::Enhancement {
                continue;
            }
            let is_paint = raster.kind == RasterLayerKind::Paint;
            // Automatic cleaning starts from the page as it shows: every
            // visible cleanup layer, the owner's and earlier results alike.
            // Hiding a layer keeps it out, so hiding them all sends the page
            // itself.
            if !(remove
                || is_paint && include_paint
                || raster.kind == RasterLayerKind::Cleanup && (include_cleanup || automatic))
            {
                continue;
            }
            let mut opacity = 1.0;
            let mut ancestor = Some(entity);
            while let Some(current) = ancestor.filter(|current| *current != page) {
                if let Some(visibility) = input.scene.component::<Visibility>(current)? {
                    if !visibility.visible {
                        opacity = 0.0;
                        break;
                    }
                    opacity *= visibility.opacity;
                }
                ancestor = input.scene.parent(current)?;
            }
            let opacity = opacity.clamp(0.0, 1.0);
            if opacity == 0.0 {
                continue;
            }
            let Some(image) = input.images.get(&input.scene, entity, "source").await? else {
                continue;
            };
            let mut layer = image.to_rgba8();
            if layer.dimensions() != original.dimensions() {
                bail!("raster layer dimensions do not match page {page}");
            }
            if opacity != 1.0 {
                for pixel in layer.pixels_mut() {
                    pixel[3] = (f32::from(pixel[3]) * opacity).round() as u8;
                }
            }
            if is_paint && !remove {
                image::imageops::overlay(
                    paint.get_or_insert_with(|| RgbaImage::new(layer.width(), layer.height())),
                    &layer,
                    0,
                    0,
                );
            }
            image::imageops::overlay(&mut composite, &layer, 0, 0);
        }
        source = Arc::new(DynamicImage::ImageRgba8(composite));
    }
    let mut mask = GrayImage::new(source.width(), source.height());
    let mut text_mask = GrayImage::new(source.width(), source.height());
    if let Some(transient) = &input.inpainting_mask {
        let layer = image::load_from_memory(&transient.png)?.to_luma8();
        if layer.dimensions() != mask.dimensions() {
            bail!("inpainting mask dimensions do not match page {page}");
        }
        mask = layer;
    } else {
        for role in ["text-mask", "coo-mask"] {
            if let Some(image) = input.images.get(&input.scene, page, role).await? {
                let layer = image.to_luma8();
                if layer.dimensions() != mask.dimensions() {
                    bail!("{role} dimensions do not match page {page}");
                }
                for (target, source) in mask.iter_mut().zip(layer.as_raw()) {
                    *target = (*target).max(*source);
                }
                if role == "text-mask" {
                    text_mask = layer;
                }
            }
        }
    }
    if keep_sound_effects {
        drop_sound_effects(input, &mut mask, &mut text_mask)?;
    }
    if let Some(bounds) = input.region {
        for (x, y, pixel) in mask.enumerate_pixels_mut() {
            if f64::from(x + 1) <= bounds.x
                || f64::from(y + 1) <= bounds.y
                || f64::from(x) >= bounds.x + bounds.width
                || f64::from(y) >= bounds.y + bounds.height
            {
                *pixel = Luma([0]);
                text_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    let balloons = flat_fill_regions(input, source.width(), source.height())?;
    let mut filled = source.to_rgb8();
    let mut unfilled_mask = mask.clone();
    let flat_fill = fill_flat_lettering(&mut filled, &mut unfilled_mask, &text_mask, &balloons);
    if !flat_fill.is_empty() {
        source = Arc::new(DynamicImage::ImageRgb8(filled));
    }
    let restore_bounds = if input.inpainting_intent() == Some(InpaintingIntent::Restore) {
        mask_bounds(&mask)
    } else {
        None
    };
    if let Some([left, top, right, bottom]) = restore_bounds {
        source = Arc::new(source.crop_imm(left, top, right - left, bottom - top));
    }
    Ok(InpaintInput {
        image: source,
        original,
        paint,
        mask,
        unfilled_mask,
        flat_fill,
        restore_bounds,
    })
}

/// Text boxes whose lettering a cloud result left untouched. In each
/// detected text box the fine detail (a pixel minus its 5x5 mean) of the
/// result is compared with the page that was sent: lettering left in place
/// keeps the same detail (correlation 0.66-0.94 on HY results), redrawn
/// artwork does not (below 0.3). A box counts when the correlation is at
/// least 0.5 and the result keeps at least 75% of the detail (untouched
/// letters kept 83-97%). Letters the cloud only faded are left out: a
/// second request did not erase them and left a blurred patch instead.
/// Boxes whose sent detail is faint (a balloon already filled locally) are
/// skipped.
fn residual_boxes(sent: &GrayImage, result: &GrayImage, boxes: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let detail = |image: &GrayImage, x: u32, y: u32| -> f64 {
        let (width, height) = image.dimensions();
        let mut sum = 0.0;
        for dy in -2_i64..=2 {
            for dx in -2_i64..=2 {
                let sx = (i64::from(x) + dx).clamp(0, i64::from(width) - 1) as u32;
                let sy = (i64::from(y) + dy).clamp(0, i64::from(height) - 1) as u32;
                sum += f64::from(image.get_pixel(sx, sy)[0]);
            }
        }
        f64::from(image.get_pixel(x, y)[0]) - sum / 25.0
    };
    boxes
        .iter()
        .copied()
        .filter(|&[left, top, right, bottom]| {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut count) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for y in top..bottom {
                for x in left..right {
                    let (a, b) = (detail(sent, x, y), detail(result, x, y));
                    sa += a;
                    sb += b;
                    saa += a * a;
                    sbb += b * b;
                    sab += a * b;
                    count += 1.0;
                }
            }
            if count < 16.0 {
                return false;
            }
            let variance_sent = saa / count - (sa / count).powi(2);
            let variance_result = sbb / count - (sb / count).powi(2);
            if variance_sent.sqrt() < RESIDUAL_MIN_DETAIL {
                return false;
            }
            let correlation = (sab / count - sa * sb / (count * count))
                / (variance_sent * variance_result).sqrt().max(1e-9);
            correlation >= 0.5 && variance_result >= 0.5625 * variance_sent
        })
        .collect()
}

/// Cleanup layers already on the page. Cleaning again sends a new seed, so a
/// rerun gives another result instead of the same one.
fn cleanup_layers(input: &StageInput) -> Result<u64> {
    let mut count = 0;
    for entity in input.scene.descendants(input.page()?)? {
        if input
            .scene
            .component::<RasterLayer>(entity.id())?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
        {
            count += 1;
        }
    }
    Ok(count)
}

/// The text regions of the page whose lettering the cloud answer left
/// untouched (see `residual_boxes`). They are recorded on the cleanup layer
/// for review instead of sending a second request: on HY a second request
/// did not erase what the first left, and blurred faded letters.
fn residual_regions(
    input: &StageInput,
    sent: &DynamicImage,
    result: &DynamicImage,
) -> Result<Vec<EntityId>> {
    if result.dimensions() != sent.dimensions() {
        return Ok(Vec::new());
    }
    let page = input.page()?;
    let (width, height) = sent.dimensions();
    let mut regions = Vec::new();
    for entity in input.scene.descendants(page)? {
        let id = entity.id();
        if !input
            .scene
            .component::<Region>(id)?
            .is_some_and(|region| region.kind == TextRegion::kind())
        {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&geometry) else {
            continue;
        };
        let clamp = |value: f64, limit: u32| value.clamp(0.0, f64::from(limit)) as u32;
        let bounds = [
            clamp(left.floor(), width),
            clamp(top.floor(), height),
            clamp(right.ceil(), width),
            clamp(bottom.ceil(), height),
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push((id, bounds));
        }
    }
    let boxes: Vec<[u32; 4]> = regions.iter().map(|(_, bounds)| *bounds).collect();
    let residual = residual_boxes(&sent.to_luma8(), &result.to_luma8(), &boxes);
    Ok(regions
        .into_iter()
        .filter(|(_, bounds)| residual.contains(bounds))
        .map(|(id, _)| id)
        .collect())
}

/// Asset attribute of a cleanup layer listing the text regions whose
/// lettering the cloud answer left in place, comma separated.
const RESIDUAL_TEXT_ATTRIBUTE: &str = "dev.koharu.cleanup.residual-text";

/// Whether the cloud cleaning left the lettering of a text region in place.
/// The topmost visible cleanup layer of the page decides, so cleaning again
/// or a later manual cleanup clears the flag.
pub fn cleaning_unverified(scene: &koharu_scene::Snapshot, region: EntityId) -> Result<bool> {
    let mut page = region;
    while let Some(parent) = scene.parent(page)? {
        page = parent;
    }
    let mut topmost = None;
    for entity in scene.descendants(page)? {
        let id = entity.id();
        if scene
            .component::<RasterLayer>(id)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
            && scene
                .component::<Visibility>(id)?
                .is_none_or(|visibility| visibility.visible && visibility.opacity > 0.0)
        {
            topmost = Some(id);
        }
    }
    let Some(layer) = topmost else {
        return Ok(false);
    };
    let Some(asset) = scene.asset(layer, &AssetRole::new("source")?)? else {
        return Ok(false);
    };
    let region = region.to_string();
    Ok(asset
        .metadata
        .attributes
        .get(RESIDUAL_TEXT_ATTRIBUTE)
        .is_some_and(|ids| ids.split(',').any(|id| id == region)))
}

const RESIDUAL_MIN_DETAIL: f64 = 8.0;

const TILE_SIZE: u32 = 512;
const TILE_CONTEXT: u32 = 128;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;
const BALLOON_FILL: &str = "balloon-fill";
/// Most a channel of a lettering pixel's surroundings may vary for the pixel
/// to be filled flat. Paper grain on a scan stays within it; a background
/// seen through a translucent balloon, an outline or screentone does not.
const FLAT_FILL_MAX_DEVIATION: f64 = 5.0;
/// A lettering pixel needs known surroundings on at least a sixth of its
/// neighbourhood (`window / FLAT_FILL_MIN_SHARE` pixels).
const FLAT_FILL_MIN_SHARE: u32 = 6;
/// Brightness distance from a balloon's own colour within which a pixel
/// counts as that balloon: the ends of letters a mask missed fall outside.
const FLAT_FILL_INLIER_RANGE: u8 = 24;
/// Share of a neighbourhood's unmasked pixels that must be the balloon's own
/// colour for a flat fill: a few stray letter ends pass, hatching does not.
const FLAT_FILL_MIN_INLIERS: f64 = 0.6;
/// Share of a balloon's unmasked interior that must be its own colour for
/// all its lettering to fill at once.
const FLAT_FILL_BALLOON_INLIERS: f64 = 0.9;
/// Side of the page tiles lettering is grouped by for the fill.
const FLAT_FILL_TILE: u32 = 128;
/// Passes that grow the fill inward, one neighbourhood radius each.
const FLAT_FILL_PASSES: usize = 12;
const MASKED_FILL_TOLERANCE: u8 = 32;
const MASKED_FILL_MIN_SHARE: f64 = 0.6;

#[derive(Clone, Debug, PartialEq, Eq)]
struct InpaintTile {
    components: Vec<u32>,
    core: [u32; 4],
    crop: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaskComponent {
    label: u32,
    bounds: [u32; 4],
}

// BallonsTranslator avoids model inference for a text block when the non-text
// pixels inside its balloon are nearly uniform, and otherwise sends an enlarged
// block crop to the inpainter:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/base.py#L168-L200
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first for every provider; only the remaining mask is split into
// bounded model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    pending_mask: &GrayImage,
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        image.dimensions() == pending_mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}",
        image.dimensions(),
        pending_mask.dimensions()
    );

    let mut output = image.to_rgb8();
    let (component_labels, tiles) = inpaint_tiles(pending_mask);
    for tile in &tiles {
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&output, left, top, crop_width, crop_height).to_image();
        let crop_mask = crop_tile_mask(pending_mask, tile);

        let generated = inference(&DynamicImage::ImageRgb8(crop_image), &crop_mask)?;
        let generated = if generated.dimensions() == (crop_width, crop_height) {
            generated.to_rgb8()
        } else {
            generated
                .resize_exact(
                    crop_width,
                    crop_height,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgb8()
        };
        composite_generated(&mut output, &component_labels, tile, &generated);
    }
    Ok(DynamicImage::ImageRgb8(output))
}

// Flat fill is kept to classic balloons: a smooth closed outline, plain
// white or black inside, all lettering clear of the outline. A fill is locked
// against the cloud result, so on anything else it did harm the model does
// not: it erased the spikes of ragged balloons, cut outlines that hearts or
// sound effects cross, and turned translucent balloons into opaque patches.
// Those balloons go to the model whole (docs/fork/cleaning-models.ru.md).

/// Fills lettering whose surroundings are one flat colour, pixel by pixel,
/// and returns every pixel a model result must not change, with its colour.
///
/// Each lettering pixel looks at the pixels outside the mask within
/// `flat_fill_radius` of it. When they are flat (every channel's standard
/// deviation at most `FLAT_FILL_MAX_DEVIATION`), it takes their mean;
/// otherwise it stays for the model. A pixel with no unmasked pixels that
/// close reads the pixels filled on earlier passes instead, so the fill
/// reaches the middle of thick strokes, but only when the unmasked pixels
/// three radii around are flat and of the same colour: otherwise the fill
/// would creep along a mask into artwork it never saw. Nothing
/// depends on a balloon being closed or classically shaped: lettering on an
/// open balloon, on plain paper or in a tinted or gradient balloon fills,
/// while letters touching an outline, a sound effect over artwork and a
/// translucent balloon with the background showing through keep enough
/// texture around them to go to the model. Inside a detected balloon only
/// pixels of the same balloon count, so its fill never borrows the colour
/// outside it; a balloon whose mask covers its whole interior (white
/// lettering on a dark balloon) takes its dominant interior colour.
fn fill_flat_lettering(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    balloons: &[FlatFillRegion],
) -> Vec<(u32, u32, Rgb<u8>)> {
    let (width, height) = output.dimensions();
    let before = output.clone();
    let index = |x: u32, y: u32| y as usize * width as usize + x as usize;
    let mut balloon = vec![0u16; width as usize * height as usize];
    let interiors = balloons.iter().map(polygon_interior).collect::<Vec<_>>();
    for (number, (region, inside)) in balloons.iter().zip(&interiors).enumerate() {
        let [left, top, right, bottom] = region.bounds;
        for y in top..bottom {
            for x in left..right {
                if inside.contains(x, y) {
                    balloon[index(x, y)] = number as u16 + 1;
                }
            }
        }
    }
    let original = pending_mask
        .as_raw()
        .iter()
        .map(|value| *value < 127)
        .collect::<Vec<_>>();
    let mut known = original.clone();
    let radius = flat_fill_radius(width, height);
    let minimum = u64::from(((2 * radius + 1).pow(2) / FLAT_FILL_MIN_SHARE).max(6));
    let reach = 3 * radius + 1;
    // Lettering is grouped by page tile; each group's crop is its tile plus
    // what its neighbourhoods and their context can reach.
    let columns = width.div_ceil(FLAT_FILL_TILE);
    let tiles = (columns * height.div_ceil(FLAT_FILL_TILE)) as usize;
    let mut groups = vec![Vec::new(); tiles];
    for (at, (pending, text)) in pending_mask
        .as_raw()
        .iter()
        .zip(text_mask.as_raw())
        .enumerate()
    {
        if *pending >= 127 && *text >= 127 {
            let (x, y) = ((at % width as usize) as u32, (at / width as usize) as u32);
            groups[(y / FLAT_FILL_TILE * columns + x / FLAT_FILL_TILE) as usize].push((x, y));
        }
    }
    let crops = (0..tiles as u32)
        .map(|tile| {
            let (left, top) = (
                tile % columns * FLAT_FILL_TILE,
                tile / columns * FLAT_FILL_TILE,
            );
            [
                left.saturating_sub(reach),
                top.saturating_sub(reach),
                (left + FLAT_FILL_TILE + reach).min(width),
                (top + FLAT_FILL_TILE + reach).min(height),
            ]
        })
        .collect::<Vec<_>>();
    let fill = |output: &mut RgbImage,
                pending_mask: &mut GrayImage,
                known: &mut Vec<bool>,
                fills: Vec<(u32, u32, Rgb<u8>, u16)>| {
        for (x, y, color, _) in fills {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
            known[index(x, y)] = true;
        }
    };
    // A detected balloon whose unmasked interior is one flat colour fills all
    // its lettering with it at once, however little of it shows around the
    // letters (small balloons).
    for (number, (region, inside)) in balloons.iter().zip(&interiors).enumerate() {
        let label = number as u16 + 1;
        let [left, top, right, bottom] = region.bounds;
        let mut pixels = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if inside.contains(x, y) && original[index(x, y)] {
                    pixels.push(*output.get_pixel(x, y));
                }
            }
        }
        if pixels.len() < UNIFORM_BACKGROUND_MIN_PIXELS {
            continue;
        }
        let mut brightness = pixels.iter().map(luma).collect::<Vec<_>>();
        let reference = median(&mut brightness);
        let inliers = pixels
            .iter()
            .copied()
            .filter(|pixel| luma(pixel).abs_diff(reference) <= FLAT_FILL_INLIER_RANGE)
            .collect::<Vec<_>>();
        if (inliers.len() as f64) < FLAT_FILL_BALLOON_INLIERS * pixels.len() as f64 {
            continue;
        }
        let Some(color) = flat_color(&inliers) else {
            continue;
        };
        let fills = groups
            .iter()
            .flatten()
            .copied()
            .filter(|&(x, y)| balloon[index(x, y)] == label)
            .map(|(x, y)| (x, y, color, label))
            .collect::<Vec<_>>();
        fill(output, pending_mask, &mut known, fills);
    }
    for group in &mut groups {
        group.retain(|&(x, y)| !known[index(x, y)]);
    }
    // First, every pixel with enough unmasked surroundings, from those alone.
    // The rest are deep inside strokes and remember the colour of the
    // unmasked pixels three radii around, when those are flat.
    let mut deep = Vec::new();
    for (group, &crop) in groups.iter().zip(&crops) {
        if group.is_empty() {
            continue;
        }
        let mut labels = group
            .iter()
            .map(|&(x, y)| balloon[index(x, y)])
            .collect::<Vec<_>>();
        labels.sort_unstable();
        labels.dedup();
        let mut fills = Vec::new();
        for label in labels {
            let members = group
                .iter()
                .copied()
                .filter(|&(x, y)| balloon[index(x, y)] == label)
                .collect::<Vec<_>>();
            // Only as far as this balloon's own letters can reach.
            let [left, top, right, bottom] = members.iter().fold(
                [u32::MAX, u32::MAX, 0, 0],
                |[left, top, right, bottom], &(x, y)| {
                    [left.min(x), top.min(y), right.max(x + 1), bottom.max(y + 1)]
                },
            );
            let crop = [
                left.saturating_sub(reach).max(crop[0]),
                top.saturating_sub(reach).max(crop[1]),
                (right + reach).min(crop[2]),
                (bottom + reach).min(crop[3]),
            ];
            // Lettering inside a balloon reads only that balloon; lettering
            // outside every detected balloon (often a traced outline that
            // missed part of the text) reads anything around it.
            let same = |at: usize| label == 0 || balloon[at] == label;
            let include = |x: u32, y: u32| original[index(x, y)] && same(index(x, y));
            // The balloon's (or the paper's) own colour: the median of its
            // unmasked pixels here. Pixels far from it, such as the ends of
            // letters the mask missed, do not count against a flat fill.
            let mut histogram = [0u32; 256];
            for y in crop[1]..crop[3] {
                for x in crop[0]..crop[2] {
                    if include(x, y) {
                        histogram[usize::from(luma(output.get_pixel(x, y)))] += 1;
                    }
                }
            }
            let half = histogram.iter().sum::<u32>().div_ceil(2);
            let mut seen_so_far = 0;
            let reference = histogram
                .iter()
                .position(|count| {
                    seen_so_far += count;
                    seen_so_far >= half
                })
                .unwrap_or(0) as u8;
            let seen = WindowSums::new(output, crop, include, reference);
            for (x, y) in members {
                let (count, color) = seen.flat_mean(x, y, radius);
                if count >= minimum {
                    if let Some(color) = color {
                        fills.push((x, y, color, label));
                    }
                } else {
                    // The unmasked pixels three radii around, or when a large
                    // mask leaves none that close, the whole crop's.
                    let context = match seen.flat_mean(x, y, 3 * radius) {
                        (0, _) => {
                            seen.flat_mean(x, y, crop[2] - crop[0] + crop[3] - crop[1])
                                .1
                        }
                        (_, context) => context,
                    };
                    if let Some(context) = context {
                        deep.push((x, y, label, context, crop, reference));
                    }
                }
            }
        }
        fill(output, pending_mask, &mut known, fills);
    }
    // Then the deep pixels grow inward from filled ones, pass by pass.
    for _ in 1..FLAT_FILL_PASSES {
        let mut fills = Vec::new();
        let mut tables = Vec::<([u32; 4], u16, WindowSums)>::new();
        deep.retain(|&(x, y, label, context, crop, reference)| {
            let near = match tables
                .iter()
                .position(|(seen_crop, seen_label, _)| *seen_crop == crop && *seen_label == label)
            {
                Some(table) => &tables[table].2,
                None => {
                    tables.push((
                        crop,
                        label,
                        WindowSums::new(
                            output,
                            crop,
                            |x, y| {
                                known[index(x, y)] && (label == 0 || balloon[index(x, y)] == label)
                            },
                            reference,
                        ),
                    ));
                    &tables[tables.len() - 1].2
                }
            };
            let (inner, color) = near.flat_mean(x, y, radius);
            let Some(color) = color.filter(|color| {
                inner >= minimum
                    && (0..3).all(|channel| {
                        f64::from(color[channel].abs_diff(context[channel]))
                            <= 2.0 * FLAT_FILL_MAX_DEVIATION
                    })
            }) else {
                return true;
            };
            fills.push((x, y, color, label));
            false
        });
        if fills.is_empty() {
            break;
        }
        fill(output, pending_mask, &mut known, fills);
    }
    let targets = groups
        .concat()
        .into_iter()
        .filter(|&(x, y)| !known[index(x, y)])
        .collect::<Vec<_>>();
    for (number, (region, inside)) in balloons.iter().zip(&interiors).enumerate() {
        let label = number as u16 + 1;
        let [left, top, right, bottom] = region.bounds;
        let remaining = targets
            .iter()
            .filter(|&&(x, y)| balloon[index(x, y)] == label)
            .copied()
            .collect::<Vec<_>>();
        let known_inside = (top..bottom)
            .flat_map(|y| (left..right).map(move |x| (x, y)))
            .filter(|&(x, y)| inside.contains(x, y) && known[index(x, y)])
            .count();
        if !remaining.is_empty()
            && known_inside < UNIFORM_BACKGROUND_MIN_PIXELS
            && let Some(color) = uniform_region_color(output, pending_mask, region, inside)
        {
            for (x, y) in remaining {
                output.put_pixel(x, y, color);
                pending_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    // Lettering is filled whole or not at all: a block the fill could only
    // partly erase goes back to the model intact, since an image-edit model
    // takes the fragments of half-erased letters for drawing and keeps them.
    let lettering = GrayImage::from_fn(width, height, |x, y| {
        Luma([u8::from(!original[index(x, y)] && text_mask.get_pixel(x, y)[0] >= 127) * u8::MAX])
    });
    let (blocks, count) = {
        let (labels, components) = mask_components(&lettering);
        (labels, components.len())
    };
    let mut unfinished = vec![false; count + 1];
    for (at, block) in blocks.as_raw().iter().enumerate() {
        if *block != 0 && pending_mask.as_raw()[at] >= 127 {
            unfinished[*block as usize] = true;
        }
    }
    let mut filled = Vec::new();
    let mut filled_in = vec![false; balloons.len() + 1];
    for (at, block) in blocks.as_raw().iter().enumerate() {
        if *block == 0 {
            continue;
        }
        let (x, y) = (at as u32 % width, at as u32 / width);
        if unfinished[*block as usize] {
            output.put_pixel(x, y, *before.get_pixel(x, y));
            pending_mask.put_pixel(x, y, Luma([u8::MAX]));
            continue;
        }
        let label = balloon[at];
        filled_in[usize::from(label)] = true;
        if label == 0 {
            filled.push((x, y, *output.get_pixel(x, y)));
        }
    }
    // A filled balloon keeps its whole interior: an image-edit model sees the
    // page and may repaint a balloon it was told to keep.
    for (number, (region, inside)) in balloons.iter().zip(&interiors).enumerate() {
        if !filled_in[number + 1] {
            continue;
        }
        let [left, top, right, bottom] = region.bounds;
        for y in top..bottom {
            for x in left..right {
                if pending_mask.get_pixel(x, y)[0] < 127 && inside.contains(x, y) {
                    filled.push((x, y, *output.get_pixel(x, y)));
                }
            }
        }
    }
    filled
}

/// Over a crop, as an integral image: how many pixels `include` accepts, and
/// how many of those lie within `FLAT_FILL_INLIER_RANGE` of the `reference`
/// brightness, with their per-channel sums and squares.
struct WindowSums {
    crop: [u32; 4],
    stride: usize,
    table: Vec<[u64; 8]>,
}

impl WindowSums {
    fn new(
        image: &RgbImage,
        crop: [u32; 4],
        include: impl Fn(u32, u32) -> bool,
        reference: u8,
    ) -> Self {
        let [left, top, right, bottom] = crop;
        let stride = (right - left + 1) as usize;
        let mut table = vec![[0u64; 8]; stride * (bottom - top + 1) as usize];
        for y in top..bottom {
            let mut row = [0u64; 8];
            for x in left..right {
                if include(x, y) {
                    let pixel = image.get_pixel(x, y);
                    row[0] += 1;
                    if luma(pixel).abs_diff(reference) <= FLAT_FILL_INLIER_RANGE {
                        row[1] += 1;
                        for channel in 0..3 {
                            let value = u64::from(pixel[channel]);
                            row[2 + channel] += value;
                            row[5 + channel] += value * value;
                        }
                    }
                }
                let at = (y - top + 1) as usize * stride + (x - left + 1) as usize;
                let above = table[at - stride];
                for value in 0..8 {
                    table[at][value] = above[value] + row[value];
                }
            }
        }
        Self {
            crop,
            stride,
            table,
        }
    }

    /// How many pixels near the reference lie within `reach` of (x, y), and
    /// their mean colour when they are most of the accepted pixels there
    /// (`FLAT_FILL_MIN_INLIERS`) and every channel deviates at most
    /// `FLAT_FILL_MAX_DEVIATION`.
    fn flat_mean(&self, x: u32, y: u32, reach: u32) -> (u64, Option<Rgb<u8>>) {
        let [left, top, right, bottom] = self.crop;
        let (x0, y0) = (
            x.saturating_sub(reach).max(left),
            y.saturating_sub(reach).max(top),
        );
        let (x1, y1) = ((x + reach + 1).min(right), (y + reach + 1).min(bottom));
        let row = |y: u32| (y - top) as usize * self.stride;
        let (a, b, c, d) = (
            &self.table[row(y1) + (x1 - left) as usize],
            &self.table[row(y0) + (x0 - left) as usize],
            &self.table[row(y1) + (x0 - left) as usize],
            &self.table[row(y0) + (x1 - left) as usize],
        );
        let mut sums = [0u64; 8];
        for value in 0..8 {
            sums[value] = a[value] + b[value] - c[value] - d[value];
        }
        let (all, count) = (sums[0], sums[1]);
        if count == 0 {
            return (0, None);
        }
        let flat = count as f64 >= FLAT_FILL_MIN_INLIERS * all as f64
            && (0..3).all(|channel| {
                let mean = sums[2 + channel] as f64 / count as f64;
                sums[5 + channel] as f64 / count as f64 - mean * mean
                    <= FLAT_FILL_MAX_DEVIATION * FLAT_FILL_MAX_DEVIATION
            });
        let mean =
            Rgb([0, 1, 2].map(|channel| (sums[2 + channel] as f64 / count as f64).round() as u8));
        (count, flat.then_some(mean))
    }
}

fn luma(pixel: &Rgb<u8>) -> u8 {
    ((u32::from(pixel[0]) * 299 + u32::from(pixel[1]) * 587 + u32::from(pixel[2]) * 114) / 1000)
        as u8
}

/// The neighbourhood a lettering pixel reads its fill colour from: 6 px on a
/// 1280 px page, growing with larger pages.
fn flat_fill_radius(width: u32, height: u32) -> u32 {
    (width.max(height) / 200).max(6)
}

fn inpaint_tiles(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<InpaintTile>) {
    let (labels, components) = mask_components(mask);
    let mut bounded_tiles: Vec<InpaintTile> = Vec::new();
    let mut split_tiles = Vec::new();

    for component in components {
        let [left, top, right, bottom] = component.bounds;
        if right - left <= TILE_SIZE && bottom - top <= TILE_SIZE {
            let best = bounded_tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    let bounds = union_bounds(tile.core, component.bounds);
                    ((bounds[2] - bounds[0] <= TILE_SIZE) && (bounds[3] - bounds[1] <= TILE_SIZE))
                        .then_some((bounds_area(bounds) - bounds_area(tile.core), index, bounds))
                })
                .min_by_key(|(growth, index, _)| (*growth, *index));
            if let Some((_, index, bounds)) = best {
                let tile = &mut bounded_tiles[index];
                tile.components.push(component.label);
                tile.core = bounds;
                tile.crop = expand_bounds(bounds, mask.width(), mask.height());
            } else {
                bounded_tiles.push(InpaintTile {
                    components: vec![component.label],
                    core: component.bounds,
                    crop: expand_bounds(component.bounds, mask.width(), mask.height()),
                });
            }
            continue;
        }

        let mut core_top = top;
        while core_top < bottom {
            let core_bottom = core_top.saturating_add(TILE_SIZE).min(bottom);
            let mut core_left = left;
            while core_left < right {
                let core_right = core_left.saturating_add(TILE_SIZE).min(right);
                let core = [core_left, core_top, core_right, core_bottom];
                if let Some(owned_bounds) = component_bounds_in(&labels, component.label, core) {
                    split_tiles.push(InpaintTile {
                        components: vec![component.label],
                        core,
                        crop: expand_bounds(owned_bounds, mask.width(), mask.height()),
                    });
                }
                core_left = core_right;
            }
            core_top = core_bottom;
        }
    }

    bounded_tiles.append(&mut split_tiles);
    bounded_tiles.sort_by_key(|tile| (tile.core[1], tile.core[0], tile.core[3], tile.core[2]));
    (labels, bounded_tiles)
}

fn mask_components(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<MaskComponent>) {
    let binary = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([(mask.get_pixel(x, y)[0] >= 127) as u8 * u8::MAX])
    });
    let labels = connected_components(&binary, Connectivity::Eight, Luma([0]));
    let mut bounds = Vec::<Option<[u32; 4]>>::new();
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0] as usize;
        if label == 0 {
            continue;
        }
        if label >= bounds.len() {
            bounds.resize(label + 1, None);
        }
        if let Some(bounds) = &mut bounds[label] {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
        } else {
            bounds[label] = Some([x, y, x + 1, y + 1]);
        }
    }
    let components = bounds
        .into_iter()
        .enumerate()
        .skip(1)
        .filter_map(|(label, bounds)| {
            bounds.map(|bounds| MaskComponent {
                label: label as u32,
                bounds,
            })
        })
        .collect();
    (labels, components)
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn bounds_area([left, top, right, bottom]: [u32; 4]) -> u64 {
    u64::from(right - left) * u64::from(bottom - top)
}

fn expand_bounds([left, top, right, bottom]: [u32; 4], width: u32, height: u32) -> [u32; 4] {
    [
        left.saturating_sub(TILE_CONTEXT),
        top.saturating_sub(TILE_CONTEXT),
        right.saturating_add(TILE_CONTEXT).min(width),
        bottom.saturating_add(TILE_CONTEXT).min(height),
    ]
}

fn component_bounds_in(
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    label: u32,
    [region_left, region_top, region_right, region_bottom]: [u32; 4],
) -> Option<[u32; 4]> {
    let mut left = region_right;
    let mut top = region_bottom;
    let mut right = 0;
    let mut bottom = 0;
    for y in region_top..region_bottom {
        for x in region_left..region_right {
            if labels.get_pixel(x, y)[0] == label {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn crop_tile_mask(mask: &GrayImage, tile: &InpaintTile) -> GrayImage {
    let [left, top, right, bottom] = tile.crop;
    let mut crop = GrayImage::new(right - left, bottom - top);
    for y in top..bottom {
        for x in left..right {
            if mask.get_pixel(x, y)[0] >= 127 {
                crop.put_pixel(x - left, y - top, Luma([u8::MAX]));
            }
        }
    }
    crop
}

fn uniform_region_color(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
    inside: &PolygonInterior,
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let outline = OutlineIndex::new(&region.polygon, FLAT_FILL_EDGE_MARGIN);
    let mut interior = Vec::new();
    let mut unmasked = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if !inside.contains(x, y)
                || outline.within(point, FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN)
            {
                continue;
            }
            let pixel = *image.get_pixel(x, y);
            interior.push(pixel);
            if mask.get_pixel(x, y)[0] < 127 {
                unmasked.push(pixel);
            }
        }
    }
    if let Some(color) = flat_color(&unmasked) {
        return Some(color);
    }

    // Deliberate divergence from BallonsTranslator: a text mask can cover the
    // whole balloon, or leave only a few stray pixels unmasked, as happens with
    // white lettering on a dark balloon. The balloon fill is still the dominant
    // colour of the interior, so keep the pixels near the interior median and
    // require them to be most of the balloon and flat on their own. Textured,
    // gradient, or lettering-dense interiors still go to the model.
    let center = median_color(&interior)?;
    let inliers = interior
        .iter()
        .copied()
        .filter(|pixel| {
            (0..3).all(|channel| pixel[channel].abs_diff(center[channel]) <= MASKED_FILL_TOLERANCE)
        })
        .collect::<Vec<_>>();
    if (inliers.len() as f64) < interior.len() as f64 * MASKED_FILL_MIN_SHARE {
        return None;
    }
    flat_color(&inliers)
}

fn median_color(pixels: &[Rgb<u8>]) -> Option<Rgb<u8>> {
    if pixels.len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }
    let mut channels = std::array::from_fn::<_, 3, _>(|channel| {
        pixels
            .iter()
            .map(|pixel| pixel[channel])
            .collect::<Vec<_>>()
    });
    Some(Rgb(channels.each_mut().map(|values| median(values))))
}

fn flat_color(pixels: &[Rgb<u8>]) -> Option<Rgb<u8>> {
    let medians = median_color(pixels)?.0;
    let channels = std::array::from_fn::<_, 3, _>(|channel| {
        pixels
            .iter()
            .map(|pixel| pixel[channel])
            .collect::<Vec<_>>()
    });
    let deviations = std::array::from_fn::<_, 3, _>(|channel| {
        standard_deviation(&channels[channel], f64::from(medians[channel]))
    });
    let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
    let channel_spread = (deviations
        .iter()
        .map(|deviation| (deviation - mean_deviation).powi(2))
        .sum::<f64>()
        / deviations.len() as f64)
        .sqrt();
    let threshold = if channel_spread > 1.0 { 7.0 } else { 10.0 };
    (deviations.iter().copied().fold(0.0, f64::max) < threshold).then_some(Rgb(medians))
}

#[cfg(test)]
fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        if (current.1 > point.1) != (previous.1 > point.1) {
            let intersection_x = (previous.0 - current.0) * (point.1 - current.1)
                / (previous.1 - current.1)
                + current.0;
            if point.0 < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

#[cfg(test)]
fn polygon_edge_distance_squared(point: (f32, f32), polygon: &[(f32, f32)]) -> f32 {
    let mut minimum = f32::INFINITY;
    let mut start = polygon[polygon.len() - 1];
    for &end in polygon {
        minimum = minimum.min(segment_distance_squared(point, start, end));
        start = end;
    }
    minimum
}

fn segment_distance_squared(point: (f32, f32), start: (f32, f32), end: (f32, f32)) -> f32 {
    let segment = (end.0 - start.0, end.1 - start.1);
    let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
    let projection = if length_squared > 0.0 {
        ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    let closest = (
        start.0 + segment.0 * projection,
        start.1 + segment.1 * projection,
    );
    let dx = point.0 - closest.0;
    let dy = point.1 - closest.1;
    dx * dx + dy * dy
}

/// The pixel centres of a region's bounds that lie inside its polygon, by the
/// even-odd rule of `point_in_polygon`. Each row's outline crossings are found
/// once instead of once per pixel, which gives the same answer per pixel at a
/// fraction of the cost on traced outlines with hundreds of points.
struct PolygonInterior {
    bounds: [u32; 4],
    inside: Vec<bool>,
}

impl PolygonInterior {
    fn contains(&self, x: u32, y: u32) -> bool {
        let [left, top, right, _] = self.bounds;
        self.inside[(y - top) as usize * (right - left) as usize + (x - left) as usize]
    }
}

fn polygon_interior(region: &FlatFillRegion) -> PolygonInterior {
    let [left, top, right, bottom] = region.bounds;
    let width = (right - left) as usize;
    let mut inside = vec![false; width * (bottom - top) as usize];
    let mut crossings = Vec::new();
    for y in top..bottom {
        let row_y = y as f32 + 0.5;
        crossings.clear();
        let mut previous = region.polygon[region.polygon.len() - 1];
        for &current in &region.polygon {
            if (current.1 > row_y) != (previous.1 > row_y) {
                crossings.push(
                    (previous.0 - current.0) * (row_y - current.1) / (previous.1 - current.1)
                        + current.0,
                );
            }
            previous = current;
        }
        crossings.sort_by(f32::total_cmp);
        let row = &mut inside[(y - top) as usize * width..][..width];
        // Crossings at or left of the pixel centre; the ones right of it decide.
        let mut passed = 0;
        for (x, cell) in (left..right).zip(row) {
            let column_x = x as f32 + 0.5;
            while passed < crossings.len() && crossings[passed] <= column_x {
                passed += 1;
            }
            *cell = (crossings.len() - passed) % 2 == 1;
        }
    }
    PolygonInterior {
        bounds: region.bounds,
        inside,
    }
}

/// An outline's segments bucketed on a coarse grid, so a test of whether a
/// point lies within some distance of the outline measures only the segments
/// near it. A segment farther than `reach` from a cell is never listed for it,
/// and the distances that are measured are those of the full scan.
struct OutlineIndex<'a> {
    polygon: &'a [(f32, f32)],
    origin: (f32, f32),
    columns: usize,
    rows: usize,
    cells: Vec<Vec<u32>>,
}

const OUTLINE_CELL: f32 = 16.0;

impl<'a> OutlineIndex<'a> {
    fn new(polygon: &'a [(f32, f32)], reach: f32) -> Self {
        // One extra pixel keeps rounding at the margin on the listed side.
        let reach = reach + 1.0;
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let origin = (left - reach, top - reach);
        let columns = ((right + reach - origin.0) / OUTLINE_CELL).floor() as usize + 1;
        let rows = ((bottom + reach - origin.1) / OUTLINE_CELL).floor() as usize + 1;
        let mut cells = vec![Vec::new(); columns * rows];
        let cell = |value: f32, start: f32, count: usize| {
            (((value - start) / OUTLINE_CELL).floor().max(0.0) as usize).min(count - 1)
        };
        let mut start = polygon[polygon.len() - 1];
        for (index, &end) in polygon.iter().enumerate() {
            let (first_column, last_column) = (
                cell(start.0.min(end.0) - reach, origin.0, columns),
                cell(start.0.max(end.0) + reach, origin.0, columns),
            );
            let (first_row, last_row) = (
                cell(start.1.min(end.1) - reach, origin.1, rows),
                cell(start.1.max(end.1) + reach, origin.1, rows),
            );
            for row in first_row..=last_row {
                for column in first_column..=last_column {
                    cells[row * columns + column].push(index as u32);
                }
            }
            start = end;
        }
        Self {
            polygon,
            origin,
            columns,
            rows,
            cells,
        }
    }

    /// Whether the outline comes closer to `point` than the square root of
    /// `limit_squared`, which must not exceed the reach the index was built for.
    fn within(&self, point: (f32, f32), limit_squared: f32) -> bool {
        let column = ((point.0 - self.origin.0) / OUTLINE_CELL).floor();
        let row = ((point.1 - self.origin.1) / OUTLINE_CELL).floor();
        if column < 0.0 || row < 0.0 {
            return false;
        }
        let (column, row) = (column as usize, row as usize);
        if column >= self.columns || row >= self.rows {
            return false;
        }
        let count = self.polygon.len();
        self.cells[row * self.columns + column]
            .iter()
            .any(|&index| {
                let index = index as usize;
                let start = self.polygon[(index + count - 1) % count];
                segment_distance_squared(point, start, self.polygon[index]) < limit_squared
            })
    }
}

fn median(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
    } else {
        values[middle]
    }
}

fn standard_deviation(values: &[u8], center: f64) -> f64 {
    (values
        .iter()
        .map(|value| (f64::from(*value) - center).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn composite_generated(
    output: &mut RgbImage,
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    tile: &InpaintTile,
    generated: &RgbImage,
) {
    let [left, top, _, _] = tile.crop;
    let [core_left, core_top, core_right, core_bottom] = tile.core;
    for y in core_top..core_bottom {
        for x in core_left..core_right {
            if tile
                .components
                .binary_search(&labels.get_pixel(x, y)[0])
                .is_ok()
            {
                output.put_pixel(x, y, *generated.get_pixel(x - left, y - top));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Times balloon fill on every page of a real project and prints a digest
    /// of each committed layer's pixels, so a change can be checked to leave
    /// the output identical. Runs on a temporary copy of the project:
    ///
    /// `KOHARU_BENCH_PROJECT=<project.khrproj> cargo test --release -p koharu-pipeline balloon_fill_benchmark -- --ignored --nocapture`
    /// The row scan and the outline index give the per-pixel answers of the
    /// full scans they replace, on outlines like traced balloons: many points,
    /// fractional coordinates, repeated points and flat runs.
    #[test]
    fn region_scans_match_the_per_pixel_tests() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        for shape in 0..40 {
            let points = [3, 4, 7, 24, 120, 400][shape % 6];
            let center = (60.0 + random() * 20.0, 50.0 + random() * 20.0);
            let mut polygon = (0..points)
                .map(|step| {
                    let angle = step as f32 / points as f32 * std::f32::consts::TAU;
                    let radius = 15.0 + random() * 30.0 * if shape % 2 == 0 { 1.0 } else { 0.2 };
                    let point = (
                        center.0 + radius * angle.cos(),
                        center.1 + radius * angle.sin(),
                    );
                    if shape % 3 == 0 {
                        (point.0.round(), point.1.round())
                    } else {
                        point
                    }
                })
                .collect::<Vec<_>>();
            if shape % 5 == 0 {
                let repeated = polygon[1];
                polygon.insert(1, repeated);
            }
            let (width, height) = (140, 120);
            let mut left_top = (f32::INFINITY, f32::INFINITY);
            let mut right_bottom = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &(x, y) in &polygon {
                left_top = (left_top.0.min(x), left_top.1.min(y));
                right_bottom = (right_bottom.0.max(x), right_bottom.1.max(y));
            }
            let region = FlatFillRegion {
                bounds: [
                    left_top.0.floor().clamp(0.0, width as f32) as u32,
                    left_top.1.floor().clamp(0.0, height as f32) as u32,
                    right_bottom.0.ceil().clamp(0.0, width as f32) as u32,
                    right_bottom.1.ceil().clamp(0.0, height as f32) as u32,
                ],
                polygon,
            };
            let inside = polygon_interior(&region);
            let [left, top, right, bottom] = region.bounds;
            for y in top..bottom {
                for x in left..right {
                    let point = (x as f32 + 0.5, y as f32 + 0.5);
                    assert_eq!(
                        inside.contains(x, y),
                        point_in_polygon(point, &region.polygon),
                        "shape {shape} at {x},{y}"
                    );
                }
            }
            for radius in [FLAT_FILL_EDGE_MARGIN, 4.0, 4.3, 8.0] {
                let outline = OutlineIndex::new(&region.polygon, radius);
                for y in 0..height {
                    for x in 0..width {
                        let point = (x as f32 + 0.5, y as f32 + 0.5);
                        assert_eq!(
                            outline.within(point, radius * radius),
                            polygon_edge_distance_squared(point, &region.polygon) < radius * radius,
                            "shape {shape} radius {radius} at {x},{y}"
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "needs a real project in KOHARU_BENCH_PROJECT"]
    async fn cleaning_benchmark() {
        // KOHARU_BENCH_MODEL picks `balloon-fill` (default) or `lama`;
        // KOHARU_BENCH_PAGES limits it to some 1-based page numbers;
        // KOHARU_BENCH_THREADS sets libtorch's thread count; with
        // KOHARU_BENCH_OUT each page's cleanup layer is saved there, and with
        // KOHARU_BENCH_REFERENCE it is compared to an earlier run's.

        use std::{
            hash::{DefaultHasher, Hash as _, Hasher as _},
            time::Instant,
        };

        fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                let target = to.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_dir(&entry.path(), &target);
                } else if entry.file_name() != "project.lock" {
                    std::fs::copy(entry.path(), target).unwrap();
                }
            }
        }

        let source = std::env::var("KOHARU_BENCH_PROJECT").expect("KOHARU_BENCH_PROJECT");
        let copy = std::env::temp_dir().join(format!("koharu-fill-bench-{}", std::process::id()));
        copy_dir(std::path::Path::new(&source), &copy);
        let mut session = koharu_scene::Session::open(&copy).await.unwrap();
        let pages = session
            .snapshot()
            .pages()
            .map(|page| page.id())
            .collect::<Vec<_>>();
        koharu_ml::init().await.unwrap();
        let model_name = std::env::var("KOHARU_BENCH_MODEL").unwrap_or("balloon-fill".into());
        let model = match model_name.as_str() {
            "lama" => Model::load(koharu_ml::Device::cpu(), &InpaintingModel::LaMa {})
                .await
                .unwrap(),
            _ => Model::BalloonFill,
        };
        if let Ok(threads) = std::env::var("KOHARU_BENCH_THREADS") {
            koharu_torch::set_num_threads(threads.parse().unwrap());
        }
        println!(
            "{model_name}, {} libtorch threads",
            koharu_torch::get_num_threads()
        );
        let output = std::env::var("KOHARU_BENCH_OUT")
            .ok()
            .map(std::path::PathBuf::from);
        if let Some(output) = &output {
            std::fs::create_dir_all(output).unwrap();
        }
        let reference = std::env::var("KOHARU_BENCH_REFERENCE")
            .ok()
            .map(std::path::PathBuf::from);
        let (mut changed_total, mut worst_total) = (0_u64, 0_u8);
        let images = Arc::new(crate::ImageCache::default());
        let (mut prepare_total, mut run_total, mut digest) = (0.0, 0.0, DefaultHasher::new());
        for (index, &page) in pages.iter().enumerate() {
            // KOHARU_BENCH_PAGES limits the run to some 1-based page numbers.
            if std::env::var("KOHARU_BENCH_PAGES").is_ok_and(|pages| {
                !pages
                    .split(',')
                    .any(|number| number.trim() == (index + 1).to_string())
            }) {
                continue;
            }
            // Each page's commit moves the head, so every page reads the latest.
            let base = session.snapshot();
            let input =
                || StageInput::new(base.clone(), page, None, None, Arc::clone(&images), None);
            // Warms the image cache, so both timings below leave decoding out.
            let started = Instant::now();
            prepare(&input(), false, false, false).await.unwrap();
            let first = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let prepared = prepare(&input(), false, false, false).await.unwrap();
            if let Some(output) = &output {
                prepared
                    .original
                    .save(output.join(format!("page-{:02}-base.png", index + 1)))
                    .unwrap();
                // What an image-edit model would receive for this page.
                prepare(&input(), true, false, false)
                    .await
                    .unwrap()
                    .image
                    .save(output.join(format!("page-{:02}-sent.png", index + 1)))
                    .unwrap();
            }
            let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let tiles = inpaint_tiles(&prepared.unfilled_mask).1;
            let tile_pixels = tiles.iter().map(|tile| bounds_area(tile.crop)).sum::<u64>();
            let patch = model.run(input()).await.unwrap();
            let run_ms = started.elapsed().as_secs_f64() * 1000.0;
            prepare_total += prepare_ms;
            run_total += run_ms;
            let committed = session.commit(patch).await.unwrap().snapshot;
            let mut page_digest = DefaultHasher::new();
            prepared.flat_fill.len().hash(&mut page_digest);
            for entity in committed.descendants(page).unwrap() {
                let Some(layer) = committed.component::<RasterLayer>(entity.id()).unwrap() else {
                    continue;
                };
                if layer.kind != RasterLayerKind::Cleanup {
                    continue;
                }
                let asset = committed
                    .asset(entity.id(), &AssetRole::new("source").unwrap())
                    .unwrap()
                    .unwrap();
                let bytes = committed.read_blob(asset.blob).await.unwrap();
                let layer = image::load_from_memory(&bytes).unwrap().to_rgba8();
                layer.as_raw().hash(&mut page_digest);
                // Earlier cleanup layers, such as a cloud run's, are only hashed.
                let earlier = base
                    .asset(entity.id(), &AssetRole::new("source").unwrap())
                    .ok()
                    .flatten();
                if earlier.is_some_and(|earlier| earlier.blob == asset.blob) {
                    continue;
                }
                let name = format!("page-{:02}.png", index + 1);
                if let Some(output) = &output {
                    layer.save(output.join(&name)).unwrap();
                }
                if let Some(reference) = &reference {
                    let earlier = image::open(reference.join(&name)).unwrap().to_rgba8();
                    let (mut changed, mut worst) = (0_u64, 0_u8);
                    for (now, then) in layer.as_raw().iter().zip(earlier.as_raw()) {
                        let difference = now.abs_diff(*then);
                        changed += u64::from(difference > 2);
                        worst = worst.max(difference);
                    }
                    println!(
                        "  vs reference: {changed} channel values off by more than 2, worst {worst}"
                    );
                    changed_total += changed;
                    worst_total = worst_total.max(worst);
                }
            }
            let page_digest = page_digest.finish();
            page_digest.hash(&mut digest);
            println!(
                "page {:>2}: first prepare {first:>7.1} ms, prepare {prepare_ms:>7.1} ms, \
                 {model_name} {run_ms:>7.1} ms, {} filled pixels, {} tiles of {:.2} Mpx, \
                 digest {page_digest:016x}",
                index + 1,
                prepared.flat_fill.len(),
                tiles.len(),
                tile_pixels as f64 / 1e6,
            );
        }
        println!(
            "{} pages: prepare {prepare_total:.0} ms, {model_name} {run_total:.0} ms, digest {:016x}",
            pages.len(),
            digest.finish()
        );
        if reference.is_some() {
            println!(
                "vs reference: {changed_total} channel values off by more than 2, worst {worst_total}"
            );
        }
        drop(session);
        let _ = std::fs::remove_dir_all(&copy);
    }

    #[tokio::test]
    async fn transient_inpainting_mask_replaces_persistent_page_masks() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        let persistent = DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, Luma([255])));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in [("source", &source), ("text-mask", &persistent)] {
                    edit.set_asset(
                        id,
                        &AssetRole::new(role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let mut transient = GrayImage::new(8, 8);
        transient.put_pixel(3, 4, Luma([255]));
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(transient)),
                intent: crate::InpaintingIntent::Remove,
            }),
        );

        let prepared = prepare(&input, false, false, false).await.unwrap();
        assert_eq!(prepared.mask.get_pixel(3, 4), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert_eq!(prepared.unfilled_mask, prepared.mask);
        assert!(prepared.flat_fill.is_empty());
    }

    #[tokio::test]
    async fn api_cleanup_composites_visible_paint_without_baking_it_into_cleanup() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let source = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255])));
        let mut visible_paint = RgbaImage::new(2, 1);
        visible_paint.put_pixel(0, 0, Rgba([200, 0, 0, 255]));
        let mut hidden_paint = RgbaImage::new(2, 1);
        hidden_paint.put_pixel(1, 0, Rgba([0, 200, 0, 255]));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 2.0, 1.0),
                    koharu_scene::At::End,
                )?;
                edit.set_asset(
                    id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        encode(&source),
                        "image/png",
                        AssetMetadata {
                            width: Some(2),
                            height: Some(1),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                for (image, visible) in [(&visible_paint, true), (&hidden_paint, false)] {
                    let layer = edit.add_entity(id, At::End)?;
                    edit.set(
                        layer,
                        &RasterLayer {
                            origin: Origin::User,
                            name: "Paint".to_owned(),
                            kind: RasterLayerKind::Paint,
                        },
                    )?;
                    edit.set(
                        layer,
                        &koharu_scene::Visibility {
                            origin: Origin::User,
                            visible,
                            opacity: 1.0,
                        },
                    )?;
                    edit.set_asset(
                        layer,
                        &AssetRole::new("source")?,
                        AssetInput::new(
                            encode(&DynamicImage::ImageRgba8(image.clone())),
                            "image/png",
                            AssetMetadata {
                                width: Some(2),
                                height: Some(1),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let input = StageInput::new(
            snapshot,
            page.unwrap(),
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );

        let prepared = prepare(&input, true, false, false).await.unwrap();
        assert_eq!(
            prepared.image.to_rgba8().get_pixel(0, 0),
            &Rgba([200, 0, 0, 255])
        );
        assert_eq!(
            prepared.image.to_rgba8().get_pixel(1, 0),
            &Rgba([10, 20, 30, 255])
        );

        let mut cleanup = RgbaImage::from_pixel(2, 1, Rgba([1, 2, 3, 255]));
        exclude_paint_from_cleanup(&mut cleanup, prepared.paint.as_ref());
        assert_eq!(cleanup.get_pixel(0, 0), &Rgba([1, 2, 3, 0]));
        assert_eq!(cleanup.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));

        let without_paint = prepare(&input, false, false, false).await.unwrap();
        assert_eq!(
            without_paint.image.to_rgba8().get_pixel(0, 0),
            &Rgba([10, 20, 30, 255])
        );
        assert!(without_paint.paint.is_none());
    }

    #[tokio::test]
    async fn restore_composites_visible_cleanup_and_paint_in_layer_order_with_opacity() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let source = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255])));
        let mut cleanup = RgbaImage::new(2, 1);
        cleanup.put_pixel(0, 0, Rgba([200, 0, 0, 255]));
        let mut paint = RgbaImage::new(2, 1);
        paint.put_pixel(0, 0, Rgba([0, 0, 200, 255]));
        let mut hidden_cleanup = RgbaImage::new(2, 1);
        hidden_cleanup.put_pixel(1, 0, Rgba([0, 200, 0, 255]));
        let mut hidden_paint = RgbaImage::new(2, 1);
        hidden_paint.put_pixel(1, 0, Rgba([200, 200, 0, 255]));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 2.0, 1.0),
                    koharu_scene::At::End,
                )?;
                edit.set_asset(
                    id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        encode(&source),
                        "image/png",
                        AssetMetadata {
                            width: Some(2),
                            height: Some(1),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                for (image, kind, visible) in [
                    (&cleanup, RasterLayerKind::Cleanup, true),
                    (&paint, RasterLayerKind::Paint, true),
                    (&hidden_cleanup, RasterLayerKind::Cleanup, false),
                    (&hidden_paint, RasterLayerKind::Paint, false),
                ] {
                    let layer = edit.add_entity(id, At::End)?;
                    edit.set(
                        layer,
                        &RasterLayer {
                            origin: Origin::User,
                            name: "Layer".to_owned(),
                            kind,
                        },
                    )?;
                    edit.set(
                        layer,
                        &Visibility {
                            origin: Origin::User,
                            visible,
                            opacity: 0.5,
                        },
                    )?;
                    edit.set_asset(
                        layer,
                        &AssetRole::new("source")?,
                        AssetInput::new(
                            encode(&DynamicImage::ImageRgba8(image.clone())),
                            "image/png",
                            AssetMetadata {
                                width: Some(2),
                                height: Some(1),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let input = StageInput::new(
            snapshot,
            page.unwrap(),
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );

        let prepared = prepare(&input, true, true, false).await.unwrap();
        assert_eq!(
            prepared.image.to_rgba8().get_pixel(0, 0),
            &Rgba([52, 4, 107, 254])
        );
        assert_eq!(
            prepared.image.to_rgba8().get_pixel(1, 0),
            &Rgba([10, 20, 30, 255])
        );
        assert_eq!(prepared.paint.as_ref().unwrap().get_pixel(0, 0)[3], 128);
        assert_eq!(prepared.paint.as_ref().unwrap().get_pixel(1, 0)[3], 0);
    }

    #[tokio::test]
    async fn restoration_uses_exact_mask_crop_and_page_coordinates() {
        let mut mask = GrayImage::new(6, 5);
        for y in 1..4 {
            for x in 2..5 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        mask.put_pixel(3, 2, Luma([0]));
        let bounds = mask_bounds(&mask).unwrap();
        assert_eq!(bounds, [2, 1, 5, 4]);

        let source = DynamicImage::ImageRgba8(RgbaImage::from_fn(6, 5, |x, y| {
            Rgba([x as u8, y as u8, 0, 255])
        }));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(koharu_scene::PageDraft::new("page", 6.0, 5.0), At::End)?;
                edit.set_asset(
                    id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        encode(&source),
                        "image/png",
                        AssetMetadata {
                            width: Some(6),
                            height: Some(5),
                            attributes: BTreeMap::new(),
                        },
                    ),
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
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(mask.clone())),
                intent: InpaintingIntent::Restore,
            }),
        );
        let prepared = prepare(&input, true, true, false).await.unwrap();
        let crop = &prepared.image;
        assert_eq!(crop.dimensions(), (3, 3));
        assert_eq!(crop.get_pixel(0, 0), Rgba([2, 1, 0, 255]));
        assert_eq!(crop.get_pixel(2, 2), Rgba([4, 3, 0, 255]));
        assert_eq!(prepared.original.dimensions(), (6, 5));
        assert_eq!(prepared.mask.dimensions(), (6, 5));

        let response = RgbaImage::from_pixel(3, 3, Rgba([9, 8, 7, 255]));
        let cleanup = cleanup_overlay(&response, &mask, false, prepared.restore_bounds);
        assert_eq!(cleanup.get_pixel(1, 1), &Rgba([0, 0, 0, 0]));
        assert_eq!(cleanup.get_pixel(2, 1), &Rgba([9, 8, 7, 255]));
        assert_eq!(cleanup.get_pixel(3, 2), &Rgba([0, 0, 0, 0]));
        assert_eq!(cleanup.get_pixel(4, 3), &Rgba([9, 8, 7, 255]));
        assert_eq!(cleanup.get_pixel(5, 4), &Rgba([0, 0, 0, 0]));
    }

    #[tokio::test]
    async fn repeated_restoration_layers_stack_above_cleanup_and_below_paint() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let mut old_cleanup = None;
        let mut paint = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 2.0, 1.0),
                    koharu_scene::At::End,
                )?;
                let cleanup = edit.add_entity(id, At::End)?;
                edit.set(
                    cleanup,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                let paint_layer = edit.add_entity(id, At::End)?;
                edit.set(
                    paint_layer,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Paint".to_owned(),
                        kind: RasterLayerKind::Paint,
                    },
                )?;
                page = Some(id);
                old_cleanup = Some(cleanup);
                paint = Some(paint_layer);
                Ok(())
            })
            .unwrap();
        session.commit(patch).await.unwrap();
        let page = page.unwrap();
        let mut restorations = Vec::new();
        for _ in 0..2 {
            let snapshot = session.snapshot();
            let at = restoration_layer_placement(&snapshot, page).unwrap();
            let mut restoration = None;
            let patch = snapshot
                .patch(|edit| {
                    let layer = edit.add_entity(page, at)?;
                    edit.set(
                        layer,
                        &RasterLayer {
                            origin: Origin::User,
                            name: "Cleanup".to_owned(),
                            kind: RasterLayerKind::Cleanup,
                        },
                    )?;
                    restoration = Some(layer);
                    Ok(())
                })
                .unwrap();
            session.commit(patch).await.unwrap();
            restorations.push(restoration.unwrap());
        }

        assert_ne!(restorations[0], restorations[1]);
        assert_eq!(
            session
                .snapshot()
                .children(page)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![
                old_cleanup.unwrap(),
                restorations[0],
                restorations[1],
                paint.unwrap(),
            ]
        );
    }

    #[test]
    fn automatic_cleanup_protection_does_not_mutate_the_exact_restoration_prompt() {
        let restoration = ImageEditConfig {
            prompt: "  exact restoration prompt  ".to_owned(),
            apply_mode: InpaintingApplyMode::Mask,
        };

        let automatic = protected_cleanup_config(&restoration);

        assert_eq!(restoration.prompt, "  exact restoration prompt  ");
        assert_eq!(
            automatic.prompt,
            format!("exact restoration prompt\n\nSafety boundary: {DEFAULT_CLEANUP_PROMPT}")
        );
    }

    #[tokio::test]
    async fn cleaning_can_run_again_after_a_committed_result() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        let cleanup_image = DynamicImage::new_rgba8(8, 8);
        let mask = DynamicImage::ImageLuma8(GrayImage::new(8, 8));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                edit.set_asset(
                    id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        encode(&source),
                        "image/png",
                        AssetMetadata {
                            width: Some(8),
                            height: Some(8),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                let cleanup_id = edit.add_entity(id, At::Start)?;
                edit.set(
                    cleanup_id,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup_id,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        encode(&cleanup_image),
                        "image/png",
                        AssetMetadata {
                            width: Some(8),
                            height: Some(8),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let automatic = StageInput::new(
            snapshot.clone(),
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );
        let manual = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&mask),
                intent: crate::InpaintingIntent::Remove,
            }),
        );
        let processor = Processor::new(InpaintingModel::LaMa {}, koharu_ml::Device::cpu()).unwrap();

        assert!(!processor.skip(&automatic).unwrap());
        assert!(!processor.skip(&manual).unwrap());
        assert_eq!(
            prepare(&automatic, false, false, false)
                .await
                .unwrap()
                .image
                .dimensions(),
            (8, 8)
        );
        assert_eq!(
            prepare(&manual, false, false, false)
                .await
                .unwrap()
                .image
                .dimensions(),
            (8, 8)
        );
    }

    #[test]
    fn lettering_the_cloud_left_untouched_is_found() {
        // Grey page with dark strokes in a 20x20 box: the "lettering".
        let lettering =
            |x: u32, y: u32| (10..30).contains(&x) && (10..30).contains(&y) && x % 4 < 2;
        let sent = GrayImage::from_fn(60, 40, |x, y| {
            Luma([if lettering(x, y) { 20 } else { 200 }])
        });
        let boxes = [[10, 10, 30, 30], [40, 10, 55, 30]];

        // Left as is, and slightly lighter (85% of the contrast): residual.
        assert_eq!(residual_boxes(&sent, &sent, &boxes), vec![[10, 10, 30, 30]]);
        let ghost = GrayImage::from_fn(60, 40, |x, y| {
            Luma([if lettering(x, y) { 47 } else { 200 }])
        });
        assert_eq!(
            residual_boxes(&sent, &ghost, &boxes),
            vec![[10, 10, 30, 30]]
        );
        // Faded to half the contrast: left for the user, no second request.
        let faded = GrayImage::from_fn(60, 40, |x, y| {
            Luma([if lettering(x, y) { 110 } else { 200 }])
        });
        assert!(residual_boxes(&sent, &faded, &boxes).is_empty());
        // Redrawn as flat background: clean. The second box had no lettering.
        let clean = GrayImage::from_pixel(60, 40, Luma([200]));
        assert!(residual_boxes(&sent, &clean, &boxes).is_empty());
    }

    #[tokio::test]
    async fn the_topmost_visible_cleanup_layer_decides_whether_text_was_left() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut ids = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let page =
                    edit.add_page(koharu_scene::PageDraft::new("page", 8.0, 8.0), At::End)?;
                let region = edit.add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(1.0, 1.0, 4.0, 4.0),
                    None,
                )?;
                let mut layers = Vec::new();
                for residual in [region.to_string(), String::new()] {
                    let layer = edit.add_entity(page, At::End)?;
                    edit.set(
                        layer,
                        &RasterLayer {
                            origin: Origin::User,
                            name: "Cleanup".to_owned(),
                            kind: RasterLayerKind::Cleanup,
                        },
                    )?;
                    let mut bytes = Cursor::new(Vec::new());
                    DynamicImage::new_rgba8(8, 8)
                        .write_to(&mut bytes, ImageFormat::Png)
                        .unwrap();
                    let mut attributes = BTreeMap::new();
                    if !residual.is_empty() {
                        attributes.insert(RESIDUAL_TEXT_ATTRIBUTE.to_owned(), residual);
                    }
                    edit.set_asset(
                        layer,
                        &AssetRole::new("source")?,
                        AssetInput::new(
                            Arc::<[u8]>::from(bytes.into_inner()),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes,
                            },
                        ),
                    )?;
                    layers.push(layer);
                }
                ids = Some((region, layers));
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let (region, layers) = ids.unwrap();
        // A later cleanup on top (no residual) clears the flag.
        assert!(!cleaning_unverified(&snapshot, region).unwrap());

        let hidden = snapshot
            .patch(|edit| {
                edit.set(
                    layers[1],
                    &Visibility {
                        origin: Origin::User,
                        visible: false,
                        opacity: 1.0,
                    },
                )
            })
            .unwrap();
        let snapshot = session.commit(hidden).await.unwrap().snapshot;
        // With it hidden, the cleanup that left the text is on top again.
        assert!(cleaning_unverified(&snapshot, region).unwrap());
    }

    #[test]
    fn fal_cleanup_respects_full_page_automatic_and_manual_masks() {
        let generated = RgbaImage::from_pixel(2, 1, Rgba([1, 2, 3, 40]));
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(1, 0, Luma([255]));

        let full = cleanup_overlay(&generated, &mask, true, None);
        assert_eq!(full.get_pixel(0, 0), &Rgba([1, 2, 3, 255]));
        assert_eq!(full.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));

        let masked = cleanup_overlay(&generated, &mask, false, None);
        assert_eq!(masked.get_pixel(0, 0), &Rgba([0, 0, 0, 0]));
        assert_eq!(masked.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));
    }

    fn rectangle_region([left, top, right, bottom]: [u32; 4]) -> FlatFillRegion {
        FlatFillRegion {
            bounds: [left, top, right, bottom],
            polygon: vec![
                (left as f32, top as f32),
                (right as f32, top as f32),
                (right as f32, bottom as f32),
                (left as f32, bottom as f32),
            ],
        }
    }

    #[test]
    fn uniform_mask_background_is_filled_without_inference() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([240, 241, 242]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;
        let mut pending = mask.clone();
        fill_flat_lettering(
            &mut image,
            &mut pending,
            &mask,
            &[rectangle_region([0, 0, 96, 96])],
        );

        let output = inpaint_tiled(&DynamicImage::ImageRgb8(image), &pending, |_, _| {
            calls += 1;
            Ok(DynamicImage::new_rgb8(1, 1))
        })
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(48, 48), &Rgb([240, 241, 242]));
    }

    #[tokio::test]
    async fn balloon_fill_commits_only_the_filled_balloons() {
        let encode = |image: DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        // A white balloon on the left, a gradient balloon on the right, both
        // with dark lettering, and a mark between them outside any balloon.
        let mut text = GrayImage::new(40, 20);
        for left in [6, 26] {
            for y in 8..12 {
                for x in left..left + 4 {
                    text.put_pixel(x, y, Luma([255]));
                }
            }
        }
        text.put_pixel(19, 10, Luma([255]));
        let source = RgbImage::from_fn(40, 20, |x, y| {
            if text.get_pixel(x, y)[0] > 0 {
                Rgb([10, 10, 10])
            } else if x >= 20 {
                let value = (150 + (x - 20) * 5) as u8;
                Rgb([value, value, value])
            } else {
                Rgb([255, 255, 255])
            }
        });
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id =
                    edit.add_page(koharu_scene::PageDraft::new("page", 40.0, 20.0), At::End)?;
                for (role, image) in [
                    ("source", DynamicImage::ImageRgb8(source.clone())),
                    ("text-mask", DynamicImage::ImageLuma8(text.clone())),
                ] {
                    edit.set_asset(
                        id,
                        &AssetRole::new(role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(40),
                                height: Some(20),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                for left in [1.0, 21.0] {
                    edit.add_analysis_region::<BubbleRegion>(
                        id,
                        At::End,
                        &Geometry::rectangle(left, 1.0, 17.0, 18.0),
                        None,
                    )?;
                }
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
            Arc::new(crate::ImageCache::default()),
            None,
        );

        let patch = Model::BalloonFill.run(input).await.unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let layer = snapshot
            .descendants(page)
            .unwrap()
            .map(|entity| entity.id())
            .find(|id| {
                snapshot
                    .component::<RasterLayer>(*id)
                    .unwrap()
                    .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
            })
            .expect("a cleanup layer");
        let asset = snapshot
            .asset(layer, &AssetRole::new("source").unwrap())
            .unwrap()
            .unwrap();
        let cleanup =
            image::load_from_memory(&snapshot.read_blob(asset.blob).await.unwrap()).unwrap();
        let cleanup = cleanup.to_rgba8();
        assert_eq!(cleanup.get_pixel(7, 9), &Rgba([255, 255, 255, 255]));
        // The gradient balloon and the text outside balloons stay for the model.
        assert_eq!(cleanup.get_pixel(27, 9)[3], 0);
        assert_eq!(cleanup.get_pixel(19, 10)[3], 0);
    }

    #[tokio::test]
    async fn automatic_cleaning_leaves_a_page_marked_as_cleaned() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(koharu_scene::PageDraft::new("page", 8.0, 8.0), At::End)?;
                edit.set(
                    id,
                    &koharu_scene::CleanupDone {
                        origin: Origin::User,
                    },
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let processor =
            Processor::new(InpaintingModel::BalloonFill {}, koharu_ml::Device::cpu()).unwrap();
        let input = StageInput::new(
            snapshot.clone(),
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );
        assert!(processor.skip(&input).unwrap());
        assert!(cleaned(&snapshot, page).unwrap());
    }

    /// Cuts lettering-shaped holes into screentone on a real page, fills them
    /// with LaMa, then with LaMa and rescreening, and compares both with the
    /// page: mean error and the correlation of fine detail (dots in place).
    ///
    /// `KOHARU_SCREENTONE_PAGE=<image> KOHARU_SCREENTONE_WIDTH=1280
    /// KOHARU_SCREENTONE_HOLES="x,y;x,y" KOHARU_SCREENTONE_OUT=<dir>
    /// cargo test -p koharu-pipeline screentone_benchmark -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "needs a page image and the LaMa weights"]
    async fn screentone_benchmark() {
        let page = image::open(std::env::var("KOHARU_SCREENTONE_PAGE").unwrap()).unwrap();
        let width = std::env::var("KOHARU_SCREENTONE_WIDTH")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(page.width());
        let page = page
            .resize(width, u32::MAX, image::imageops::FilterType::Lanczos3)
            .to_rgb8();
        let holes = std::env::var("KOHARU_SCREENTONE_HOLES")
            .unwrap()
            .split(';')
            .map(|hole| {
                let (x, y) = hole.split_once(',').unwrap();
                (x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap())
            })
            .collect::<Vec<_>>();
        let output = std::path::PathBuf::from(std::env::var("KOHARU_SCREENTONE_OUT").unwrap());
        std::fs::create_dir_all(&output).unwrap();
        // Three strokes of a character, 4 px wide, in a 36 x 28 box, times
        // KOHARU_SCREENTONE_SCALE.
        let scale = std::env::var("KOHARU_SCREENTONE_SCALE")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(1);
        let mut mask = GrayImage::new(page.width(), page.height());
        for &(left, top) in &holes {
            for y in 0..28 * scale {
                for x in 0..36 * scale {
                    let (x0, y0) = (x / scale, y / scale);
                    let stroke =
                        (4..8).contains(&y0) || (12..16).contains(&x0) || (20..24).contains(&y0);
                    if stroke {
                        mask.put_pixel(left + x, top + y, Luma([255]));
                    }
                }
            }
        }
        let mut damaged = page.clone();
        for (x, y, value) in mask.enumerate_pixels() {
            if value[0] > 0 {
                damaged.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        koharu_ml::init().await.unwrap();
        let Model::LaMa(model) = Model::load(koharu_ml::Device::cpu(), &InpaintingModel::LaMa {})
            .await
            .unwrap()
        else {
            unreachable!("LaMa loads as LaMa")
        };
        let lama = inpaint_tiled(
            &DynamicImage::ImageRgb8(damaged.clone()),
            &mask,
            |image, mask| {
                Ok(DynamicImage::ImageRgb8(model.lock().unwrap().inference(
                    image,
                    mask,
                    &InpaintRequest::default(),
                )?))
            },
        )
        .unwrap()
        .to_rgb8();
        let mut rescreened = lama.clone();
        let redrawn = screentone::rescreen(&damaged, &mut rescreened, &mask);
        let luma = |image: &RgbImage, x: u32, y: u32| f64::from(image.get_pixel(x, y)[0]);
        let detail = |image: &RgbImage, x: u32, y: u32| {
            let mut sum = 0.0;
            for dy in 0..5 {
                for dx in 0..5 {
                    sum += luma(image, x + dx - 2, y + dy - 2);
                }
            }
            luma(image, x, y) - sum / 25.0
        };
        for (name, result) in [("lama", &lama), ("rescreen", &rescreened)] {
            let (mut error, mut count) = (0.0, 0.0);
            let mut sums = [0.0f64; 5];
            for (x, y, value) in mask.enumerate_pixels() {
                if value[0] == 0 {
                    continue;
                }
                error += (luma(result, x, y) - luma(&page, x, y)).abs();
                count += 1.0;
                let (a, b) = (detail(&page, x, y), detail(result, x, y));
                sums[0] += a;
                sums[1] += b;
                sums[2] += a * a;
                sums[3] += b * b;
                sums[4] += a * b;
            }
            let covariance = sums[4] / count - sums[0] * sums[1] / (count * count);
            let correlation = covariance
                / ((sums[2] / count - (sums[0] / count).powi(2))
                    * (sums[3] / count - (sums[1] / count).powi(2)))
                .sqrt()
                .max(1e-9);
            println!(
                "{name}: mean error {:.1}, detail correlation {correlation:.2}",
                error / count
            );
        }
        println!(
            "rescreened {redrawn} of {} masked pixels",
            mask.pixels().filter(|value| value[0] > 0).count()
        );
        for (name, image) in [("page", &page), ("lama", &lama), ("rescreen", &rescreened)] {
            image.save(output.join(format!("{name}.png"))).unwrap();
        }
        for (index, &(left, top)) in holes.iter().enumerate() {
            let (x, y) = (
                left.saturating_sub(18 * scale),
                top.saturating_sub(18 * scale),
            );
            let crop = |image: &RgbImage| {
                image::imageops::resize(
                    &image::imageops::crop_imm(image, x, y, 72 * scale, 64 * scale).to_image(),
                    288,
                    256,
                    image::imageops::FilterType::Nearest,
                )
            };
            let mut sheet = RgbImage::from_pixel(288 * 4 + 30, 256, Rgb([255, 0, 0]));
            for (column, image) in [&page, &damaged, &lama, &rescreened]
                .into_iter()
                .enumerate()
            {
                image::imageops::overlay(&mut sheet, &crop(image), column as i64 * 298, 0);
            }
            sheet
                .save(output.join(format!("hole-{index}.png")))
                .unwrap();
        }
    }

    #[test]
    fn lettering_fills_where_its_surroundings_are_flat() {
        // Left: plain paper with no detected balloon (an open balloon).
        // Middle: a translucent balloon with a 230/255 pattern showing
        // through. Right: a sound effect running from paper onto hatching.
        let mut image = RgbImage::from_fn(300, 100, |x, y| {
            if (100..200).contains(&x) && (x + y).is_multiple_of(2) {
                Rgb([230, 230, 230])
            } else if x >= 250 && (x / 2).is_multiple_of(2) {
                Rgb([40, 40, 40])
            } else {
                Rgb([250, 250, 250])
            }
        });
        let mut mask = GrayImage::new(300, 100);
        for y in 40..60 {
            for x in (20..80).chain(120..180).chain(220..280) {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut pending = mask.clone();
        let filled = fill_flat_lettering(&mut image, &mut pending, &mask, &[]);

        assert_eq!(image.get_pixel(50, 50), &Rgb([250, 250, 250]));
        assert_eq!(pending.get_pixel(50, 50)[0], 0);
        assert!(filled.contains(&(50, 50, Rgb([250, 250, 250]))));
        // The pattern behind the translucent balloon is left to the model.
        assert_eq!(pending.get_pixel(150, 50)[0], u8::MAX);
        // A sound effect running from paper onto hatching cannot be filled
        // whole, so none of it is: the model gets it intact.
        assert_eq!(pending.get_pixel(225, 50)[0], u8::MAX);
        assert_eq!(pending.get_pixel(275, 50)[0], u8::MAX);
        assert_eq!(image.get_pixel(225, 50), &Rgb([10, 10, 10]));
    }

    #[test]
    fn uniform_bubbles_are_filled_independently_inside_one_textured_tile() {
        let mut image = RgbImage::new(200, 100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([20, 80, 140])
            } else {
                Rgb([220, 80, 120])
            };
        }
        for y in 10..90 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgb([245, 245, 245]));
            }
            for x in 110..190 {
                image.put_pixel(x, y, Rgb([240, 244, 250]));
            }
        }
        let mut mask = GrayImage::new(200, 100);
        for y in 40..60 {
            for x in 30..60 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
            for x in 135..165 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;
        let mut pending = mask.clone();
        fill_flat_lettering(
            &mut image,
            &mut pending,
            &mask,
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
        );

        let output = inpaint_tiled(&DynamicImage::ImageRgb8(image), &pending, |_, _| {
            calls += 1;
            Ok(DynamicImage::new_rgb8(1, 1))
        })
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(output.get_pixel(150, 50), &Rgb([240, 244, 250]));
    }

    #[test]
    fn dark_balloon_covered_by_its_text_mask_keeps_its_own_fill() {
        let dark = Rgb([30, 40, 40]);
        let mut image = RgbImage::from_pixel(80, 120, Rgb([200, 200, 200]));
        let mut mask = GrayImage::new(80, 120);
        for y in 10..110 {
            for x in 10..70 {
                image.put_pixel(x, y, dark);
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        for y in (20..100).step_by(10) {
            for x in 25..55 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
                image.put_pixel(x, y + 1, Rgb([255, 255, 255]));
            }
        }
        let mut pending = mask.clone();

        let filled = fill_flat_lettering(
            &mut image,
            &mut pending,
            &mask,
            &[rectangle_region([10, 10, 70, 110])],
        );

        assert!(pending.pixels().all(|pixel| pixel[0] == 0));
        assert_eq!(image.get_pixel(30, 20), &dark);
        assert_eq!(image.get_pixel(40, 61), &dark);

        // A full-page image-edit result that repaints the balloon white is
        // overridden by the balloon's own fill.
        let generated = RgbaImage::from_pixel(80, 120, Rgba([250, 250, 250, 255]));
        let mut cleanup = cleanup_overlay(&generated, &mask, true, None);
        lock_flat_fill(&mut cleanup, &mask, true, &filled);
        assert_eq!(cleanup.get_pixel(40, 61), &Rgba([30, 40, 40, 255]));
        assert_eq!(cleanup.get_pixel(5, 5), &Rgba([250, 250, 250, 255]));
    }

    #[test]
    fn connected_mask_crossing_both_page_grid_axes_is_inferred_once() {
        let image = RgbImage::from_pixel(900, 900, Rgb([20, 40, 60]));
        let mut mask = GrayImage::new(900, 900);
        for y in 500..525 {
            for x in 500..525 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(&DynamicImage::ImageRgb8(image), &mask, |tile, tile_mask| {
            calls += 1;
            assert_eq!(
                tile_mask.pixels().filter(|pixel| pixel[0] >= 127).count(),
                25 * 25
            );
            Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                tile.width(),
                tile.height(),
                Rgb([1, 2, 3]),
            )))
        })
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(500, 500), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(524, 524), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(499, 499), &Rgb([20, 40, 60]));
    }

    #[test]
    fn oversized_component_tiles_mask_their_continuation_context() {
        let mut mask = GrayImage::new(1000, 300);
        for y in 100..120 {
            for x in 50..850 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }

        let (labels, tiles) = inpaint_tiles(&mask);

        assert_eq!(tiles.len(), 2);
        for tile in &tiles {
            assert!(tile.crop[2] - tile.crop[0] <= TILE_SIZE + TILE_CONTEXT * 2);
            assert!(tile.crop[3] - tile.crop[1] <= TILE_SIZE + TILE_CONTEXT * 2);
            let label = tile.components[0];
            let [left, top, right, bottom] = tile.core;
            let mut owned = 0;
            for y in top..bottom {
                for x in left..right {
                    if labels.get_pixel(x, y)[0] == label {
                        owned += 1;
                    }
                }
            }
            let masked = crop_tile_mask(&mask, tile)
                .pixels()
                .filter(|pixel| pixel[0] >= 127)
                .count();
            assert!(masked > owned);
        }
    }

    #[test]
    fn textured_mask_regions_are_inferred_as_bounded_tiles() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let original = image.clone();
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));
        let mut calls = 0;

        let output = inpaint_tiled(&DynamicImage::ImageRgb8(image), &mask, |tile, _| {
            calls += 1;
            assert!(tile.width() <= TILE_CONTEXT * 2 + 1);
            assert!(tile.height() <= TILE_CONTEXT * 2 + 1);
            Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                tile.width(),
                tile.height(),
                Rgb([1, 2, 3]),
            )))
        })
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }
}
