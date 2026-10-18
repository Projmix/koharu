use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
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
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityOrigin, Geometry, Origin,
    RasterLayer, RasterLayerKind, Region, RegionSpec,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{
    StageInput, StageProcessor,
    fal::{
        Fal, FalModel, SANITIZED_FALLBACK_PROMPT, is_content_policy_or_unprocessable,
    },
    finish, generation,
    openrouter_image::{
        DEFAULT_FALLBACK_MODEL, OpenRouterImage, is_moderation_or_policy_error,
    },
};
use crate::{ImageEditConfig, InpaintingApplyMode, ModelCell, StopToken, config::InpaintingModel};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";
const API_DILATION_RADIUS: u32 = 6;
const API_FEATHER_RADIUS: u32 = 4;
const API_RING_WIDTH: u32 = 4;

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
            InpaintingModel::LaMa {} | InpaintingModel::AotInpainting {} => {}
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
            InpaintingModel::MaiImage2_5Edit(settings)
            | InpaintingModel::MaiImage2_5ProEdit(settings) => {
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
            InpaintingModel::MaiImage2_5Edit(_) => "microsoft/mai-image-2.5/edit",
            InpaintingModel::MaiImage2_5ProEdit(_) => "microsoft/mai-image-2.5-pro/edit",
            InpaintingModel::OpenRouterImageEdit { model, .. } => model,
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

    fn skip(&self, input: &StageInput) -> Result<bool> {
        if input.inpainting_mask.is_some() {
            return Ok(false);
        }
        let source = AssetRole::new("source")?;
        for entity in input.scene.children(input.page()?)? {
            if input
                .scene
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
                && input.scene.asset(entity, &source)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
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
            .ok_or_else(|| anyhow!("inpainting model is not loaded"))?
            .run(input)
            .await
    }
}

enum Model {
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
        device: koharu_ml::Device,
        fallback_lama: Arc<ModelCell<Arc<Mutex<LaMa>>>>,
    },
    OpenRouter {
        client: OpenRouterImage,
        model: String,
        config: ImageEditConfig,
        device: koharu_ml::Device,
        fallback_lama: Arc<ModelCell<Arc<Mutex<LaMa>>>>,
    },
}

#[derive(Clone)]
enum CloudProvider {
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
}

impl CloudProvider {
    fn model_id(&self) -> &str {
        match self {
            Self::Fal { model, .. } => model.id(),
            Self::OpenRouter { model, .. } => model.as_str(),
        }
    }

    fn apply_mode(&self) -> InpaintingApplyMode {
        match self {
            Self::Fal { config, .. } | Self::OpenRouter { config, .. } => config.apply_mode,
        }
    }

    async fn edit_image(
        &self,
        image: &DynamicImage,
        prompt_override: Option<&str>,
        use_alternate_model: bool,
        stop: &StopToken,
    ) -> Result<Option<DynamicImage>> {
        match self {
            Self::Fal {
                client,
                model,
                config,
            } => {
                let target_model = if use_alternate_model {
                    model.alternate()
                } else {
                    *model
                };
                let effective_config = match prompt_override {
                    Some(prompt) => ImageEditConfig {
                        prompt: prompt.to_owned(),
                        apply_mode: config.apply_mode,
                    },
                    None => config.clone(),
                };
                client
                    .edit(target_model, &effective_config, image, stop)
                    .await
            }
            Self::OpenRouter {
                client,
                model,
                config,
            } => {
                let effective_prompt = prompt_override.unwrap_or(&config.prompt);
                client.edit(model, effective_prompt, image, stop).await
            }
        }
    }

    fn is_recoverable_policy_error(&self, error: &anyhow::Error) -> bool {
        match self {
            Self::Fal { .. } => is_content_policy_or_unprocessable(error),
            Self::OpenRouter { .. } => is_moderation_or_policy_error(error),
        }
    }
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
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
            InpaintingModel::MaiImage2_5Edit(config) => Ok(Self::Fal {
                client: Fal::new()?,
                model: FalModel::MaiImage2_5Edit,
                config: config.clone(),
                device,
                fallback_lama: Arc::new(ModelCell::new()),
            }),
            InpaintingModel::MaiImage2_5ProEdit(config) => Ok(Self::Fal {
                client: Fal::new()?,
                model: FalModel::MaiImage2_5ProEdit,
                config: config.clone(),
                device,
                fallback_lama: Arc::new(ModelCell::new()),
            }),
            InpaintingModel::OpenRouterImageEdit { model, config } => Ok(Self::OpenRouter {
                client: OpenRouterImage::new()?,
                model: model.clone(),
                config: config.clone(),
                device,
                fallback_lama: Arc::new(ModelCell::new()),
            }),
        }
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let manual = input.inpainting_mask.is_some();
        let configured_full_page = !manual
            && matches!(
                self,
                Self::Fal { config, .. } | Self::OpenRouter { config, .. }
                    if config.apply_mode == InpaintingApplyMode::FullPage
            );
        let mut prepared = prepare(&input).await?;
        let mask_has_pixels = has_active_pixels(&prepared.mask);
        if !configured_full_page && !mask_has_pixels {
            return finish(input.scene.edit());
        }
        let mut effective_mask = prepared.mask.clone();
        let mut effective_full_page = configured_full_page && !mask_has_pixels;
        let original = prepared.original.clone();
        let cleanup = prepared.cleanup.take();
        let cleanup_entity = prepared.cleanup_entity;
        let (model_name, image) = match self {
            Self::LaMa(model) => {
                let model = model.clone();
                (
                    "lama".to_owned(),
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await?,
                )
            }
            Self::Aot(model) => {
                let model = model.clone();
                (
                    "aot-inpainting".to_owned(),
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("AOT model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(image, mask)?))
                            },
                        )
                    })
                    .await?,
                )
            }
            Self::Flux { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "flux2-klein".to_owned(),
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                model.inference(
                                    &config.prompt,
                                    image,
                                    None,
                                    &DynamicImage::ImageLuma8(mask.clone()),
                                    &Flux2KleinInpaintOptions::default(),
                                )
                            },
                        )
                    })
                    .await?,
                )
            }
            Self::Rorem { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "rorem-mixed".to_owned(),
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("RORem model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &config.prompt,
                                    &config.negative_prompt,
                                    &RoremMixedOptions::default(),
                                )?))
                            },
                        )
                    })
                    .await?,
                )
            }
            Self::Fal {
                client,
                model,
                config,
                device,
                fallback_lama,
            } => {
                let provider = CloudProvider::Fal {
                    client: client.clone(),
                    model: *model,
                    config: config.clone(),
                };
                let Some(outcome) = run_cloud_inpaint_with_fallback(
                    &provider,
                    &prepared,
                    manual,
                    device,
                    fallback_lama,
                    &input.stop,
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                effective_mask = outcome.overlay_mask;
                effective_full_page = outcome.full_page_overlay;
                (outcome.model_name, outcome.image)
            }
            Self::OpenRouter {
                client,
                model,
                config,
                device,
                fallback_lama,
            } => {
                let provider = CloudProvider::OpenRouter {
                    client: client.clone(),
                    model: model.clone(),
                    config: config.clone(),
                };
                let Some(outcome) = run_cloud_inpaint_with_fallback(
                    &provider,
                    &prepared,
                    manual,
                    device,
                    fallback_lama,
                    &input.stop,
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                effective_mask = outcome.overlay_mask;
                effective_full_page = outcome.full_page_overlay;
                (outcome.model_name, outcome.image)
            }
        };
        let page = input.page()?;
        let mut edit = if manual {
            input.scene.edit()
        } else {
            input.scene.edit_as(generation(PRODUCER, &model_name)?)
        };
        edit.observe_assets(page)?;
        if let Some(entity) = cleanup_entity {
            edit.observe::<RasterLayer>(entity)?;
            edit.observe_assets(entity)?;
        }
        let image = image.to_rgba8();
        if image.dimensions() != original.dimensions()
            || image.dimensions() != effective_mask.dimensions()
        {
            bail!("inpainted image dimensions do not match page {page}");
        }
        let overlay = cleanup_overlay(
            &image,
            &effective_mask,
            cleanup,
            manual,
            effective_full_page,
        );
        let mut bytes = Cursor::new(Vec::new());
        let width = overlay.width();
        let height = overlay.height();
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
        let cleanup_entity = if let Some(entity) = cleanup_entity {
            if manual {
                let mut layer = input
                    .scene
                    .component::<RasterLayer>(entity)?
                    .context("cleanup entity has no raster layer component")?;
                let generated = layer.origin != Origin::User
                    || input
                        .scene
                        .component::<EntityOrigin>(entity)?
                        .is_some_and(|origin| origin.origin != Origin::User);
                if generated {
                    edit.promote_entity_to_user(entity)?;
                    layer.origin = Origin::User;
                    edit.set(entity, &layer)?;
                }
            }
            entity
        } else {
            let entity = edit.add_entity(page, At::Start)?;
            edit.set(
                entity,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            entity
        };
        edit.set_asset(
            cleanup_entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

fn cleanup_overlay(
    image: &RgbaImage,
    mask: &GrayImage,
    cleanup: Option<RgbaImage>,
    manual: bool,
    full_page: bool,
) -> RgbaImage {
    let mut overlay = if manual {
        cleanup.unwrap_or_else(|| RgbaImage::new(image.width(), image.height()))
    } else {
        RgbaImage::new(image.width(), image.height())
    };
    for (x, y, target) in overlay.enumerate_pixels_mut() {
        if !full_page && mask.get_pixel(x, y)[0] < 127 {
            continue;
        }
        let generated = image.get_pixel(x, y);
        // Opaque cleanup texels prevent linear canvas sampling from revealing
        // the original lettering at edited edges.
        *target = Rgba([generated[0], generated[1], generated[2], 255]);
    }
    overlay
}

#[derive(Clone, Debug)]
struct FlatFillRegion {
    bounds: [u32; 4],
    polygon: Vec<(f32, f32)>,
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
    cleanup_entity: Option<koharu_scene::EntityId>,
    cleanup: Option<RgbaImage>,
    mask: GrayImage,
    text_mask: GrayImage,
    flat_fill_regions: Vec<FlatFillRegion>,
}

async fn prepare(input: &StageInput) -> Result<InpaintInput> {
    let page = input.page()?;
    let original = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let cleanup_entity = input.scene.children(page)?.find(|entity| {
        input
            .scene
            .component::<RasterLayer>(*entity)
            .ok()
            .flatten()
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
    });
    let cleanup = if let Some(entity) = cleanup_entity {
        input
            .images
            .get(&input.scene, entity, "source")
            .await?
            .map(|image| image.to_rgba8())
    } else {
        None
    };
    if cleanup
        .as_ref()
        .is_some_and(|image| image.dimensions() != original.dimensions())
    {
        bail!("cleanup layer dimensions do not match page {page}");
    }
    let source = if input.inpainting_mask.is_some() {
        if let Some(cleanup) = cleanup.as_ref() {
            let mut composite = original.to_rgba8();
            image::imageops::overlay(&mut composite, cleanup, 0, 0);
            Arc::new(DynamicImage::ImageRgba8(composite))
        } else {
            original.clone()
        }
    } else {
        original.clone()
    };
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
    let flat_fill_regions = flat_fill_regions(input, source.width(), source.height())?;
    Ok(InpaintInput {
        image: source,
        original,
        cleanup_entity,
        cleanup,
        mask,
        text_mask,
        flat_fill_regions,
    })
}

const TILE_SIZE: u32 = 512;
const TILE_CONTEXT: u32 = 128;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;

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
// filled first; only the remaining mask is split into bounded model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        image.dimensions() == mask.dimensions() && mask.dimensions() == text_mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}",
        image.dimensions(),
        mask.dimensions(),
        text_mask.dimensions()
    );

    let mut output = image.to_rgb8();
    let mut pending_mask = mask.clone();
    fill_uniform_regions(&mut output, &mut pending_mask, text_mask, flat_fill_regions);
    let (component_labels, tiles) = inpaint_tiles(&pending_mask);
    for tile in &tiles {
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&output, left, top, crop_width, crop_height).to_image();
        let crop_mask = crop_tile_mask(&pending_mask, tile);

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

fn has_active_pixels(mask: &GrayImage) -> bool {
    mask.as_raw().iter().any(|&value| value >= 127)
}

fn fill_uniform_regions(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
) {
    let mut unused = GrayImage::new(output.width(), output.height());
    fill_uniform_regions_tracked(output, pending_mask, text_mask, regions, &mut unused);
}

fn fill_uniform_regions_tracked(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
    filled_mask: &mut GrayImage,
) {
    for region in regions {
        let [left, top, right, bottom] = region.bounds;
        let mut targets = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if pending_mask.get_pixel(x, y)[0] >= 127
                    && text_mask.get_pixel(x, y)[0] >= 127
                    && point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), &region.polygon)
                {
                    targets.push((x, y));
                }
            }
        }
        if targets.is_empty() {
            continue;
        }
        let Some(color) = uniform_region_color(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
            filled_mask.put_pixel(x, y, Luma([u8::MAX]));
        }
    }
}

struct CloudInpaintOutcome {
    model_name: String,
    image: DynamicImage,
    overlay_mask: GrayImage,
    full_page_overlay: bool,
}

async fn run_cloud_inpaint_with_fallback(
    provider: &CloudProvider,
    prepared: &InpaintInput,
    manual: bool,
    device: &koharu_ml::Device,
    fallback_lama: &Arc<ModelCell<Arc<Mutex<LaMa>>>>,
    stop: &StopToken,
) -> Result<Option<CloudInpaintOutcome>> {
    let width = prepared.image.width();
    let height = prepared.image.height();
    let original_rgb = prepared.original.to_rgb8();
    let mut working_rgb = prepared.image.to_rgb8();
    let mut pending_mask = prepared.mask.clone();
    let mut overlay_mask = GrayImage::new(width, height);
    let initial_has_mask = has_active_pixels(&prepared.mask);

    // Step 0: Always fill simple uniform speech bubbles locally BEFORE calling any
    // cloud API so bubble outlines are never erased and simple bubbles cost 0 API calls.
    if !manual && initial_has_mask {
        fill_uniform_regions_tracked(
            &mut working_rgb,
            &mut pending_mask,
            &prepared.text_mask,
            &prepared.flat_fill_regions,
            &mut overlay_mask,
        );
        if !has_active_pixels(&pending_mask) {
            return Ok(Some(CloudInpaintOutcome {
                model_name: provider.model_id().to_owned(),
                image: DynamicImage::ImageRgb8(working_rgb),
                overlay_mask,
                full_page_overlay: false,
            }));
        }
    }

    let mut used_model_name = provider.model_id().to_owned();

    // Step 1: Primary cloud attempt.
    // If FullPage mode is configured and there are no masks at all on the page, send the
    // full page and apply as full_page_overlay. Otherwise, composite via dilated feathered
    // masks with boundary-ring luminance correction and verify with DiffInspector.
    if !manual && provider.apply_mode() == InpaintingApplyMode::FullPage {
        let input_dynamic = DynamicImage::ImageRgb8(working_rgb.clone());
        match provider.edit_image(&input_dynamic, None, false, stop).await {
            Ok(None) => return Ok(None),
            Ok(Some(generated)) => {
                let generated_rgb = ensure_rgb_dimensions(&generated, width, height);
                if !initial_has_mask {
                    return Ok(Some(CloudInpaintOutcome {
                        model_name: used_model_name,
                        image: DynamicImage::ImageRgb8(generated_rgb),
                        overlay_mask,
                        full_page_overlay: true,
                    }));
                }
                let mut candidate_rgb = working_rgb.clone();
                let mut candidate_overlay = overlay_mask.clone();
                composite_api_with_feathering(
                    &mut candidate_rgb,
                    &generated_rgb,
                    &pending_mask,
                    &mut candidate_overlay,
                    API_DILATION_RADIUS,
                    API_FEATHER_RADIUS,
                );
                let inspection = inspect_inpaint_diff(
                    &original_rgb,
                    &candidate_rgb,
                    &pending_mask,
                    &prepared.text_mask,
                    &prepared.flat_fill_regions,
                );
                accept_passed_components(
                    &mut working_rgb,
                    &candidate_rgb,
                    &mut overlay_mask,
                    &candidate_overlay,
                    &pending_mask,
                    &inspection.failed_mask,
                );
                pending_mask = inspection.failed_mask;
                if !has_active_pixels(&pending_mask) {
                    return Ok(Some(CloudInpaintOutcome {
                        model_name: used_model_name,
                        image: DynamicImage::ImageRgb8(working_rgb),
                        overlay_mask,
                        full_page_overlay: false,
                    }));
                }
                tracing::warn!(
                    model = %used_model_name,
                    "FullPage inpainting left residual ink or border damage; switching to tiled fallback"
                );
            }
            Err(error) if provider.is_recoverable_policy_error(&error) => {
                tracing::warn!(
                    model = %used_model_name,
                    %error,
                    "FullPage cloud inpainting rejected by content policy / 422; switching to tiled crop fallback"
                );
                if !initial_has_mask {
                    // Without masks we cannot tile; return original unchanged rather than crashing.
                    return Ok(Some(CloudInpaintOutcome {
                        model_name: used_model_name,
                        image: DynamicImage::ImageRgb8(working_rgb),
                        overlay_mask,
                        full_page_overlay: false,
                    }));
                }
            }
            Err(error) => return Err(error),
        }
    } else if has_active_pixels(&pending_mask) {
        // Primary mode is Mask (tiled) or manual mask.
        let tiled_ok = try_cloud_tiled_pass(
            provider,
            &original_rgb,
            &mut working_rgb,
            &mut pending_mask,
            &prepared.text_mask,
            &prepared.flat_fill_regions,
            &mut overlay_mask,
            None,
            false,
            stop,
        )
        .await?;
        if stop.stopped() {
            return Ok(None);
        }
        if tiled_ok && !has_active_pixels(&pending_mask) {
            return Ok(Some(CloudInpaintOutcome {
                model_name: used_model_name,
                image: DynamicImage::ImageRgb8(working_rgb),
                overlay_mask,
                full_page_overlay: false,
            }));
        }
    }

    // Fallback Step 1: Tiled crops around remaining `pending_mask` with sanitized technical
    // prompt (and alternate model variant on Fal if primary was already tiled).
    if has_active_pixels(&pending_mask) {
        let use_alternate_model =
            manual || provider.apply_mode() == InpaintingApplyMode::Mask;
        let _ = try_cloud_tiled_pass(
            provider,
            &original_rgb,
            &mut working_rgb,
            &mut pending_mask,
            &prepared.text_mask,
            &prepared.flat_fill_regions,
            &mut overlay_mask,
            Some(SANITIZED_FALLBACK_PROMPT),
            use_alternate_model,
            stop,
        )
        .await;
        if stop.stopped() {
            return Ok(None);
        }
        if !has_active_pixels(&pending_mask) {
            return Ok(Some(CloudInpaintOutcome {
                model_name: used_model_name,
                image: DynamicImage::ImageRgb8(working_rgb),
                overlay_mask,
                full_page_overlay: false,
            }));
        }
    }

    // Fallback Step 2: Cross-provider cloud fallback (Fal <-> OpenRouterImage) if the
    // alternate provider's secret is configured.
    if has_active_pixels(&pending_mask)
        && let Some(alt_provider) = alternate_cloud_provider(provider)
    {
        tracing::info!(
            primary = %provider.model_id(),
            alternate = %alt_provider.model_id(),
            "Attempting cross-provider cloud inpainting fallback for remaining masks"
        );
        let alt_name = alt_provider.model_id().to_owned();
        let _ = try_cloud_tiled_pass(
            &alt_provider,
            &original_rgb,
            &mut working_rgb,
            &mut pending_mask,
            &prepared.text_mask,
            &prepared.flat_fill_regions,
            &mut overlay_mask,
            Some(SANITIZED_FALLBACK_PROMPT),
            false,
            stop,
        )
        .await;
        if stop.stopped() {
            return Ok(None);
        }
        if !has_active_pixels(&pending_mask) {
            return Ok(Some(CloudInpaintOutcome {
                model_name: alt_name,
                image: DynamicImage::ImageRgb8(working_rgb),
                overlay_mask,
                full_page_overlay: false,
            }));
        }
        used_model_name = alt_name;
    }

    // Fallback Step 3: Local LaMa model on remaining `pending_mask` tiles, followed by
    // deterministic harmonic boundary interpolation if local model weights are unavailable.
    if has_active_pixels(&pending_mask) {
        if fallback_lama
            .ensure(|| async {
                Ok(Arc::new(Mutex::new(LaMa::load(device.clone()).await?)))
            })
            .await
            .is_ok()
            && let Some(lama) = fallback_lama.lock().await.as_ref().cloned()
        {
            let input_rgb = working_rgb.clone();
            let mask_clone = pending_mask.clone();
            let text_mask_clone = prepared.text_mask.clone();
            let regions_clone = prepared.flat_fill_regions.clone();
            if let Ok(Ok(local_result)) =
                tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                    let model = lama
                        .lock()
                        .map_err(|_| anyhow!("fallback LaMa lock is poisoned"))?;
                    inpaint_tiled(
                        &DynamicImage::ImageRgb8(input_rgb),
                        &mask_clone,
                        &text_mask_clone,
                        &regions_clone,
                        |crop_img, crop_mask| {
                            Ok(DynamicImage::ImageRgb8(model.inference(
                                crop_img,
                                crop_mask,
                                &InpaintRequest::default(),
                            )?))
                        },
                    )
                })
                .await
            {
                let local_rgb = local_result.to_rgb8();
                composite_api_with_feathering(
                    &mut working_rgb,
                    &local_rgb,
                    &pending_mask,
                    &mut overlay_mask,
                    API_DILATION_RADIUS,
                    API_FEATHER_RADIUS,
                );
                pending_mask = GrayImage::new(width, height);
                used_model_name = "lama".to_owned();
            }
        }

        if has_active_pixels(&pending_mask) {
            inpaint_harmonic_fallback(&mut working_rgb, &pending_mask, &mut overlay_mask);
        }
    }

    Ok(Some(CloudInpaintOutcome {
        model_name: used_model_name,
        image: DynamicImage::ImageRgb8(working_rgb),
        overlay_mask,
        full_page_overlay: false,
    }))
}

fn alternate_cloud_provider(primary: &CloudProvider) -> Option<CloudProvider> {
    match primary {
        CloudProvider::Fal { config, .. } => {
            let has_openrouter = koharu_secrets::get("openrouter")
                .ok()
                .flatten()
                .is_some_and(|secret| {
                    use koharu_secrets::ExposeSecret as _;
                    !secret.expose_secret().trim().is_empty()
                });
            if !has_openrouter {
                return None;
            }
            OpenRouterImage::new()
                .ok()
                .map(|client| CloudProvider::OpenRouter {
                    client,
                    model: DEFAULT_FALLBACK_MODEL.to_owned(),
                    config: config.clone(),
                })
        }
        CloudProvider::OpenRouter { config, .. } => {
            let has_fal = koharu_secrets::get("fal")
                .ok()
                .flatten()
                .is_some_and(|secret| {
                    use koharu_secrets::ExposeSecret as _;
                    !secret.expose_secret().trim().is_empty()
                });
            if !has_fal {
                return None;
            }
            Fal::new().ok().map(|client| CloudProvider::Fal {
                client,
                model: FalModel::MaiImage2_5Edit,
                config: config.clone(),
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn try_cloud_tiled_pass(
    provider: &CloudProvider,
    original_rgb: &RgbImage,
    working_rgb: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
    overlay_mask: &mut GrayImage,
    prompt_override: Option<&str>,
    use_alternate_model: bool,
    stop: &StopToken,
) -> Result<bool> {
    let (_, tiles) = inpaint_tiles(pending_mask);
    if tiles.is_empty() {
        return Ok(true);
    }
    let mut candidate_rgb = working_rgb.clone();
    let mut attempted_mask = GrayImage::new(working_rgb.width(), working_rgb.height());
    let mut any_succeeded = false;

    for tile in &tiles {
        if stop.stopped() {
            return Ok(false);
        }
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&candidate_rgb, left, top, crop_width, crop_height)
                .to_image();
        let crop_mask = crop_tile_mask(pending_mask, tile);
        if !has_active_pixels(&crop_mask) {
            continue;
        }
        match provider
            .edit_image(
                &DynamicImage::ImageRgb8(crop_image),
                prompt_override,
                use_alternate_model,
                stop,
            )
            .await
        {
            Ok(None) => return Ok(false),
            Ok(Some(generated_crop)) => {
                any_succeeded = true;
                let generated_crop_rgb =
                    ensure_rgb_dimensions(&generated_crop, crop_width, crop_height);
                let mut crop_target =
                    image::imageops::crop_imm(&candidate_rgb, left, top, crop_width, crop_height)
                        .to_image();
                let mut crop_overlay = GrayImage::new(crop_width, crop_height);
                composite_api_with_feathering(
                    &mut crop_target,
                    &generated_crop_rgb,
                    &crop_mask,
                    &mut crop_overlay,
                    API_DILATION_RADIUS,
                    API_FEATHER_RADIUS,
                );
                let [core_left, core_top, core_right, core_bottom] = tile.core;
                let pad = API_DILATION_RADIUS + API_FEATHER_RADIUS;
                let apply_left = core_left.saturating_sub(pad).max(left);
                let apply_top = core_top.saturating_sub(pad).max(top);
                let apply_right = core_right.saturating_add(pad).min(right);
                let apply_bottom = core_bottom.saturating_add(pad).min(bottom);
                for y in apply_top..apply_bottom {
                    for x in apply_left..apply_right {
                        let cx = x - left;
                        let cy = y - top;
                        if crop_overlay.get_pixel(cx, cy)[0] >= 127 {
                            candidate_rgb.put_pixel(x, y, *crop_target.get_pixel(cx, cy));
                        }
                        if pending_mask.get_pixel(x, y)[0] >= 127 {
                            attempted_mask.put_pixel(x, y, Luma([u8::MAX]));
                        }
                    }
                }
            }
            Err(error) if provider.is_recoverable_policy_error(&error) => {
                tracing::warn!(
                    model = %provider.model_id(),
                    ?tile.crop,
                    %error,
                    "Tiled crop rejected by content policy / 422; leaving tile for next fallback stage"
                );
            }
            Err(error) => {
                if prompt_override.is_some() {
                    tracing::warn!(
                        model = %provider.model_id(),
                        ?tile.crop,
                        %error,
                        "Fallback tiled crop failed; continuing fallback chain"
                    );
                } else {
                    return Err(error);
                }
            }
        }
    }

    if !any_succeeded {
        return Ok(false);
    }

    let mut candidate_overlay = overlay_mask.clone();
    mark_dilated_overlay_pixels(
        &attempted_mask,
        &mut candidate_overlay,
        API_DILATION_RADIUS + API_FEATHER_RADIUS,
    );
    let inspection = inspect_inpaint_diff(
        original_rgb,
        &candidate_rgb,
        &attempted_mask,
        text_mask,
        flat_fill_regions,
    );
    let mut next_pending = pending_mask.clone();
    for (x, y, pixel) in attempted_mask.enumerate_pixels() {
        if pixel[0] >= 127 && inspection.failed_mask.get_pixel(x, y)[0] < 127 {
            next_pending.put_pixel(x, y, Luma([0]));
        }
    }
    accept_passed_components(
        working_rgb,
        &candidate_rgb,
        overlay_mask,
        &candidate_overlay,
        &attempted_mask,
        &inspection.failed_mask,
    );
    *pending_mask = next_pending;
    Ok(inspection.passed)
}

fn ensure_rgb_dimensions(image: &DynamicImage, width: u32, height: u32) -> RgbImage {
    if image.dimensions() == (width, height) {
        image.to_rgb8()
    } else {
        image
            .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgb8()
    }
}

fn accept_passed_components(
    working_rgb: &mut RgbImage,
    candidate_rgb: &RgbImage,
    overlay_mask: &mut GrayImage,
    candidate_overlay: &GrayImage,
    attempted_mask: &GrayImage,
    failed_mask: &GrayImage,
) {
    let pad = API_DILATION_RADIUS + API_FEATHER_RADIUS;
    let passed_seed = GrayImage::from_fn(attempted_mask.width(), attempted_mask.height(), |x, y| {
        if attempted_mask.get_pixel(x, y)[0] >= 127 && failed_mask.get_pixel(x, y)[0] < 127 {
            Luma([u8::MAX])
        } else {
            Luma([0])
        }
    });
    if !has_active_pixels(&passed_seed) {
        return;
    }
    let dist_sq = bounded_edt_squared(&passed_seed, pad);
    let max_dist_sq = pad * pad;
    let width = working_rgb.width();
    for y in 0..working_rgb.height() {
        let row_offset = (y * width) as usize;
        for x in 0..width {
            if dist_sq[row_offset + x as usize] <= max_dist_sq {
                working_rgb.put_pixel(x, y, *candidate_rgb.get_pixel(x, y));
                if candidate_overlay.get_pixel(x, y)[0] >= 127 {
                    overlay_mask.put_pixel(x, y, Luma([u8::MAX]));
                }
            }
        }
    }
}

fn mark_dilated_overlay_pixels(mask: &GrayImage, overlay_mask: &mut GrayImage, radius: u32) {
    if !has_active_pixels(mask) {
        return;
    }
    let dist_sq = bounded_edt_squared(mask, radius);
    let max_dist_sq = radius * radius;
    let width = mask.width();
    for y in 0..mask.height() {
        let row_offset = (y * width) as usize;
        for x in 0..width {
            if dist_sq[row_offset + x as usize] <= max_dist_sq {
                overlay_mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
    }
}

/// Composites `generated` over `base` only within `dilate(mask, dilation_radius + feather_radius)`,
/// correcting per-component local brightness/color shifts measured in the outer boundary ring
/// `[dilation_radius + feather_radius, dilation_radius + feather_radius + API_RING_WIDTH]`
/// and applying a C1 smoothstep feather ramp across `feather_radius` pixels.
fn composite_api_with_feathering(
    base: &mut RgbImage,
    generated: &RgbImage,
    mask: &GrayImage,
    overlay_mask: &mut GrayImage,
    dilation_radius: u32,
    feather_radius: u32,
) {
    if !has_active_pixels(mask) {
        return;
    }
    let width = base.width();
    let height = base.height();
    let outer_radius = dilation_radius + feather_radius;
    let ring_outer = outer_radius + API_RING_WIDTH;
    let dist_sq = bounded_edt_squared(mask, ring_outer);
    let (_, components) = mask_components(mask);

    for component in &components {
        let [c_left, c_top, c_right, c_bottom] = component.bounds;
        let box_left = c_left.saturating_sub(ring_outer);
        let box_top = c_top.saturating_sub(ring_outer);
        let box_right = c_right.saturating_add(ring_outer).min(width);
        let box_bottom = c_bottom.saturating_add(ring_outer).min(height);

        let outer_sq = outer_radius * outer_radius;
        let ring_sq = ring_outer * ring_outer;
        let mut ring_diff_sum = [0.0_f64; 3];
        let mut ring_count = 0_usize;

        for y in box_top..box_bottom {
            let row_offset = (y * width) as usize;
            for x in box_left..box_right {
                let d2 = dist_sq[row_offset + x as usize];
                if d2 > outer_sq && d2 <= ring_sq {
                    let b_px = base.get_pixel(x, y);
                    let g_px = generated.get_pixel(x, y);
                    for c in 0..3 {
                        ring_diff_sum[c] += f64::from(g_px[c]) - f64::from(b_px[c]);
                    }
                    ring_count += 1;
                }
            }
        }

        let shift = if ring_count >= 8 {
            [
                (ring_diff_sum[0] / ring_count as f64).clamp(-48.0, 48.0) as f32,
                (ring_diff_sum[1] / ring_count as f64).clamp(-48.0, 48.0) as f32,
                (ring_diff_sum[2] / ring_count as f64).clamp(-48.0, 48.0) as f32,
            ]
        } else {
            [0.0; 3]
        };

        let blend_left = c_left.saturating_sub(outer_radius);
        let blend_top = c_top.saturating_sub(outer_radius);
        let blend_right = c_right.saturating_add(outer_radius).min(width);
        let blend_bottom = c_bottom.saturating_add(outer_radius).min(height);
        let inner_r = dilation_radius as f32;
        let feather_w = (feather_radius.max(1)) as f32;

        for y in blend_top..blend_bottom {
            let row_offset = (y * width) as usize;
            for x in blend_left..blend_right {
                let d2 = dist_sq[row_offset + x as usize];
                if d2 > outer_sq {
                    continue;
                }
                let d = (d2 as f32).sqrt();
                let alpha = if feather_radius == 0 || d <= inner_r {
                    1.0_f32
                } else {
                    let t = ((d - inner_r) / feather_w).clamp(0.0, 1.0);
                    1.0 - t * t * (3.0 - 2.0 * t)
                };
                if alpha <= 0.0 {
                    continue;
                }
                let b_px = base.get_pixel(x, y);
                let g_px = generated.get_pixel(x, y);
                let mut blended = [0_u8; 3];
                for c in 0..3 {
                    let corrected = (f32::from(g_px[c]) - shift[c]).clamp(0.0, 255.0);
                    let out = corrected * alpha + f32::from(b_px[c]) * (1.0 - alpha);
                    blended[c] = out.round().clamp(0.0, 255.0) as u8;
                }
                base.put_pixel(x, y, Rgb(blended));
                overlay_mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
    }
}

/// Computes squared Euclidean distance to the nearest pixel with `mask(x, y) >= 127`,
/// bounded by `max_radius` (values beyond `max_radius` are capped at `(max_radius + 1)^2`).
fn bounded_edt_squared(mask: &GrayImage, max_radius: u32) -> Vec<u32> {
    let width = mask.width() as usize;
    let height = mask.height() as usize;
    let inf = (max_radius + 1) * (max_radius + 1);
    let mut horiz = vec![inf; width * height];

    for y in 0..height {
        let row = y * width;
        let mut dist = max_radius + 1;
        for x in 0..width {
            if mask.get_pixel(x as u32, y as u32)[0] >= 127 {
                dist = 0;
            } else {
                dist = (dist + 1).min(max_radius + 1);
            }
            horiz[row + x] = dist * dist;
        }
        dist = max_radius + 1;
        for x in (0..width).rev() {
            if mask.get_pixel(x as u32, y as u32)[0] >= 127 {
                dist = 0;
            } else {
                dist = (dist + 1).min(max_radius + 1);
            }
            let d2 = dist * dist;
            if d2 < horiz[row + x] {
                horiz[row + x] = d2;
            }
        }
    }

    let mut out = vec![inf; width * height];
    let r = max_radius as isize;
    for y in 0..height {
        let row = y * width;
        let y_min = (y as isize - r).max(0) as usize;
        let y_max = (y as isize + r).min(height as isize - 1) as usize;
        for x in 0..width {
            let mut best = horiz[row + x];
            if best == 0 {
                out[row + x] = 0;
                continue;
            }
            for ny in y_min..=y_max {
                let dy = (y as isize - ny as isize).unsigned_abs() as u32;
                let dy2 = dy * dy;
                if dy2 >= best {
                    continue;
                }
                let candidate = horiz[ny * width + x].saturating_add(dy2);
                if candidate < best {
                    best = candidate;
                }
            }
            out[row + x] = best.min(inf);
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ComponentInspection {
    pub label: u32,
    pub bounds: [u32; 4],
    pub is_simple_bubble: bool,
    pub residual_ink_score: f64,
    pub sobel_stroke_correlation: f64,
    pub bubble_border_integrity: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiffInspectionReport {
    pub passed: bool,
    pub components: Vec<ComponentInspection>,
    pub failed_mask: GrayImage,
}

/// Autonomous Diff-Inspector:
/// - For simple bubbles (`FlowsIn` with low background variance), checks luminance variance and
///   deviation from the bubble's background color inside `text_mask`.
/// - For SFX (`coo-mask`) and text over screentones/artwork, does NOT reject based on variance
///   (which would falsely reject reconstructed screentones); instead computes Sobel gradient
///   stroke silhouette correlation between `original[mask]` and `generated[mask]` + checks
///   `Bubble Border Integrity` along any enclosing/adjacent bubble polygon contour.
pub(crate) fn inspect_inpaint_diff(
    original: &RgbImage,
    generated: &RgbImage,
    mask: &GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
) -> DiffInspectionReport {
    let width = original.width();
    let height = original.height();
    let (labels, components) = mask_components(mask);
    let mut failed_mask = GrayImage::new(width, height);
    let mut reports = Vec::with_capacity(components.len());
    let mut all_passed = true;

    for component in components {
        let [left, top, right, bottom] = component.bounds;
        let matching_region = flat_fill_regions.iter().find(|region| {
            let cx = (left + right) as f32 * 0.5;
            let cy = (top + bottom) as f32 * 0.5;
            point_in_polygon((cx, cy), &region.polygon)
        });
        let simple_bg_color =
            matching_region.and_then(|region| uniform_region_color(original, mask, region));
        let is_simple_bubble = simple_bg_color.is_some();

        let bubble_border_integrity = matching_region
            .map(|region| compute_bubble_border_integrity(original, generated, mask, region))
            .unwrap_or(1.0);

        let (residual_ink_score, sobel_stroke_correlation, content_passed) =
            if let Some(bg_color) = simple_bg_color {
                let score = compute_simple_bubble_residual_ink(
                    generated,
                    &labels,
                    component,
                    text_mask,
                    bg_color,
                );
                (score, 0.0, score < 18.0)
            } else {
                let (stroke_corr, stroke_retention) =
                    compute_sobel_stroke_correlation(original, generated, &labels, component);
                // Reject only when both the oriented Sobel stroke silhouette correlates strongly
                // with the original lettering AND a significant fraction of original ink strokes
                // remained uncleaned at the exact same coordinates.
                let passed = !(stroke_corr > 0.78 && stroke_retention > 0.55);
                (stroke_retention * 100.0, stroke_corr, passed)
            };

        let passed = content_passed && bubble_border_integrity >= 0.78;
        if !passed {
            all_passed = false;
            for y in top..bottom {
                for x in left..right {
                    if labels.get_pixel(x, y)[0] == component.label {
                        failed_mask.put_pixel(x, y, Luma([u8::MAX]));
                    }
                }
            }
        }

        reports.push(ComponentInspection {
            label: component.label,
            bounds: component.bounds,
            is_simple_bubble,
            residual_ink_score,
            sobel_stroke_correlation,
            bubble_border_integrity,
            passed,
        });
    }

    DiffInspectionReport {
        passed: all_passed,
        components: reports,
        failed_mask,
    }
}

fn pixel_luma(pixel: &Rgb<u8>) -> f64 {
    0.299 * f64::from(pixel[0]) + 0.587 * f64::from(pixel[1]) + 0.114 * f64::from(pixel[2])
}

fn compute_simple_bubble_residual_ink(
    generated: &RgbImage,
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    component: MaskComponent,
    text_mask: &GrayImage,
    bg_color: Rgb<u8>,
) -> f64 {
    let [left, top, right, bottom] = component.bounds;
    let bg_luma = pixel_luma(&bg_color);
    let mut lumas = Vec::new();
    let mut high_contrast_count = 0_usize;

    for y in top..bottom {
        for x in left..right {
            if labels.get_pixel(x, y)[0] != component.label {
                continue;
            }
            if text_mask.get_pixel(x, y)[0] > 0 && text_mask.get_pixel(x, y)[0] < 127 {
                continue;
            }
            let luma = pixel_luma(generated.get_pixel(x, y));
            if (luma - bg_luma).abs() > 35.0 {
                high_contrast_count += 1;
            }
            lumas.push(luma);
        }
    }
    if lumas.is_empty() {
        return 0.0;
    }
    let mean = lumas.iter().sum::<f64>() / lumas.len() as f64;
    let variance = lumas.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / lumas.len() as f64;
    let std_dev = variance.sqrt();
    let outlier_penalty = (high_contrast_count as f64 / lumas.len() as f64) * 50.0;
    std_dev.max((mean - bg_luma).abs() * 0.5) + outlier_penalty
}

fn compute_sobel_stroke_correlation(
    original: &RgbImage,
    generated: &RgbImage,
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    component: MaskComponent,
) -> (f64, f64) {
    let [left, top, right, bottom] = component.bounds;
    let width = original.width();
    let height = original.height();

    let mut orig_mags = Vec::new();
    let mut gen_mags = Vec::new();
    let mut strong_stroke_edges = 0_usize;
    let mut retained_stroke_edges = 0_usize;

    for y in top..bottom {
        for x in left..right {
            if labels.get_pixel(x, y)[0] != component.label {
                continue;
            }
            let (ox, oy, omag) = sobel_gradient(original, x, y, width, height);
            let (gx, gy, gmag) = sobel_gradient(generated, x, y, width, height);
            orig_mags.push(omag);
            gen_mags.push(gmag);

            if omag >= 70.0 {
                strong_stroke_edges += 1;
                let cos_sim = if omag > 1e-5 && gmag > 1e-5 {
                    (ox * gx + oy * gy) / (omag * gmag)
                } else {
                    0.0
                };
                let orig_luma = pixel_luma(original.get_pixel(x, y));
                let gen_luma = pixel_luma(generated.get_pixel(x, y));
                if gmag >= omag * 0.60 && cos_sim >= 0.72 && (orig_luma - gen_luma).abs() < 24.0 {
                    retained_stroke_edges += 1;
                }
            }
        }
    }

    if orig_mags.len() < 4 {
        return (0.0, 0.0);
    }

    let mean_o = orig_mags.iter().sum::<f64>() / orig_mags.len() as f64;
    let mean_g = gen_mags.iter().sum::<f64>() / gen_mags.len() as f64;
    let mut cov = 0.0_f64;
    let mut var_o = 0.0_f64;
    let mut var_g = 0.0_f64;
    for (&o, &g) in orig_mags.iter().zip(gen_mags.iter()) {
        let do_ = o - mean_o;
        let dg = g - mean_g;
        cov += do_ * dg;
        var_o += do_ * do_;
        var_g += dg * dg;
    }
    let denom = (var_o * var_g).sqrt();
    let correlation = if denom > 1e-6 {
        (cov / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let retention = if strong_stroke_edges > 0 {
        retained_stroke_edges as f64 / strong_stroke_edges as f64
    } else {
        // Fallback when component is small solid ink without internal gradient transitions:
        // compare whether pixel luma changed meaningfully from original ink.
        let mut unchanged = 0_usize;
        let mut total = 0_usize;
        for y in top..bottom {
            for x in left..right {
                if labels.get_pixel(x, y)[0] == component.label {
                    total += 1;
                    let diff = (pixel_luma(original.get_pixel(x, y))
                        - pixel_luma(generated.get_pixel(x, y)))
                    .abs();
                    if diff < 12.0 {
                        unchanged += 1;
                    }
                }
            }
        }
        if total > 0 {
            unchanged as f64 / total as f64
        } else {
            0.0
        }
    };

    (correlation, retention)
}

fn sobel_gradient(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> (f64, f64, f64) {
    let sample = |dx: i32, dy: i32| -> f64 {
        let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
        let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
        pixel_luma(image.get_pixel(sx, sy))
    };
    let gx = -sample(-1, -1) + sample(1, -1) - 2.0 * sample(-1, 0) + 2.0 * sample(1, 0)
        - sample(-1, 1)
        + sample(1, 1);
    let gy = -sample(-1, -1) - 2.0 * sample(0, -1) - sample(1, -1)
        + sample(-1, 1)
        + 2.0 * sample(0, 1)
        + sample(1, 1);
    let mag = gx.hypot(gy);
    (gx, gy, mag)
}

fn compute_bubble_border_integrity(
    original: &RgbImage,
    generated: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
) -> f64 {
    let [left, top, right, bottom] = region.bounds;
    let pad = 3_u32;
    let box_left = left.saturating_sub(pad);
    let box_top = top.saturating_sub(pad);
    let box_right = right.saturating_add(pad).min(original.width());
    let box_bottom = bottom.saturating_add(pad).min(original.height());

    let mut diff_sum = 0.0_f64;
    let mut count = 0_usize;
    for y in box_top..box_bottom {
        for x in box_left..box_right {
            if mask.get_pixel(x, y)[0] >= 127 {
                continue;
            }
            let pt = (x as f32 + 0.5, y as f32 + 0.5);
            if polygon_edge_distance_squared(pt, &region.polygon) <= 9.0 {
                let o_luma = pixel_luma(original.get_pixel(x, y));
                let g_luma = pixel_luma(generated.get_pixel(x, y));
                diff_sum += (o_luma - g_luma).abs() / 255.0;
                count += 1;
            }
        }
    }
    if count == 0 {
        return 1.0;
    }
    (1.0 - diff_sum / count as f64).clamp(0.0, 1.0)
}

/// Deterministic boundary-harmonic diffusion fallback for any mask components that could not
/// be processed by cloud providers (e.g., 422 content policy) when local neural weights are
/// unavailable. Propagates surrounding boundary colors inward and smooths iteratively.
fn inpaint_harmonic_fallback(
    image: &mut RgbImage,
    mask: &GrayImage,
    overlay_mask: &mut GrayImage,
) {
    let width = image.width();
    let height = image.height();
    let (_, components) = mask_components(mask);
    for component in components {
        let [left, top, right, bottom] = component.bounds;
        let box_left = left.saturating_sub(4);
        let box_top = top.saturating_sub(4);
        let box_right = right.saturating_add(4).min(width);
        let box_bottom = bottom.saturating_add(4).min(height);

        let mut border_r = Vec::new();
        let mut border_g = Vec::new();
        let mut border_b = Vec::new();
        for y in box_top..box_bottom {
            for x in box_left..box_right {
                if mask.get_pixel(x, y)[0] < 127 {
                    let px = image.get_pixel(x, y);
                    border_r.push(px[0]);
                    border_g.push(px[1]);
                    border_b.push(px[2]);
                }
            }
        }
        let seed_color = if !border_r.is_empty() {
            Rgb([
                median(&mut border_r),
                median(&mut border_g),
                median(&mut border_b),
            ])
        } else {
            Rgb([255, 255, 255])
        };

        let mut masked_coords = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if mask.get_pixel(x, y)[0] >= 127 {
                    image.put_pixel(x, y, seed_color);
                    overlay_mask.put_pixel(x, y, Luma([u8::MAX]));
                    masked_coords.push((x, y));
                }
            }
        }

        // Perform 12 Gauss-Seidel harmonic relaxation iterations over the masked pixels.
        for _ in 0..12 {
            for &(x, y) in &masked_coords {
                let mut sum = [0_u32; 3];
                let mut count = 0_u32;
                for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let px = image.get_pixel(nx as u32, ny as u32);
                        sum[0] += u32::from(px[0]);
                        sum[1] += u32::from(px[1]);
                        sum[2] += u32::from(px[2]);
                        count += 1;
                    }
                }
                if count > 0 {
                    image.put_pixel(
                        x,
                        y,
                        Rgb([
                            (sum[0] / count) as u8,
                            (sum[1] / count) as u8,
                            (sum[2] / count) as u8,
                        ]),
                    );
                }
            }
        }
    }
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
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if mask.get_pixel(x, y)[0] >= 127
                || !point_in_polygon(point, &region.polygon)
                || polygon_edge_distance_squared(point, &region.polygon)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
            {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    if channels[0].len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let medians = channels.each_mut().map(|values| median(values));
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

fn polygon_edge_distance_squared(point: (f32, f32), polygon: &[(f32, f32)]) -> f32 {
    let mut minimum = f32::INFINITY;
    let mut start = polygon[polygon.len() - 1];
    for &end in polygon {
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
        minimum = minimum.min(dx * dx + dy * dy);
        start = end;
    }
    minimum
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

    #[test]
    fn feathered_api_compositing_corrects_boundary_luminance_shift_and_preserves_unmasked_regions() {
        let mut base = RgbImage::from_pixel(64, 64, Rgb([240, 240, 240]));
        // Draw dark text inside [28..36, 28..36]
        let mut mask = GrayImage::new(64, 64);
        for y in 28..36 {
            for x in 28..36 {
                base.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        // Simulate a FullPage API response that removed the text (filled with 210 instead of 240,
        // i.e. a -30 gray patch shift across the region) AND accidentally erased a bubble border at (5, 5).
        let mut api_response = RgbImage::from_pixel(64, 64, Rgb([210, 210, 210]));
        base.put_pixel(5, 5, Rgb([10, 10, 10])); // Original bubble border outside mask
        api_response.put_pixel(5, 5, Rgb([210, 210, 210])); // Hallucinated erasure outside mask

        let mut overlay_mask = GrayImage::new(64, 64);
        composite_api_with_feathering(&mut base, &api_response, &mask, &mut overlay_mask, 4, 4);

        // 1. Center of inpainted mask should have its -30 luminance shift corrected back to ~240!
        let center = base.get_pixel(32, 32);
        assert!(
            (i32::from(center[0]) - 240).abs() <= 2,
            "expected luminance shift correction near 240, got {center:?}"
        );
        // 2. Unmasked bubble border at (5, 5) must be 100% untouched.
        assert_eq!(base.get_pixel(5, 5), &Rgb([10, 10, 10]));
        assert_eq!(overlay_mask.get_pixel(5, 5), &Luma([0]));
        assert_eq!(overlay_mask.get_pixel(32, 32), &Luma([255]));
    }

    #[test]
    fn diff_inspector_distinguishes_reconstructed_screentone_from_residual_sfx_strokes() {
        let mut original = RgbImage::new(80, 80);
        let mut reconstructed_screentone = RgbImage::new(80, 80);
        // Background is a high-variance checkerboard screentone
        for y in 0..80 {
            for x in 0..80 {
                let tone = if ((x / 2) + (y / 2)) % 2 == 0 {
                    Rgb([180, 180, 180])
                } else {
                    Rgb([230, 230, 230])
                };
                original.put_pixel(x, y, tone);
                reconstructed_screentone.put_pixel(x, y, tone);
            }
        }
        // Draw sharp black SFX strokes in original inside mask [20..60, 20..60]
        let mut mask = GrayImage::new(80, 80);
        let text_mask = GrayImage::new(80, 80);
        for y in 24..56 {
            for x in 24..56 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
                if (x >= 28 && x <= 32) || (y >= 38 && y <= 42) {
                    original.put_pixel(x, y, Rgb([5, 5, 5]));
                }
            }
        }

        // Case 1: Reconstructed screentone has high pixel variance, but the black SFX strokes are gone.
        let report_clean = inspect_inpaint_diff(
            &original,
            &reconstructed_screentone,
            &mask,
            &text_mask,
            &[],
        );
        assert!(
            report_clean.passed,
            "reconstructed screentone must pass Diff-Inspector despite high variance: {report_clean:?}"
        );

        // Case 2: Model failed to remove the SFX strokes (left original image unchanged).
        let report_residual = inspect_inpaint_diff(&original, &original, &mask, &text_mask, &[]);
        assert!(
            !report_residual.passed,
            "uncleaned SFX strokes must fail Sobel stroke correlation check: {report_residual:?}"
        );
        assert!(has_active_pixels(&report_residual.failed_mask));
    }

    #[test]
    fn diff_inspector_detects_erased_bubble_border() {
        let mut original = RgbImage::from_pixel(80, 80, Rgb([245, 245, 245]));
        let region = rectangle_region([16, 16, 64, 64]);
        // Draw black bubble border along the rectangle perimeter [16..64, 16..64]
        for x in 16..=64 {
            original.put_pixel(x, 16, Rgb([10, 10, 10]));
            original.put_pixel(x, 64, Rgb([10, 10, 10]));
        }
        for y in 16..=64 {
            original.put_pixel(16, y, Rgb([10, 10, 10]));
            original.put_pixel(64, y, Rgb([10, 10, 10]));
        }
        let mut mask = GrayImage::new(80, 80);
        for y in 30..50 {
            for x in 30..50 {
                original.put_pixel(x, y, Rgb([15, 15, 15]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        // Generated image cleaned the text, but ALSO wiped out the bubble border to white (245)
        let erased_border = RgbImage::from_pixel(80, 80, Rgb([245, 245, 245]));
        let report =
            inspect_inpaint_diff(&original, &erased_border, &mask, &mask, &[region]);
        assert!(
            !report.passed,
            "erased bubble contour should fail bubble_border_integrity: {report:?}"
        );
        assert!(report.components[0].bubble_border_integrity < 0.78);
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
            }),
        );

        let prepared = prepare(&input).await.unwrap();
        assert_eq!(prepared.mask.get_pixel(3, 4), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert!(prepared.text_mask.pixels().all(|pixel| pixel[0] == 0));
    }

    #[tokio::test]
    async fn automatic_inpainting_skips_committed_cleanup() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                let cleanup = edit.add_entity(id, At::Start)?;
                edit.set(
                    cleanup,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        Arc::<[u8]>::from([0]),
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
                png: Arc::<[u8]>::from([]),
            }),
        );
        let processor = Processor::new(InpaintingModel::LaMa {}, koharu_ml::Device::cpu()).unwrap();

        assert!(processor.skip(&automatic).unwrap());
        assert!(!processor.skip(&manual).unwrap());
    }

    #[test]
    fn fal_cleanup_respects_full_page_automatic_and_manual_masks() {
        let generated = RgbaImage::from_pixel(2, 1, Rgba([1, 2, 3, 40]));
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(1, 0, Luma([255]));

        let full = cleanup_overlay(&generated, &mask, None, false, true);
        assert_eq!(full.get_pixel(0, 0), &Rgba([1, 2, 3, 255]));
        assert_eq!(full.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));

        let masked = cleanup_overlay(&generated, &mask, None, false, false);
        assert_eq!(masked.get_pixel(0, 0), &Rgba([0, 0, 0, 0]));
        assert_eq!(masked.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));

        let existing = RgbaImage::from_pixel(2, 1, Rgba([9, 8, 7, 255]));
        let manual = cleanup_overlay(&generated, &mask, Some(existing), true, false);
        assert_eq!(manual.get_pixel(0, 0), &Rgba([9, 8, 7, 255]));
        assert_eq!(manual.get_pixel(1, 0), &Rgba([1, 2, 3, 255]));
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

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[rectangle_region([0, 0, 96, 96])],
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(48, 48), &Rgb([240, 241, 242]));
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
                image.put_pixel(x, y, Rgb([250, 240, 220]));
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

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(output.get_pixel(150, 50), &Rgb([250, 240, 220]));
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

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[],
            |tile, tile_mask| {
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
            },
        )
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

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[],
            |tile, _| {
                calls += 1;
                assert!(tile.width() <= TILE_CONTEXT * 2 + 1);
                assert!(tile.height() <= TILE_CONTEXT * 2 + 1);
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }
}
