//! Scan enhancement: removes compression artifacts and grey paper or washed
//! ink from a low-quality scan before detection reads the page.
//!
//! The imported source stays untouched. The result is one opaque full-page
//! raster layer at the bottom of the page, owned by this producer, so hiding
//! it compares before and after and a rerun replaces it. Later stages read
//! that layer instead of the source while it is visible
//! (`ImageCache::page_image`). A result that fails the checks in `scan` never
//! replaces the source: the page keeps its original and the run reports why.

mod scan;

use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, ImageFormat};
use koharu_ml::manga_jpeg::{MangaJpeg, MangaJpegVariant};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, EntityId, Origin, RasterLayer, RasterLayerKind,
    RemovePolicy, Snapshot, Visibility,
};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{EnhancementConfig, ModelCell};

const PRODUCER: &str = "dev.koharu.pipeline.enhancement";
/// Bumped whenever the analysis, filters or checks change what a page gets.
const REVISION: &str = "1";
const LAYER_NAME: &str = "Scan enhancement";

/// The page's scan enhancement layer, when it has one.
pub(crate) fn enhancement_layer(scene: &Snapshot, page: EntityId) -> Result<Option<EntityId>> {
    for child in scene.children(page)? {
        if scene
            .component::<RasterLayer>(child)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Enhancement)
        {
            return Ok(Some(child));
        }
    }
    Ok(None)
}

/// The enhancement layer later stages read instead of the source: present
/// and visible.
pub(crate) fn visible_enhancement_layer(
    scene: &Snapshot,
    page: EntityId,
) -> Result<Option<EntityId>> {
    let Some(layer) = enhancement_layer(scene, page)? else {
        return Ok(None);
    };
    let visible = scene
        .component::<Visibility>(layer)?
        .is_none_or(|visibility| visibility.visible && visibility.opacity > 0.0);
    Ok(visible.then_some(layer))
}

pub(super) struct Processor {
    config: EnhancementConfig,
    device: koharu_ml::Device,
    model: ModelCell<Arc<Mutex<MangaJpeg>>>,
}

impl Processor {
    pub(super) fn new(config: EnhancementConfig, device: koharu_ml::Device) -> Self {
        Self {
            config,
            device,
            model: ModelCell::new(),
        }
    }

    fn model_id(&self) -> String {
        let mut parts = vec![if self.config.mangajpeg {
            "mangajpeg"
        } else {
            "classical"
        }];
        if self.config.tone {
            parts.push("tone");
        }
        format!("{}#{REVISION}", parts.join("+"))
    }

    /// MangaJPEG for `variant`, loaded on first use. One variant is resident
    /// at a time; a page of another quality swaps it.
    async fn restore(
        &self,
        variant: MangaJpegVariant,
        luma: GrayImage,
        input: &StageInput,
    ) -> Result<Option<GrayImage>> {
        let network = {
            let mut model = self.model.lock().await;
            let loaded = model.as_ref().and_then(|model| {
                model
                    .lock()
                    .ok()
                    .map(|network| network.variant() == variant)
            });
            if loaded != Some(true) {
                *model = None;
                let network = MangaJpeg::load(variant, self.device.clone()).await?;
                *model = Some(Arc::new(Mutex::new(network)));
            }
            model.clone().expect("the model was loaded above")
        };
        let stop = input.stop.clone();
        tokio_rayon::spawn(move || {
            network
                .lock()
                .map_err(|_| anyhow!("MangaJPEG lock is poisoned"))?
                .inference(&luma, &|| stop.stopped())
        })
        .await
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, _input: &StageInput) -> String {
        self.model_id()
    }

    fn recovers_from_memory_pressure(&self, _input: &StageInput) -> bool {
        self.config.mangajpeg
    }

    /// A page keeps a layer this configuration already produced, and a layer
    /// the user took over is never replaced.
    fn skip(&self, input: &StageInput) -> Result<bool> {
        let page = input.page()?;
        let Some(layer) = enhancement_layer(&input.scene, page)? else {
            return Ok(false);
        };
        Ok(match input.scene.component::<RasterLayer>(layer)? {
            Some(RasterLayer {
                origin: Origin::Generated(owner),
                ..
            }) => {
                owner.producer.as_str() == PRODUCER
                    && owner.model.as_deref() == Some(self.model_id().as_str())
            }
            _ => true,
        })
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self, _input: &StageInput) -> Result<()> {
        // The MangaJPEG variant depends on the page's measured quality, so
        // `process` loads it once the page is analysed.
        Ok(())
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let page = input.page()?;
        let source_role = AssetRole::new("source")?;
        let asset = input
            .scene
            .asset(page, &source_role)?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let encoded = input.scene.read_blob(asset.blob).await?;
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;

        let (rgb, before, analysis) = tokio_rayon::spawn(move || {
            let rgb = source.to_rgb8();
            let before = scan::Plane::luma(&rgb);
            let analysis = scan::Analysis::new(&rgb, &before, &encoded);
            (rgb, before, analysis)
        })
        .await;
        let rgb = Arc::new(rgb);
        let before = Arc::new(before);

        let mut after = (*before).clone();
        let mut method = None;
        let mut rejected = Vec::new();
        if let Some(quality) = analysis.damage() {
            let mut candidates = Vec::new();
            if self.config.mangajpeg {
                let variant = MangaJpegVariant::for_quality(quality);
                let Some(restored) = self.restore(variant, before.to_gray(), &input).await? else {
                    return Ok(input.scene.edit().finish()?);
                };
                candidates.push((variant.name(), scan::Plane::from_gray(&restored)));
            }
            let classical = {
                let before = before.clone();
                tokio_rayon::spawn(move || scan::surface_blur(&before)).await
            };
            candidates.push(("Surface Blur", classical));
            for (name, candidate) in candidates {
                let checks = {
                    let (before, candidate) = (before.clone(), candidate.clone());
                    tokio_rayon::spawn(move || scan::Checks::new(&before, &candidate)).await
                };
                tracing::debug!(%page, quality, name, ?checks, "scan enhancement candidate");
                if let Some(reason) = checks.rejection() {
                    rejected.push(format!("{name}: {reason}"));
                } else if checks.improves() {
                    after = candidate;
                    method = Some(format!("{name} for JPEG quality {quality}"));
                    break;
                }
            }
        }
        let tone = if self.config.tone && analysis.monochrome {
            let (rgb, before, after) = (rgb.clone(), before.clone(), after.clone());
            tokio_rayon::spawn(move || scan::tone(&rgb, &before, &after)).await
        } else {
            None
        };

        let mut edit = input.scene.edit_as(generation(PRODUCER, &self.model_id())?);
        edit.observe_assets(page)?;
        edit.observe_subtree(page)?;
        if let Some(layer) = enhancement_layer(&input.scene, page)? {
            edit.remove_entity(layer, RemovePolicy::Cascade)?;
        }
        if method.is_none() && tone.is_none() {
            if !rejected.is_empty() {
                bail!("kept the original page: {}", rejected.join("; "));
            }
            // Nothing to fix. A layer from another configuration goes, so the
            // page shows what this run decided.
            return finish(edit);
        }
        tracing::info!(%page, ?analysis, method, ?tone, "enhanced scan");

        let png = tokio_rayon::spawn(move || -> Result<(Vec<u8>, u32, u32)> {
            let image = scan::compose(&rgb, analysis, &before, &after, tone);
            let (width, height) = (image.width(), image.height());
            let mut bytes = Cursor::new(Vec::new());
            match image {
                DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_) => {
                    image.write_to(&mut bytes, ImageFormat::Png)?;
                }
                _ => unreachable!("compose returns 8-bit grey or RGB"),
            }
            Ok((bytes.into_inner(), width, height))
        })
        .await?;
        let (bytes, width, height) = png;
        let layer = edit.add_entity(page, At::Start)?;
        edit.set(
            layer,
            &RasterLayer {
                origin: Origin::Generated(generation(PRODUCER, &self.model_id())?),
                name: LAYER_NAME.to_owned(),
                kind: RasterLayerKind::Enhancement,
            },
        )?;
        edit.set_asset(
            layer,
            &source_role,
            AssetInput::new(
                Arc::<[u8]>::from(bytes),
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
