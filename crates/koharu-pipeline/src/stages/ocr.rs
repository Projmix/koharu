use std::sync::{Arc, Mutex};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{ModelCell, OcrConfig, OcrMethod, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, Rgb};
use koharu_ml::{
    baberu_ocr::BaberuOcr, manga_ocr::MangaOcr, paddle_ocr_vl::PaddleOCRVLTask,
    paddle_ocr_vl_quantized::PaddleOCRVLQuantized,
};
use koharu_scene::{
    Authored, EntityId, Geometry, LanguageTag, OcrAnalysis, Origin, RecognizedFrom, Region,
    RegionSpec, SourceText, TextDirection, TextRegion,
};
use koharu_translator::{Language, OcrRequest, Translator};

const PRODUCER: &str = "dev.koharu.pipeline.ocr";
const OCR_EXCLUSION_CONTAINMENT_THRESHOLD: f64 = 0.5;
const OCR_CROP_PADDING_SCALE: f64 = 1.2;
const API_OCR_MIN_DIMENSION: u32 = 16;

pub(super) struct Processor {
    config: OcrConfig,
    source_language: Language,
    translator: Translator,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(
        config: OcrConfig,
        source_language: Language,
        translator: Translator,
        device: koharu_ml::Device,
    ) -> Self {
        Self {
            config,
            source_language,
            translator,
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, _input: &StageInput) -> String {
        match self.config.method {
            OcrMethod::Local => local_model_name(&self.config.local_model).to_owned(),
            OcrMethod::Api => {
                let provider = Translator::model(&self.config.api.model);
                match self.config.api.model.model.as_deref() {
                    Some(model) => format!("{provider}: {model}"),
                    None => provider.to_owned(),
                }
            }
        }
    }

    fn recovers_from_memory_pressure(&self, _input: &StageInput) -> bool {
        self.config.method == OcrMethod::Local
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self, _input: &StageInput) -> Result<()> {
        match self.config.method {
            OcrMethod::Local => {
                self.model
                    .ensure(|| Model::load(self.device.clone(), &self.config.local_model))
                    .await
            }
            OcrMethod::Api => Ok(()),
        }
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        let page = input.page()?;
        let mut has_target = false;
        for entity in input.scene.descendants(page)? {
            let region = entity.id();
            if !input.contains_entity(region)?
                || !input
                    .scene
                    .component::<Region>(region)?
                    .is_some_and(|value| value.kind == TextRegion::kind())
            {
                continue;
            }
            for relation in input.scene.relations_to_as::<RecognizedFrom>(region) {
                has_target = true;
                let content = relation.value().source;
                let Some(source) = input.scene.component::<SourceText>(content)? else {
                    return Ok(false);
                };
                if matches!(source.text.origin, Origin::User) {
                    continue;
                }
                if source.text.value.trim().is_empty()
                    || !matches!(source.text.origin, Origin::Generated(_))
                {
                    return Ok(false);
                }
            }
        }
        Ok(has_target)
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        match self.config.method {
            OcrMethod::Local => {
                self.model
                    .lock()
                    .await
                    .as_ref()
                    .ok_or_else(|| anyhow!("OCR model is not loaded"))?
                    .run(input, self.source_language)
                    .await
            }
            OcrMethod::Api => {
                run_api(&self.translator, &self.config, self.source_language, input).await
            }
        }
    }
}

fn local_model_name(model: &OcrModel) -> &'static str {
    match model {
        OcrModel::MangaOcr => "manga-ocr",
        OcrModel::BaberuOcr => "baberu-ocr",
        OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
    }
}

enum Model {
    Manga(Arc<Mutex<MangaOcr>>),
    Baberu(Arc<Mutex<BaberuOcr>>),
    Paddle(Arc<Mutex<PaddleOCRVLQuantized>>),
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &OcrModel) -> Result<Self> {
        match config {
            OcrModel::MangaOcr => Ok(Self::Manga(Arc::new(Mutex::new(
                MangaOcr::load(device).await?,
            )))),
            OcrModel::BaberuOcr => Ok(Self::Baberu(Arc::new(Mutex::new(
                BaberuOcr::load(device).await?,
            )))),
            OcrModel::PaddleOcrVl1_6 => Ok(Self::Paddle(Arc::new(Mutex::new(
                PaddleOCRVLQuantized::load(device).await?,
            )))),
        }
    }

    async fn run(
        &self,
        input: StageInput,
        source_language: Language,
    ) -> Result<koharu_scene::Patch> {
        let model_name = match self {
            Self::Manga(_) => "manga-ocr",
            Self::Baberu(_) => "baberu-ocr",
            Self::Paddle(_) => "paddleocr-vl-1.6",
        };
        let (page, targets) = targets(&input).await?;

        let results = match self {
            Self::Manga(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Baberu(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Paddle(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    Ok(model.inference(image, PaddleOCRVLTask::Ocr)?.text)
                })
                .await?
            }
        };

        apply_results(input, page, model_name, source_language, results)
    }
}

async fn run_api(
    translator: &Translator,
    config: &OcrConfig,
    source_language: Language,
    input: StageInput,
) -> Result<koharu_scene::Patch> {
    let (page, targets) = targets(&input).await?;
    let model_name = config
        .api
        .model
        .model
        .as_deref()
        .unwrap_or_else(|| Translator::model(&config.api.model))
        .to_owned();
    let mut results = Vec::with_capacity(targets.len());
    for target in targets {
        let OcrTarget {
            content,
            region,
            geometry,
            image,
        } = target;
        let request = OcrRequest {
            image: Arc::new(prepare_api_image(image)),
            source_language: Some(source_language),
            instructions: config.api.instructions.clone(),
        };
        let recognized = tokio::select! {
            biased;
            _ = input.stop.cancelled() => return finish(input.scene.edit()),
            result = translator.recognize(
                &config.api.model,
                config.api.generation.clone(),
                request,
            ) => result?,
        };
        results.push(OcrResult {
            content,
            region,
            geometry,
            text: normalize_ocr_text(recognized.1),
        });
    }
    apply_results(input, page, &model_name, source_language, results)
}

/// Keep provider-facing OCR images above the minimum side length accepted by
/// vision endpoints while preserving the detector crop pixel-for-pixel.
fn prepare_api_image(image: DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    if width >= API_OCR_MIN_DIMENSION && height >= API_OCR_MIN_DIMENSION {
        return image;
    }

    let target_width = width.max(API_OCR_MIN_DIMENSION);
    let target_height = height.max(API_OCR_MIN_DIMENSION);
    let mut canvas = image::RgbImage::from_pixel(target_width, target_height, Rgb([u8::MAX; 3]));
    let source = image.to_rgb8();
    let offset_x = i64::from((target_width - width) / 2);
    let offset_y = i64::from((target_height - height) / 2);
    image::imageops::replace(&mut canvas, &source, offset_x, offset_y);
    DynamicImage::ImageRgb8(canvas)
}

async fn targets(input: &StageInput) -> Result<(EntityId, Vec<OcrTarget>)> {
    let page = input.page()?;
    let mut targets = Vec::new();
    let source = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let mut regions = Vec::new();
    for entity in input.scene.descendants(page)? {
        let region = entity.id();
        if !input.contains_entity(region)? {
            continue;
        }
        let is_text_region = input
            .scene
            .component::<Region>(region)?
            .is_some_and(|value| value.kind == TextRegion::kind());
        if !is_text_region {
            continue;
        }
        let geometry = input
            .scene
            .component::<Geometry>(region)?
            .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
        regions.push((region, geometry));
    }
    for (region, geometry) in &regions {
        let exclusions = regions
            .iter()
            .filter_map(|(candidate, child)| {
                (*candidate != *region && nested_geometry(geometry, child)).then_some(child)
            })
            .collect::<Vec<_>>();
        let crop = crop(&source, geometry, &exclusions)
            .with_context(|| format!("text region {region} is outside its source image"))?;
        for relation in input.scene.relations_to_as::<RecognizedFrom>(*region) {
            let content = relation.value().source;
            let previous = input.scene.component::<SourceText>(content)?;
            if previous
                .as_ref()
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            targets.push(OcrTarget {
                content,
                region: *region,
                geometry: geometry.clone(),
                image: crop.clone(),
            });
        }
    }
    Ok((page, targets))
}

fn apply_results(
    input: StageInput,
    page: EntityId,
    model_name: &str,
    source_language: Language,
    results: Vec<OcrResult>,
) -> Result<koharu_scene::Patch> {
    let generation = generation(PRODUCER, model_name)?;
    let mut edit = input.scene.edit_as(generation.clone());
    edit.observe_assets(page)?;
    for result in &results {
        edit.observe::<Region>(result.region)?;
        edit.observe::<Geometry>(result.region)?;
        edit.observe::<SourceText>(result.content)?;
    }
    for result in results {
        let language = Some(LanguageTag::new(source_language.tag())?);
        edit.set(
            result.content,
            &SourceText {
                text: Authored::generated(result.text, generation.clone()),
                language,
            },
        )?;
        let (min_x, min_y, max_x, max_y) = geometry_extents(&result.geometry)
            .ok_or_else(|| anyhow!("text region {} has empty geometry", result.region))?;
        edit.set(
            result.region,
            &OcrAnalysis {
                origin: Origin::Generated(generation.clone()),
                direction: if max_y - min_y >= (max_x - min_x) * 1.15 {
                    TextDirection::Vertical
                } else {
                    TextDirection::Horizontal
                },
                confidence: None,
                line_boundaries: Vec::new(),
            },
        )?;
    }
    finish(edit)
}

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    image: DynamicImage,
}

struct OcrResult {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    text: String,
}

async fn infer_text<M: Send + 'static>(
    model: Arc<Mutex<M>>,
    targets: Vec<OcrTarget>,
    inference: impl Fn(&M, &DynamicImage) -> Result<String> + Send + Sync + 'static,
) -> Result<Vec<OcrResult>> {
    tokio::task::spawn_blocking(move || {
        let model = model
            .lock()
            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
        targets
            .into_iter()
            .map(|target| {
                Ok(OcrResult {
                    content: target.content,
                    region: target.region,
                    geometry: target.geometry,
                    text: normalize_ocr_text(inference(&model, &target.image)?),
                })
            })
            .collect()
    })
    .await
    .context("OCR task panicked")?
}

// Manga OCR can emit replacement-box glyphs for an isolated Japanese ellipsis.
// Normalize only an all-placeholder sequence so ordinary OCR output is preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible
            .iter()
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | '�'))
    {
        "…".to_owned()
    } else {
        text
    }
}

fn crop(
    source: &DynamicImage,
    geometry: &Geometry,
    exclusions: &[&Geometry],
) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let half_width = (max_x - min_x) * 0.5 * OCR_CROP_PADDING_SCALE;
    let half_height = (max_y - min_y) * 0.5 * OCR_CROP_PADDING_SCALE;
    let x = (center_x - half_width).floor().max(0.0) as u32;
    let y = (center_y - half_height).floor().max(0.0) as u32;
    let right = (center_x + half_width)
        .ceil()
        .max(0.0)
        .min(f64::from(source.width())) as u32;
    let bottom = (center_y + half_height)
        .ceil()
        .max(0.0)
        .min(f64::from(source.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the image");
    }
    let mut crop = source.crop_imm(x, y, right - x, bottom - y).to_rgb8();
    for local_y in 0..crop.height() {
        for local_x in 0..crop.width() {
            let sample_x = f64::from(x + local_x) + 0.5;
            let sample_y = f64::from(y + local_y) + 0.5;
            let geometry_x = center_x + (sample_x - center_x) / OCR_CROP_PADDING_SCALE;
            let geometry_y = center_y + (sample_y - center_y) / OCR_CROP_PADDING_SCALE;
            if !point_in_geometry(geometry_x, geometry_y, geometry) {
                crop.put_pixel(local_x, local_y, Rgb([u8::MAX; 3]));
            }
        }
    }
    for exclusion in exclusions {
        let Some((child_left, child_top, child_right, child_bottom)) = geometry_extents(exclusion)
        else {
            continue;
        };
        let local_left = child_left.floor().max(f64::from(x)) as u32 - x;
        let local_top = child_top.floor().max(f64::from(y)) as u32 - y;
        let local_right = child_right.ceil().min(f64::from(right)).max(f64::from(x)) as u32 - x;
        let local_bottom = child_bottom.ceil().min(f64::from(bottom)).max(f64::from(y)) as u32 - y;
        for local_y in local_top..local_bottom {
            for local_x in local_left..local_right {
                if point_in_geometry(
                    f64::from(x + local_x) + 0.5,
                    f64::from(y + local_y) + 0.5,
                    exclusion,
                ) {
                    crop.put_pixel(local_x, local_y, Rgb([u8::MAX; 3]));
                }
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(crop))
}

fn nested_geometry(parent: &Geometry, child: &Geometry) -> bool {
    let Some((parent_left, parent_top, parent_right, parent_bottom)) = geometry_extents(parent)
    else {
        return false;
    };
    let Some((child_left, child_top, child_right, child_bottom)) = geometry_extents(child) else {
        return false;
    };
    let parent_area = polygon_area(parent);
    let child_area = polygon_area(child);
    if child_area <= 0.0 || child_area >= parent_area {
        return false;
    }
    let bounds_intersection = (parent_right.min(child_right) - parent_left.max(child_left))
        .max(0.0)
        * (parent_bottom.min(child_bottom) - parent_top.max(child_top)).max(0.0);
    if bounds_intersection <= 0.0 {
        return false;
    }

    // Geometry polygons can be rotated or concave, so AABB containment is only
    // an inexpensive no-overlap check. Estimate the child's actual area inside
    // the parent on a pixel-aligned supersampled raster instead.
    let columns = (child_right - child_left).ceil().clamp(4.0, 256.0) as usize;
    let rows = (child_bottom - child_top).ceil().clamp(4.0, 256.0) as usize;
    let mut child_samples = 0usize;
    let mut contained_samples = 0usize;
    for row in 0..rows {
        let y = child_top + (row as f64 + 0.5) * (child_bottom - child_top) / rows as f64;
        for column in 0..columns {
            let x =
                child_left + (column as f64 + 0.5) * (child_right - child_left) / columns as f64;
            if point_in_geometry(x, y, child) {
                child_samples += 1;
                contained_samples += usize::from(point_in_geometry(x, y, parent));
            }
        }
    }
    child_samples > 0
        && contained_samples as f64 / child_samples as f64 >= OCR_EXCLUSION_CONTAINMENT_THRESHOLD
}

fn polygon_area(geometry: &Geometry) -> f64 {
    if geometry.points.len() < 3 {
        return 0.0;
    }
    geometry
        .points
        .iter()
        .zip(geometry.points.iter().cycle().skip(1))
        .take(geometry.points.len())
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>()
        .abs()
        * 0.5
}

fn point_in_geometry(x: f64, y: f64, geometry: &Geometry) -> bool {
    if geometry.points.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = geometry.points.last().expect("geometry is non-empty");
    for point in &geometry.points {
        if (point.y > y) != (previous.y > y)
            && x < (previous.x - point.x) * (y - point.y) / (previous.y - point.y) + point.x
        {
            inside = !inside;
        }
        previous = point;
    }
    inside
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use koharu_scene::{At, PageDraft, TextRegion};

    use super::*;
    use crate::ImageCache;

    #[tokio::test]
    async fn completed_ocr_targets_are_skipped() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let generated = generation(PRODUCER, "test-ocr").unwrap();
        let mut generated_content = None;
        for (text, user_owned) in [("generated", false), ("user", true)] {
            let region = edit
                .add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
                    None,
                )
                .unwrap();
            let content = edit.add_text_content(page, At::End).unwrap();
            if user_owned {
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::user(text.to_owned()),
                        language: None,
                    },
                )
                .unwrap();
            } else {
                generated_content = Some(content);
            }
            edit.relate::<RecognizedFrom>(content, region).unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        let mut edit = session.snapshot().edit_as(generated.clone());
        edit.set(
            generated_content.unwrap(),
            &SourceText {
                text: Authored::generated("generated".to_owned(), generated),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert!(processor().skip(&input(&session, page)).unwrap());
    }

    #[tokio::test]
    async fn missing_or_empty_generated_ocr_targets_are_not_skipped() {
        for text in [None, Some("")] {
            let mut session = koharu_scene::Session::memory().await.unwrap();
            let mut edit = session.snapshot().edit();
            let page = edit
                .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
                .unwrap();
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
            session.commit(edit.finish().unwrap()).await.unwrap();
            if let Some(text) = text {
                let generated = generation(PRODUCER, "test-ocr").unwrap();
                let mut edit = session.snapshot().edit_as(generated.clone());
                edit.set(
                    content,
                    &SourceText {
                        text: Authored::generated(text.to_owned(), generated),
                        language: None,
                    },
                )
                .unwrap();
                session.commit(edit.finish().unwrap()).await.unwrap();
            }

            assert!(
                !processor().skip(&input(&session, page)).unwrap(),
                "OCR target with {text:?} text was skipped"
            );
        }
    }

    #[tokio::test]
    async fn orphaned_generated_text_region_is_not_skipped() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let owner = generation("dev.koharu.pipeline.detection", "test-detection").unwrap();
        let mut edit = session.snapshot().edit_as(owner);
        edit.add_analysis_region::<TextRegion>(
            page,
            At::End,
            &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
            None,
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert!(!processor().skip(&input(&session, page)).unwrap());
    }

    fn processor() -> Processor {
        let translator = Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        Processor::new(
            OcrConfig::default(),
            Language::Japanese,
            translator,
            koharu_ml::Device::cpu(),
        )
    }

    fn input(session: &koharu_scene::Session, page: EntityId) -> StageInput {
        StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        )
    }

    #[test]
    fn repeated_placeholder_glyphs_are_an_ellipsis() {
        assert_eq!(normalize_ocr_text("☐ ☐ ☐".to_owned()), "…");
        assert_eq!(normalize_ocr_text("□\n□".to_owned()), "…");
    }

    #[test]
    fn ordinary_text_and_single_boxes_are_unchanged() {
        assert_eq!(normalize_ocr_text("待って…".to_owned()), "待って…");
        assert_eq!(normalize_ocr_text("☐".to_owned()), "☐");
    }

    #[test]
    fn api_ocr_image_meets_provider_minimum_dimensions_without_distortion() {
        let source = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(7, 32, Rgb([10, 20, 30])));

        let prepared = prepare_api_image(source).to_rgb8();

        assert_eq!(prepared.dimensions(), (16, 32));
        assert_eq!(prepared.get_pixel(4, 0), &Rgb([10, 20, 30]));
        assert_eq!(prepared.get_pixel(3, 0), &Rgb([u8::MAX; 3]));
    }

    #[test]
    fn ocr_crop_includes_ten_percent_context_on_each_side() {
        let source =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(100, 100, Rgb([10, 20, 30])));
        let region = Geometry::rectangle(20.0, 20.0, 40.0, 20.0);

        let crop = crop(&source, &region, &[]).unwrap().to_rgb8();

        assert_eq!(crop.dimensions(), (48, 24));
        assert_eq!(crop.get_pixel(0, 12), &Rgb([10, 20, 30]));
        assert_eq!(crop.get_pixel(47, 12), &Rgb([10, 20, 30]));
    }

    #[test]
    fn ocr_crop_context_is_clamped_to_the_page_edges() {
        let source =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(100, 100, Rgb([10, 20, 30])));
        let region = Geometry::rectangle(0.0, 0.0, 20.0, 20.0);

        let crop = crop(&source, &region, &[]).unwrap().to_rgb8();

        assert_eq!(crop.dimensions(), (22, 22));
        assert_eq!(crop.get_pixel(0, 0), &Rgb([10, 20, 30]));
    }

    #[test]
    fn nested_text_region_is_removed_from_the_parent_ocr_crop() {
        let source = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(8, 8, Rgb([10, 20, 30])));
        let parent = Geometry::rectangle(0.0, 0.0, 8.0, 8.0);
        let child = Geometry::rectangle(2.0, 3.0, 3.0, 2.0);

        assert!(nested_geometry(&parent, &child));
        let crop = crop(&source, &parent, &[&child]).unwrap().to_rgb8();

        assert_eq!(crop.get_pixel(0, 0), &Rgb([10, 20, 30]));
        for y in 3..5 {
            for x in 2..5 {
                assert_eq!(crop.get_pixel(x, y), &Rgb([u8::MAX; 3]));
            }
        }
    }

    #[test]
    fn nested_ocr_crop_masks_the_rotated_parent_exterior() {
        let source =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(10, 10, Rgb([10, 20, 30])));
        let parent = Geometry {
            origin: Origin::User,
            points: vec![
                koharu_scene::Point { x: 1.0, y: 5.0 },
                koharu_scene::Point { x: 5.0, y: 1.0 },
                koharu_scene::Point { x: 9.0, y: 5.0 },
                koharu_scene::Point { x: 5.0, y: 9.0 },
            ],
        };
        let child = Geometry::rectangle(4.0, 4.0, 2.0, 2.0);

        let crop = crop(&source, &parent, &[&child]).unwrap().to_rgb8();

        assert_eq!(crop.get_pixel(0, 0), &Rgb([u8::MAX; 3]));
        assert_eq!(crop.get_pixel(4, 1), &Rgb([10, 20, 30]));
        assert_eq!(crop.get_pixel(4, 4), &Rgb([u8::MAX; 3]));
    }

    #[test]
    fn nested_ocr_crop_accepts_the_real_page_partial_overlap() {
        let parent = Geometry {
            origin: Origin::User,
            points: vec![
                koharu_scene::Point {
                    x: 144.80012678487373,
                    y: 1347.9436565928552,
                },
                koharu_scene::Point {
                    x: 404.50297714053954,
                    y: 1162.6994137210652,
                },
                koharu_scene::Point {
                    x: 600.1998732151262,
                    y: 1437.0563434071448,
                },
                koharu_scene::Point {
                    x: 340.49702285946046,
                    y: 1622.3005862789348,
                },
            ],
        };
        let child = Geometry {
            origin: Origin::User,
            points: vec![
                koharu_scene::Point {
                    x: 177.3230770356867,
                    y: 1268.509218177469,
                },
                koharu_scene::Point {
                    x: 305.66989205528574,
                    y: 1186.7432792591217,
                },
                koharu_scene::Point {
                    x: 363.9466983549383,
                    y: 1278.219541588156,
                },
                koharu_scene::Point {
                    x: 235.5998833353392,
                    y: 1359.9854805065033,
                },
            ],
        };

        assert!(nested_geometry(&parent, &child));
    }

    #[test]
    fn crossing_text_region_is_not_treated_as_a_nested_exclusion() {
        let parent = Geometry::rectangle(0.0, 0.0, 4.0, 4.0);
        let crossing = Geometry::rectangle(3.0, 1.0, 3.0, 2.0);
        let same = Geometry::rectangle(0.0, 0.0, 4.0, 4.0);

        assert!(!nested_geometry(&parent, &crossing));
        assert!(!nested_geometry(&parent, &same));
    }

    #[test]
    fn overlapping_polygons_with_contained_bounds_are_not_nested() {
        let parent = Geometry {
            origin: Origin::User,
            points: vec![
                koharu_scene::Point { x: 0.0, y: 5.0 },
                koharu_scene::Point { x: 5.0, y: 0.0 },
                koharu_scene::Point { x: 10.0, y: 5.0 },
                koharu_scene::Point { x: 5.0, y: 10.0 },
            ],
        };
        let crossing = Geometry::rectangle(0.0, 0.0, 4.0, 4.0);

        assert!(!nested_geometry(&parent, &crossing));
    }

    #[test]
    fn almost_contained_real_page_region_is_a_nested_exclusion() {
        let parent = Geometry::rectangle(394.544, 925.362, 955.329 - 394.544, 1119.854 - 925.362);
        let child = Geometry::rectangle(384.731, 923.990, 593.175 - 384.731, 1119.575 - 923.990);

        assert!(nested_geometry(&parent, &child));
    }
}
