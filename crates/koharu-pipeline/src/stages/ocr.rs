use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use super::{
    StageInput, StageProcessor, finish, generation,
    jev::{
        JevClient, OcrJudgeInput, OcrReadingOrderChoice, OcrRetryStrategyChoice,
    },
};
use crate::{ModelCell, OcrConfig, OcrMethod, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use imageproc::region_labelling::{Connectivity, connected_components};
use koharu_ml::{
    baberu_ocr::BaberuOcr, hayai_ocr::HayaiOcr, manga_ocr::MangaOcr,
    paddle_ocr_vl::PaddleOCRVLTask, paddle_ocr_vl_quantized::PaddleOCRVLQuantized,
};
use koharu_scene::{
    Authored, BubbleRegion, EntityId, FlowsIn, Geometry, Inside, LanguageTag, OcrAnalysis, Origin,
    Presents, RecognizedFrom, Region, RegionSpec, SourceText, TextDirection, TextRegion,
};
use koharu_translator::{Language, OcrRequest, Translator};

const PRODUCER: &str = "dev.koharu.pipeline.ocr";
const OCR_EXCLUSION_CONTAINMENT_THRESHOLD: f64 = 0.5;
const OCR_CROP_PADDING_SCALE: f64 = 1.2;
const OCR_RETRY_CROP_PADDING_SCALE: f64 = 1.4;
const API_OCR_MIN_DIMENSION: u32 = 16;

pub(super) struct Processor {
    config: OcrConfig,
    source_language: Language,
    translator: Translator,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
    fallback_model: ModelCell<Model>,
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
            fallback_model: ModelCell::new(),
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
        let primary = self.model.unload();
        let fallback = self.fallback_model.unload();
        primary || fallback
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
                let primary = self
                    .model
                    .lock()
                    .await
                    .as_ref()
                    .ok_or_else(|| anyhow!("OCR model is not loaded"))?
                    .clone();
                primary
                    .run(
                        input,
                        self.source_language,
                        &self.device,
                        &self.config.local_model,
                        &self.fallback_model,
                    )
                    .await
            }
            OcrMethod::Api => {
                run_api(
                    &self.translator,
                    &self.config,
                    self.source_language,
                    &self.device,
                    &self.fallback_model,
                    input,
                )
                .await
            }
        }
    }
}

fn local_model_name(model: &OcrModel) -> &'static str {
    match model {
        OcrModel::MangaOcr => "manga-ocr",
        OcrModel::BaberuOcr => "baberu-ocr",
        OcrModel::HayaiOcr => "hayai-ocr",
        OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
    }
}

fn alternate_local_ocr_model(model: &OcrModel) -> OcrModel {
    match model {
        OcrModel::PaddleOcrVl1_6 => OcrModel::MangaOcr,
        OcrModel::MangaOcr => OcrModel::BaberuOcr,
        OcrModel::BaberuOcr => OcrModel::MangaOcr,
        OcrModel::HayaiOcr => OcrModel::MangaOcr,
    }
}

#[derive(Clone)]
enum Model {
    Manga(Arc<Mutex<MangaOcr>>),
    Baberu(Arc<Mutex<BaberuOcr>>),
    Hayai(Arc<Mutex<HayaiOcr>>),
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
            OcrModel::HayaiOcr => Ok(Self::Hayai(Arc::new(Mutex::new(
                HayaiOcr::load(device).await?,
            )))),
            OcrModel::PaddleOcrVl1_6 => Ok(Self::Paddle(Arc::new(Mutex::new(
                PaddleOCRVLQuantized::load(device).await?,
            )))),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Manga(_) => "manga-ocr",
            Self::Baberu(_) => "baberu-ocr",
            Self::Hayai(_) => "hayai-ocr",
            Self::Paddle(_) => "paddleocr-vl-1.6",
        }
    }

    async fn infer_single(&self, image: DynamicImage) -> Result<String> {
        let model = self.clone();
        tokio::task::spawn_blocking(move || model.infer_blocking(&image))
            .await
            .context("OCR single inference panicked")?
    }

    fn infer_blocking(&self, image: &DynamicImage) -> Result<String> {
        let raw = match self {
            Self::Manga(model) => model
                .lock()
                .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                .inference(image)?,
            Self::Baberu(model) => model
                .lock()
                .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                .inference(image)?,
            Self::Hayai(model) => model
                .lock()
                .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                .inference(image)?,
            Self::Paddle(model) => {
                model
                    .lock()
                    .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                    .inference(image, PaddleOCRVLTask::Ocr)?
                    .text
            }
        };
        Ok(normalize_ocr_text(raw))
    }

    async fn run(
        &self,
        input: StageInput,
        source_language: Language,
        device: &koharu_ml::Device,
        configured_model: &OcrModel,
        fallback_cell: &ModelCell<Model>,
    ) -> Result<koharu_scene::Patch> {
        let model_name = self.name();
        let (page, targets) = targets(&input, source_language).await?;
        let model_clone = self.clone();
        let initial_candidates = tokio::task::spawn_blocking(move || {
            targets
                .into_iter()
                .map(|target| recognize_target_blocking(&model_clone, target, source_language))
                .collect::<Result<Vec<RecognizedTarget>>>()
        })
        .await
        .context("OCR task panicked")??;

        let validated = validate_and_recover_targets(
            &input,
            initial_candidates,
            source_language,
            |crop_img| {
                let model = self.clone();
                async move { model.infer_single(crop_img).await }
            },
            device,
            Some(configured_model),
            fallback_cell,
        )
        .await?;

        apply_results(input, page, model_name, source_language, validated)
    }
}

async fn run_api(
    translator: &Translator,
    config: &OcrConfig,
    source_language: Language,
    device: &koharu_ml::Device,
    fallback_cell: &ModelCell<Model>,
    input: StageInput,
) -> Result<koharu_scene::Patch> {
    let (page, targets) = targets(&input, source_language).await?;
    let model_name = config
        .api
        .model
        .model
        .as_deref()
        .unwrap_or_else(|| Translator::model(&config.api.model))
        .to_owned();
    let mut initial_candidates = Vec::with_capacity(targets.len());
    for target in targets {
        let primary_text = recognize_api_image(
            translator,
            config,
            source_language,
            &input,
            target.image.clone(),
        )
        .await?;
        let Some(primary_text) = primary_text else {
            return finish(input.scene.edit());
        };

        let mut col_texts_ltr = Vec::new();
        if source_language == Language::Japanese && target.column_crops.len() >= 2 {
            for col_crop in &target.column_crops {
                let Some(col_text) = recognize_api_image(
                    translator,
                    config,
                    source_language,
                    &input,
                    col_crop.clone(),
                )
                .await?
                else {
                    return finish(input.scene.edit());
                };
                if !col_text.trim().is_empty() {
                    col_texts_ltr.push(col_text);
                }
            }
        }

        initial_candidates.push(build_recognized_target(
            target,
            primary_text,
            col_texts_ltr,
            source_language,
        ));
    }

    let validated = validate_and_recover_targets(
        &input,
        initial_candidates,
        source_language,
        |crop_img| {
            let translator = translator.clone();
            let config = config.clone();
            let stop = input.stop.clone();
            async move {
                let request = OcrRequest {
                    image: Arc::new(prepare_api_image(crop_img)),
                    source_language: Some(source_language),
                    instructions: config.api.instructions.clone(),
                };
                tokio::select! {
                    biased;
                    _ = stop.cancelled() => bail!("OCR cancelled"),
                    result = translator.recognize(
                        &config.api.model,
                        config.api.generation.clone(),
                        request,
                    ) => Ok(normalize_ocr_text(result?.1)),
                }
            }
        },
        device,
        None,
        fallback_cell,
    )
    .await?;

    apply_results(input, page, &model_name, source_language, validated)
}

async fn recognize_api_image(
    translator: &Translator,
    config: &OcrConfig,
    source_language: Language,
    input: &StageInput,
    image: DynamicImage,
) -> Result<Option<String>> {
    let request = OcrRequest {
        image: Arc::new(prepare_api_image(image)),
        source_language: Some(source_language),
        instructions: config.api.instructions.clone(),
    };
    let recognized = tokio::select! {
        biased;
        _ = input.stop.cancelled() => return Ok(None),
        result = translator.recognize(
            &config.api.model,
            config.api.generation.clone(),
            request,
        ) => result?,
    };
    Ok(Some(normalize_ocr_text(recognized.1)))
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

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    bubble_id: Option<EntityId>,
    geometry: Geometry,
    exclusions: Vec<Geometry>,
    image: DynamicImage,
    crop_mask: GrayImage,
    direction: TextDirection,
    column_spans: Vec<(u32, u32)>,
    column_crops: Vec<DynamicImage>,
}

struct RecognizedTarget {
    target: OcrTarget,
    primary_text: String,
    candidate_rtl: String,
    candidate_ltr: Option<String>,
    rtl_columns: Vec<String>,
    ltr_columns: Vec<String>,
    rtl_inspection: JapaneseOcrInspection,
    ltr_inspection: Option<JapaneseOcrInspection>,
}

struct OcrResult {
    content: EntityId,
    region: EntityId,
    bubble_id: Option<EntityId>,
    geometry: Geometry,
    direction: TextDirection,
    confidence: Option<f32>,
    line_boundaries: Vec<usize>,
    text: String,
}

fn recognize_target_blocking(
    model: &Model,
    target: OcrTarget,
    source_language: Language,
) -> Result<RecognizedTarget> {
    let primary_text = model.infer_blocking(&target.image)?;
    let mut col_texts_ltr = Vec::new();
    if source_language == Language::Japanese && target.column_crops.len() >= 2 {
        for col_crop in &target.column_crops {
            if let Ok(col_text) = model.infer_blocking(col_crop) {
                let trimmed = col_text.trim().to_owned();
                if !trimmed.is_empty() {
                    col_texts_ltr.push(trimmed);
                }
            }
        }
    }
    Ok(build_recognized_target(
        target,
        primary_text,
        col_texts_ltr,
        source_language,
    ))
}

fn build_recognized_target(
    target: OcrTarget,
    primary_text: String,
    col_texts_ltr: Vec<String>,
    source_language: Language,
) -> RecognizedTarget {
    if source_language != Language::Japanese {
        let inspection = JapaneseOcrInspection {
            score: if primary_text.trim().is_empty() {
                0.0
            } else {
                0.95
            },
            valid_char_ratio: 1.0,
            kinsoku_score: 1.0,
            likely_furigana_noise: false,
            is_corrupted: primary_text.trim().is_empty(),
        };
        return RecognizedTarget {
            target,
            primary_text: primary_text.clone(),
            candidate_rtl: primary_text,
            candidate_ltr: None,
            rtl_columns: Vec::new(),
            ltr_columns: Vec::new(),
            rtl_inspection: inspection,
            ltr_inspection: None,
        };
    }

    // Build RTL (`X_max -> X_min`) and LTR (`X_min -> X_max`) candidates from either sliced
    // vertical columns or multi-line vertical output.
    if col_texts_ltr.len() >= 2 {
        let rtl_columns: Vec<String> = col_texts_ltr.iter().rev().cloned().collect();
        let ltr_columns = col_texts_ltr;
        let candidate_rtl = rtl_columns.join("");
        let candidate_ltr = ltr_columns.join("");

        let rtl_inspection = inspect_japanese_ocr(&candidate_rtl, Some(&rtl_columns));
        let ltr_inspection = inspect_japanese_ocr(&candidate_ltr, Some(&ltr_columns));
        let primary_inspection = inspect_japanese_ocr(&primary_text, None);

        // If full-crop primary OCR outscores both sliced variants and agrees with RTL, keep it;
        // otherwise use the column-sliced candidates so LTR concatenation bugs cannot slip through.
        let (final_rtl, final_rtl_inspection) =
            if primary_inspection.score > rtl_inspection.score + 0.12
                && primary_inspection.score > ltr_inspection.score + 0.12
            {
                (primary_text.clone(), primary_inspection)
            } else {
                (candidate_rtl, rtl_inspection)
            };

        return RecognizedTarget {
            target,
            primary_text,
            candidate_rtl: final_rtl,
            candidate_ltr: Some(candidate_ltr),
            rtl_columns,
            ltr_columns,
            rtl_inspection: final_rtl_inspection,
            ltr_inspection: Some(ltr_inspection),
        };
    }

    // Check if primary_text contains multiple lines/segments in vertical mode.
    let lines: Vec<String> = primary_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if target.direction == TextDirection::Vertical && lines.len() >= 2 {
        let reversed_lines: Vec<String> = lines.iter().rev().cloned().collect();
        let as_is_joined = lines.join("");
        let rev_joined = reversed_lines.join("");
        let as_is_insp = inspect_japanese_ocr(&as_is_joined, Some(&lines));
        let rev_insp = inspect_japanese_ocr(&rev_joined, Some(&reversed_lines));
        return RecognizedTarget {
            target,
            primary_text,
            candidate_rtl: as_is_joined,
            candidate_ltr: Some(rev_joined),
            rtl_columns: lines,
            ltr_columns: reversed_lines,
            rtl_inspection: as_is_insp,
            ltr_inspection: Some(rev_insp),
        };
    }

    let rtl_inspection = inspect_japanese_ocr(&primary_text, None);
    RecognizedTarget {
        target,
        primary_text: primary_text.clone(),
        candidate_rtl: primary_text,
        candidate_ltr: None,
        rtl_columns: Vec::new(),
        ltr_columns: Vec::new(),
        rtl_inspection,
        ltr_inspection: None,
    }
}

async fn validate_and_recover_targets<F, Fut>(
    input: &StageInput,
    mut recognized: Vec<RecognizedTarget>,
    source_language: Language,
    mut infer_crop: F,
    device: &koharu_ml::Device,
    configured_local_model: Option<&OcrModel>,
    fallback_cell: &ModelCell<Model>,
) -> Result<Vec<OcrResult>>
where
    F: FnMut(DynamicImage) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    if recognized.is_empty() {
        return Ok(Vec::new());
    }

    let jev = JevClient::new();
    let judge_inputs: Vec<OcrJudgeInput> = recognized
        .iter()
        .map(|item| {
            let (min_x, min_y, max_x, max_y) =
                geometry_extents(&item.target.geometry).unwrap_or((0.0, 0.0, 1.0, 1.0));
            let width = (max_x - min_x).max(1.0);
            let height = (max_y - min_y).max(1.0);
            OcrJudgeInput {
                region_id: item.target.region.to_string(),
                candidate_rtl: item.candidate_rtl.clone(),
                candidate_ltr: item.candidate_ltr.clone(),
                rtl_inspector_score: item.rtl_inspection.score,
                ltr_inspector_score: item.ltr_inspection.as_ref().map(|insp| insp.score),
                likely_furigana_noise: item.rtl_inspection.likely_furigana_noise,
                aspect_ratio: width / height,
                column_count: item.target.column_spans.len().max(1),
            }
        })
        .collect();

    let verdicts = if source_language == Language::Japanese {
        jev.judge_ocr_batch(&judge_inputs).await
    } else {
        judge_inputs
            .iter()
            .map(super::jev::deterministic_ocr_verdict)
            .collect()
    };

    let page = input.page()?;
    let source_image = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;

    let mut results = Vec::with_capacity(recognized.len());
    for (item, verdict) in recognized.drain(..).zip(verdicts) {
        let (mut chosen_text, mut chosen_columns, mut chosen_score) =
            match verdict.reading_order.selected {
                OcrReadingOrderChoice::Ltr => {
                    if let (Some(ltr), Some(ltr_insp)) =
                        (item.candidate_ltr.clone(), item.ltr_inspection.clone())
                    {
                        (ltr, item.ltr_columns.clone(), ltr_insp.score)
                    } else {
                        (
                            item.candidate_rtl.clone(),
                            item.rtl_columns.clone(),
                            item.rtl_inspection.score,
                        )
                    }
                }
                OcrReadingOrderChoice::Rtl | OcrReadingOrderChoice::Corrupted => (
                    item.candidate_rtl.clone(),
                    item.rtl_columns.clone(),
                    item.rtl_inspection.score,
                ),
            };
        let mut chosen_direction = item.target.direction;

        let needs_recovery = source_language == Language::Japanese
            && (verdict.reading_order.selected == OcrReadingOrderChoice::Corrupted
                || !verdict.is_valid_japanese.is_affirmative(0.45)
                || verdict.retry_strategy.selected != OcrRetryStrategyChoice::Accept);

        if needs_recovery && !input.stop.stopped() {
            // Recovery Pass 1: Furigana suppression + expanded padding crop (1.4x)
            let excl_refs: Vec<&Geometry> = item.target.exclusions.iter().collect();
            let padded_crop = crop_with_scale(
                &source_image,
                &item.target.geometry,
                &excl_refs,
                OCR_RETRY_CROP_PADDING_SCALE,
            )
            .unwrap_or_else(|_| item.target.image.clone());
            let defurigana_crop =
                suppress_furigana_in_crop(&padded_crop, &item.target.crop_mask);

            if let Ok(retried_text) = infer_crop(defurigana_crop.clone()).await {
                let retried_insp = inspect_japanese_ocr(&retried_text, None);
                if retried_insp.score > chosen_score + 0.05 {
                    chosen_text = retried_text;
                    chosen_columns.clear();
                    chosen_score = retried_insp.score;
                }
            }

            // Recovery Pass 2: If still low quality and strategy suggests orientation flip or
            // column re-slicing on the furigana-suppressed crop.
            if chosen_score < 0.58 {
                if verdict.retry_strategy.selected
                    == OcrRetryStrategyChoice::FlipOrientationAndRetry
                {
                    chosen_direction = match chosen_direction {
                        TextDirection::Vertical => TextDirection::Horizontal,
                        TextDirection::Horizontal => TextDirection::Vertical,
                    };
                }
                let sliced = slice_vertical_columns(&defurigana_crop, &item.target.column_spans);
                if sliced.len() >= 2 {
                    let mut retry_cols_ltr = Vec::new();
                    for col_img in sliced {
                        if let Ok(t) = infer_crop(col_img).await
                            && !t.trim().is_empty()
                        {
                            retry_cols_ltr.push(t.trim().to_owned());
                        }
                    }
                    if retry_cols_ltr.len() >= 2 {
                        let retry_rtl_cols: Vec<String> =
                            retry_cols_ltr.iter().rev().cloned().collect();
                        let retry_rtl = retry_rtl_cols.join("");
                        let retry_rtl_insp =
                            inspect_japanese_ocr(&retry_rtl, Some(&retry_rtl_cols));
                        if retry_rtl_insp.score > chosen_score {
                            chosen_text = retry_rtl;
                            chosen_columns = retry_rtl_cols;
                            chosen_score = retry_rtl_insp.score;
                        }
                    }
                }
            }

            // Recovery Pass 3: Alternate local OCR model fallback on the problematic crop
            if chosen_score < 0.50
                && let Some(primary_model_cfg) = configured_local_model
            {
                let alt_cfg = alternate_local_ocr_model(primary_model_cfg);
                if fallback_cell
                    .ensure(|| Model::load(device.clone(), &alt_cfg))
                    .await
                    .is_ok()
                    && let Some(alt_model) = fallback_cell.lock().await.as_ref().cloned()
                    && let Ok(alt_text) = alt_model.infer_single(defurigana_crop).await
                {
                    let alt_insp = inspect_japanese_ocr(&alt_text, None);
                    if alt_insp.score > chosen_score {
                        chosen_text = alt_text;
                        chosen_columns.clear();
                        chosen_score = alt_insp.score;
                    }
                }
            }
        }

        let line_boundaries = compute_line_boundaries(&chosen_columns);
        results.push(OcrResult {
            content: item.target.content,
            region: item.target.region,
            bubble_id: item.target.bubble_id,
            geometry: item.target.geometry,
            direction: chosen_direction,
            confidence: Some(chosen_score as f32),
            line_boundaries,
            text: chosen_text,
        });
    }

    if source_language == Language::Japanese {
        reconcile_bubble_sibling_reading_order(&mut results);
    }

    Ok(results)
}

fn compute_line_boundaries(columns: &[String]) -> Vec<usize> {
    if columns.len() < 2 {
        return Vec::new();
    }
    let mut boundaries = Vec::with_capacity(columns.len() - 1);
    let mut offset = 0_usize;
    for col in columns.iter().take(columns.len() - 1) {
        offset += col.chars().count();
        boundaries.push(offset);
    }
    boundaries
}

/// If a single `BubbleRegion` contains multiple vertical `TextRegion`s whose combined
/// Japanese sentence is grammatically inverted (`candidate_ltr` instead of `candidate_rtl`
/// across the sibling regions), reorder their texts so right-to-left reading order holds.
fn reconcile_bubble_sibling_reading_order(results: &mut [OcrResult]) {
    let mut bubble_groups: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, result) in results.iter().enumerate() {
        if result.direction == TextDirection::Vertical
            && let Some(bubble_id) = result.bubble_id
        {
            bubble_groups.entry(bubble_id).or_default().push(idx);
        }
    }

    for indices in bubble_groups.values() {
        if indices.len() < 2 {
            continue;
        }
        let mut by_x_desc = indices.clone();
        by_x_desc.sort_by(|&a, &b| {
            let cx_a = geometry_center_x(&results[a].geometry);
            let cx_b = geometry_center_x(&results[b].geometry);
            cx_b.partial_cmp(&cx_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let current_cols: Vec<String> = indices
            .iter()
            .map(|&i| results[i].text.clone())
            .collect();
        let rtl_cols: Vec<String> = by_x_desc
            .iter()
            .map(|&i| results[i].text.clone())
            .collect();
        if current_cols == rtl_cols {
            continue;
        }
        let current_joined = current_cols.join("");
        let rtl_joined = rtl_cols.join("");
        let current_insp = inspect_japanese_ocr(&current_joined, Some(&current_cols));
        let rtl_insp = inspect_japanese_ocr(&rtl_joined, Some(&rtl_cols));
        if rtl_insp.score > current_insp.score + 0.08 {
            for (&target_idx, new_text) in indices.iter().zip(rtl_cols) {
                results[target_idx].text = new_text;
            }
        }
    }
}

fn geometry_center_x(geometry: &Geometry) -> f64 {
    geometry_extents(geometry)
        .map(|(min_x, _, max_x, _)| (min_x + max_x) * 0.5)
        .unwrap_or(0.0)
}

async fn targets(
    input: &StageInput,
    source_language: Language,
) -> Result<(EntityId, Vec<OcrTarget>)> {
    let page = input.page()?;
    let mut targets = Vec::new();
    let source = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let text_mask_asset = input
        .images
        .get(&input.scene, page, "text-mask")
        .await?
        .map(|img| img.to_luma8());
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
        let crop_img = crop(&source, geometry, &exclusions)
            .with_context(|| format!("text region {region} is outside its source image"))?;
        let crop_mask =
            extract_crop_mask(&crop_img, text_mask_asset.as_ref(), geometry, &exclusions);
        for relation in input.scene.relations_to_as::<RecognizedFrom>(*region) {
            let content = relation.value().source;
            let previous = input.scene.component::<SourceText>(content)?;
            if previous
                .as_ref()
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let bubble_id = region_bubble_id(&input.scene, *region, content)?;
            let projection = analyze_region_projection(
                &crop_mask,
                geometry,
                bubble_id.is_some(),
                source_language,
            );
            let (min_x, min_y, max_x, max_y) =
                geometry_extents(geometry).unwrap_or((0.0, 0.0, 1.0, 1.0));
            let width = max_x - min_x;
            let height = max_y - min_y;
            let column_crops = if projection.direction == TextDirection::Vertical
                && width > height * 0.85
                && projection.column_spans.len() >= 2
            {
                slice_vertical_columns(&crop_img, &projection.column_spans)
            } else {
                Vec::new()
            };
            targets.push(OcrTarget {
                content,
                region: *region,
                bubble_id,
                geometry: geometry.clone(),
                exclusions: exclusions.iter().map(|g| (*g).clone()).collect(),
                image: crop_img.clone(),
                crop_mask: crop_mask.clone(),
                direction: projection.direction,
                column_spans: projection.column_spans,
                column_crops,
            });
        }
    }
    Ok((page, targets))
}

fn region_bubble_id(
    scene: &koharu_scene::Snapshot,
    region: EntityId,
    content: EntityId,
) -> Result<Option<EntityId>> {
    for rel in scene.relations_from_as::<Inside>(region) {
        let target = rel.value().target;
        if scene
            .component::<Region>(target)?
            .is_some_and(|r| r.kind == BubbleRegion::kind())
        {
            return Ok(Some(target));
        }
    }
    for pres in scene.relations_to_as::<Presents>(content) {
        let layer = pres.value().source;
        for flow in scene.relations_from_as::<FlowsIn>(layer) {
            let target = flow.value().target;
            if scene
                .component::<Region>(target)?
                .is_some_and(|r| r.kind == BubbleRegion::kind())
            {
                return Ok(Some(target));
            }
        }
    }
    Ok(None)
}

#[derive(Clone, Debug)]
pub(crate) struct ProjectionAnalysis {
    pub direction: TextDirection,
    pub column_spans: Vec<(u32, u32)>,
    pub row_spans: Vec<(u32, u32)>,
}

/// Analyzes the horizontal and vertical ink projection profiles of `crop_mask` inside a
/// `TextRegion` to determine `TextDirection` (`Vertical` vs `Horizontal`) and locate
/// distinct vertical column spans `[col_start, col_end)` separated by vertical valleys.
pub(crate) fn analyze_region_projection(
    crop_mask: &GrayImage,
    geometry: &Geometry,
    flows_in_bubble: bool,
    source_language: Language,
) -> ProjectionAnalysis {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).unwrap_or((0.0, 0.0, f64::from(crop_mask.width()), f64::from(crop_mask.height())));
    let bbox_w = (max_x - min_x).max(1.0);
    let bbox_h = (max_y - min_y).max(1.0);

    let width = crop_mask.width();
    let height = crop_mask.height();
    let mut col_hist = vec![0_u32; width as usize];
    let mut row_hist = vec![0_u32; height as usize];
    let mut active_top = vec![height; width as usize];
    let mut active_bottom = vec![0_u32; width as usize];

    for y in 0..height {
        for x in 0..width {
            if crop_mask.get_pixel(x, y)[0] >= 127 {
                col_hist[x as usize] += 1;
                row_hist[y as usize] += 1;
                active_top[x as usize] = active_top[x as usize].min(y);
                active_bottom[x as usize] = active_bottom[x as usize].max(y + 1);
            }
        }
    }

    let min_col_gap = (width / 32).max(2);
    let min_col_width = (width / 24).max(3);
    let column_spans = extract_projection_spans(&col_hist, 1, min_col_gap, min_col_width);

    let min_row_gap = (height / 32).max(2);
    let min_row_height = (height / 24).max(3);
    let row_spans = extract_projection_spans(&row_hist, 1, min_row_gap, min_row_height);

    // Check if detected column spans are genuinely tall vertical text columns.
    let tall_vertical_columns = column_spans.len() >= 2
        && column_spans
            .iter()
            .filter(|&&(c_start, c_end)| {
                let col_w = (c_end - c_start).max(1);
                let top = (c_start..c_end)
                    .map(|x| active_top[x as usize])
                    .min()
                    .unwrap_or(0);
                let bottom = (c_start..c_end)
                    .map(|x| active_bottom[x as usize])
                    .max()
                    .unwrap_or(0);
                let col_h = bottom.saturating_sub(top);
                f64::from(col_h) >= f64::from(col_w) * 1.15
            })
            .count()
            >= 2;

    let direction = if bbox_h >= bbox_w * 1.15 {
        TextDirection::Vertical
    } else if tall_vertical_columns && column_spans.len() >= row_spans.len() {
        TextDirection::Vertical
    } else if source_language == Language::Japanese
        && flows_in_bubble
        && (tall_vertical_columns || bbox_h >= bbox_w * 0.72)
        && !(row_spans.len() >= 2 && column_spans.len() <= 1 && bbox_w > bbox_h * 1.25)
    {
        TextDirection::Vertical
    } else {
        TextDirection::Horizontal
    };

    ProjectionAnalysis {
        direction,
        column_spans,
        row_spans,
    }
}

fn extract_projection_spans(
    hist: &[u32],
    noise_floor: u32,
    min_valley_width: u32,
    min_span_width: u32,
) -> Vec<(u32, u32)> {
    let mut raw_spans: Vec<(u32, u32)> = Vec::new();
    let mut start: Option<u32> = None;
    for (idx, &count) in hist.iter().enumerate() {
        let pos = idx as u32;
        if count > noise_floor {
            if start.is_none() {
                start = Some(pos);
            }
        } else if let Some(s) = start.take() {
            raw_spans.push((s, pos));
        }
    }
    if let Some(s) = start {
        raw_spans.push((s, hist.len() as u32));
    }

    // Merge spans separated by narrow sub-character gaps (< min_valley_width)
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (s, e) in raw_spans {
        if let Some(last) = merged.last_mut()
            && s.saturating_sub(last.1) < min_valley_width
        {
            last.1 = e;
            continue;
        }
        merged.push((s, e));
    }

    merged
        .into_iter()
        .filter(|&(s, e)| e.saturating_sub(s) >= min_span_width)
        .collect()
}

/// Slices a wide vertical text crop along the vertical valleys between `column_spans`
/// into individual column sub-crops ordered left-to-right (`X_min -> X_max`).
pub(crate) fn slice_vertical_columns(
    crop_image: &DynamicImage,
    column_spans: &[(u32, u32)],
) -> Vec<DynamicImage> {
    if column_spans.len() < 2 {
        return Vec::new();
    }
    let width = crop_image.width();
    let height = crop_image.height();
    if width < 8 || height < 8 {
        return Vec::new();
    }

    let mut boundaries = Vec::with_capacity(column_spans.len() + 1);
    boundaries.push(0_u32);
    for pair in column_spans.windows(2) {
        let (_, left_end) = pair[0];
        let (right_start, _) = pair[1];
        let midpoint = ((left_end + right_start) / 2).clamp(1, width.saturating_sub(1));
        boundaries.push(midpoint);
    }
    boundaries.push(width);

    let mut crops = Vec::with_capacity(column_spans.len());
    for window in boundaries.windows(2) {
        let x0 = window[0];
        let x1 = window[1];
        if x1 > x0 + 2 {
            crops.push(crop_image.crop_imm(x0, 0, x1 - x0, height));
        }
    }
    crops
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JapaneseOcrInspection {
    pub score: f64,
    pub valid_char_ratio: f64,
    pub kinsoku_score: f64,
    pub likely_furigana_noise: bool,
    pub is_corrupted: bool,
}

/// Deterministic Japanese OCR Validator & Kinsoku Shori (禁則処理) Analyzer:
/// - Evaluates valid Japanese script ratio vs OCR artifact/placeholder symbols.
/// - Checks Kinsoku Shori line-start / column-start prohibitions (closing punctuation,
///   small kana, prolonged sound mark `ー`, and postpositional case/binding particles).
/// - Scores cross-column grammatical transitions (`columns`) to detect inverted LTR vs RTL
///   column ordering.
pub(crate) fn inspect_japanese_ocr(
    text: &str,
    columns: Option<&[String]>,
) -> JapaneseOcrInspection {
    let visible: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if visible.is_empty() {
        return JapaneseOcrInspection {
            score: 0.0,
            valid_char_ratio: 0.0,
            kinsoku_score: 0.0,
            likely_furigana_noise: false,
            is_corrupted: true,
        };
    }

    let mut valid_count = 0_usize;
    let mut kana_count = 0_usize;
    let mut kanji_count = 0_usize;
    let mut garbage_count = 0_usize;

    for &ch in &visible {
        if is_hiragana(ch) || is_katakana(ch) {
            valid_count += 1;
            kana_count += 1;
        } else if is_cjk_ideograph(ch) {
            valid_count += 1;
            kanji_count += 1;
        } else if is_japanese_punctuation(ch) || ch.is_ascii_alphanumeric() {
            valid_count += 1;
        } else if matches!(ch, '☐' | '□' | '▢' | '▣' | '' | '|' | '\\' | '/' | '_') {
            garbage_count += 2;
        } else {
            garbage_count += 1;
        }
    }

    let total = visible.len();
    let valid_char_ratio =
        (valid_count as f64 / (total + garbage_count) as f64).clamp(0.0, 1.0);

    // Penalize pathological character repetition (>= 4 identical non-expressive chars in a row).
    let mut max_run = 1_usize;
    let mut current_run = 1_usize;
    for pair in visible.windows(2) {
        if pair[0] == pair[1] && !matches!(pair[0], '…' | 'ー' | '！' | '!' | '？' | '?' | 'w' | 'ｗ')
        {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 1;
        }
    }
    let repetition_penalty = if max_run >= 4 { 0.35 } else { 0.0 };

    // Kinsoku Shori & utterance boundary grammar scoring
    let mut kinsoku_score = 0.70_f64;
    let first_char = visible[0];
    if is_gyoto_kinsoku_char(first_char) {
        kinsoku_score -= 0.42;
    } else if is_enclitic_particle_starter(&visible) {
        kinsoku_score -= 0.32;
    }

    if let Some(&last_char) = visible.last() {
        if is_natural_japanese_ending(&visible, last_char) {
            kinsoku_score += 0.18;
        }
        if is_gyomatsu_kinsoku_char(last_char) {
            kinsoku_score -= 0.30;
        }
    }

    // Evaluate cross-column seam continuity if multiple columns are provided.
    if let Some(cols) = columns
        && cols.len() >= 2
    {
        kinsoku_score += score_column_seams(cols);
    }
    let kinsoku_score = kinsoku_score.clamp(0.0, 1.0);

    let likely_furigana_noise = total >= 6
        && kanji_count > 0
        && kana_count > kanji_count * 4
        && visible
            .windows(3)
            .filter(|w| is_small_kana(w[1]) && is_small_kana(w[2]))
            .count()
            >= 1;

    let score = (valid_char_ratio * 0.55 + kinsoku_score * 0.45 - repetition_penalty)
        .clamp(0.0, 1.0);
    let is_corrupted = score < 0.42 || valid_char_ratio < 0.55;

    JapaneseOcrInspection {
        score,
        valid_char_ratio,
        kinsoku_score,
        likely_furigana_noise,
        is_corrupted,
    }
}

fn score_column_seams(columns: &[String]) -> f64 {
    let mut delta = 0.0_f64;
    for pair in columns.windows(2) {
        let prev: Vec<char> = pair[0].chars().filter(|c| !c.is_whitespace()).collect();
        let next: Vec<char> = pair[1].chars().filter(|c| !c.is_whitespace()).collect();
        if prev.is_empty() || next.is_empty() {
            continue;
        }
        let prev_last = *prev.last().unwrap();
        let next_first = next[0];

        // If `prev` ends with terminal punctuation (`！`, `？`, `。`, `」`) while `next` starts
        // with an enclitic particle or auxiliary continuation (`を`, `が`, `は`, `に`, `の`, `で`,
        // `と`, `て`, `た`, `ます`, `です`), then `next` is actually the middle/end of a clause
        // that was inverted!
        let prev_is_terminal = matches!(
            prev_last,
            '。' | '！' | '!' | '？' | '?' | '」' | '』' | '）' | ')'
        );
        let next_is_continuation =
            is_gyoto_kinsoku_char(next_first) || is_enclitic_particle_starter(&next);

        if prev_is_terminal && next_is_continuation {
            delta -= 0.38;
        } else if !prev_is_terminal && next_is_continuation {
            // A non-terminal stem at the end of `prev` flowing into a particle/auxiliary at the
            // start of `next` (common in tight manga balloons!) is a strong sign that `prev -> next`
            // is the true grammatical order!
            delta += 0.24;
        } else if prev_is_terminal && !next_is_continuation {
            // If `prev` ends with terminal punctuation and `next` does NOT end with terminal
            // punctuation, check if `prev` is actually the final column of the sentence.
            let next_last = *next.last().unwrap();
            let next_is_terminal = matches!(
                next_last,
                '。' | '！' | '!' | '？' | '?' | '…' | '」' | '』'
            );
            if !next_is_terminal {
                delta -= 0.22;
            }
        }
    }
    delta
}

fn is_hiragana(ch: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&ch)
}

fn is_katakana(ch: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&ch) || ('\u{31F0}'..='\u{31FF}').contains(&ch)
}

fn is_cjk_ideograph(ch: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&ch)
        || ('\u{3400}'..='\u{4DBF}').contains(&ch)
        || ch == '々'
}

fn is_japanese_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '。' | '、'
            | '！'
            | '!'
            | '？'
            | '?'
            | '…'
            | '‥'
            | 'ー'
            | '「'
            | '」'
            | '『'
            | '』'
            | '（'
            | '）'
            | '('
            | ')'
            | '【'
            | '】'
            | '・'
            | '〜'
            | '～'
            | '—'
            | '―'
    )
}

fn is_small_kana(ch: char) -> bool {
    matches!(
        ch,
        'っ' | 'ゃ'
            | 'ゅ'
            | 'ょ'
            | 'ぁ'
            | 'ぃ'
            | 'ぅ'
            | 'ぇ'
            | 'ぉ'
            | 'ゎ'
            | 'ッ'
            | 'ャ'
            | 'ュ'
            | 'ョ'
            | 'ァ'
            | 'ィ'
            | 'ゥ'
            | 'ェ'
            | 'ォ'
            | 'ヮ'
            | 'ヵ'
            | 'ヶ'
    )
}

/// Gyoto Kinsoku (行頭禁則文字): characters that cannot legitimately begin a Japanese phrase.
fn is_gyoto_kinsoku_char(ch: char) -> bool {
    is_small_kana(ch)
        || matches!(
            ch,
            '！' | '!'
                | '？'
                | '?'
                | '。'
                | '、'
                | '」'
                | '』'
                | '）'
                | ')'
                | '】'
                | '〉'
                | '》'
                | 'ー'
                | '・'
                | 'ん'
                | 'ン'
        )
}

/// Gyomatsu Kinsoku (行末禁則文字): opening brackets/quotes that cannot end a phrase.
fn is_gyomatsu_kinsoku_char(ch: char) -> bool {
    matches!(ch, '「' | '『' | '（' | '(' | '【' | '〈' | '《')
}

/// Postpositional particles and bound auxiliaries that cannot start an independent sentence.
fn is_enclitic_particle_starter(chars: &[char]) -> bool {
    let Some(&first) = chars.first() else {
        return false;
    };
    let s: String = chars.iter().take(4).collect();
    if [
        "です",
        "ます",
        "でした",
        "ました",
        "ない",
        "なかった",
        "れる",
        "られる",
        "ちゃう",
        "じゃう",
    ]
    .iter()
    .any(|prefix| s.starts_with(prefix))
    {
        return true;
    }
    matches!(
        first,
        'は' | 'が' | 'を' | 'に' | 'へ' | 'で' | 'と' | 'も' | 'の' | 'ば'
    )
}

fn is_natural_japanese_ending(chars: &[char], last: char) -> bool {
    if matches!(
        last,
        '。' | '！'
            | '!'
            | '？'
            | '?'
            | '…'
            | '」'
            | '』'
            | '）'
            | ')'
            | 'か'
            | 'ね'
            | 'よ'
            | 'わ'
            | 'ぞ'
            | 'ぜ'
            | 'な'
            | 'さ'
            | 'だ'
            | 'た'
            | 'る'
            | 'い'
            | 'う'
            | 'ん'
            | 'ろ'
    ) {
        return true;
    }
    let tail: String = chars
        .iter()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    tail.ends_with("です") || tail.ends_with("ます") || tail.ends_with("ない")
}

/// Suppresses small furigana (rubi) connected components located immediately to the right of
/// main vertical text columns on `crop_image` by filling them with the crop's background color.
pub(crate) fn suppress_furigana_in_crop(
    crop_image: &DynamicImage,
    crop_mask: &GrayImage,
) -> DynamicImage {
    let mut rgb = crop_image.to_rgb8();
    let (width, height) = rgb.dimensions();
    if width < 12 || height < 12 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let mask = if crop_mask.dimensions() == (width, height) {
        crop_mask.clone()
    } else {
        derive_ink_mask_from_rgb(&rgb)
    };

    let bg_color = estimate_crop_background_color(&rgb, &mask);
    let binary = GrayImage::from_fn(width, height, |x, y| {
        Luma([(mask.get_pixel(x, y)[0] >= 127) as u8 * u8::MAX])
    });
    let labels = connected_components(&binary, Connectivity::Eight, Luma([0]));

    #[derive(Clone, Copy, Default)]
    struct CompStats {
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        area: u32,
    }

    let mut stats: BTreeMap<u32, CompStats> = BTreeMap::new();
    for (x, y, px) in labels.enumerate_pixels() {
        let label = px[0];
        if label == 0 {
            continue;
        }
        stats
            .entry(label)
            .and_modify(|s| {
                s.min_x = s.min_x.min(x);
                s.min_y = s.min_y.min(y);
                s.max_x = s.max_x.max(x + 1);
                s.max_y = s.max_y.max(y + 1);
                s.area += 1;
            })
            .or_insert(CompStats {
                min_x: x,
                min_y: y,
                max_x: x + 1,
                max_y: y + 1,
                area: 1,
            });
    }

    if stats.len() < 2 {
        return DynamicImage::ImageRgb8(rgb);
    }

    let mut widths: Vec<u32> = stats.values().map(|s| s.max_x - s.min_x).collect();
    let mut areas: Vec<u32> = stats.values().map(|s| s.area).collect();
    widths.sort_unstable();
    areas.sort_unstable();
    let p75_width = widths[(widths.len() * 3 / 4).min(widths.len() - 1)].max(4);
    let p75_area = areas[(areas.len() * 3 / 4).min(areas.len() - 1)].max(8);

    // Identify main character components vs tiny furigana components positioned directly to the
    // right of a main vertical column.
    let main_components: Vec<CompStats> = stats
        .values()
        .copied()
        .filter(|s| (s.max_x - s.min_x) * 2 >= p75_width || s.area * 2 >= p75_area)
        .collect();

    let mut furigana_labels = Vec::new();
    for (&label, comp) in &stats {
        let comp_w = comp.max_x - comp.min_x;
        let comp_h = comp.max_y - comp.min_y;
        let is_small =
            comp_w * 100 <= p75_width * 55 && comp_h * 100 <= p75_width * 70 && comp.area * 100 <= p75_area * 45;
        if !is_small {
            continue;
        }
        let comp_cy = (comp.min_y + comp.max_y) as i32 / 2;
        let attached_to_main_right = main_components.iter().any(|main| {
            let vertical_overlap = comp_cy >= main.min_y as i32 - 4 && comp_cy <= main.max_y as i32 + 4;
            let gap_right = comp.min_x as i32 - main.max_x as i32;
            vertical_overlap && gap_right >= -2 && gap_right <= (p75_width as i32).max(6)
        });
        if attached_to_main_right {
            furigana_labels.push(label);
        }
    }

    if furigana_labels.is_empty() {
        return DynamicImage::ImageRgb8(rgb);
    }

    for (x, y, px) in labels.enumerate_pixels() {
        if furigana_labels.contains(&px[0]) {
            rgb.put_pixel(x, y, bg_color);
        }
    }

    DynamicImage::ImageRgb8(rgb)
}

fn estimate_crop_background_color(rgb: &RgbImage, mask: &GrayImage) -> Rgb<u8> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for (x, y, px) in rgb.enumerate_pixels() {
        if mask.get_pixel(x, y)[0] < 127 {
            channels[0].push(px[0]);
            channels[1].push(px[1]);
            channels[2].push(px[2]);
        }
    }
    if channels[0].is_empty() {
        return Rgb([255, 255, 255]);
    }
    let median = |values: &mut Vec<u8>| -> u8 {
        values.sort_unstable();
        values[values.len() / 2]
    };
    Rgb([
        median(&mut channels[0]),
        median(&mut channels[1]),
        median(&mut channels[2]),
    ])
}

fn extract_crop_mask(
    crop_image: &DynamicImage,
    page_text_mask: Option<&GrayImage>,
    geometry: &Geometry,
    exclusions: &[&Geometry],
) -> GrayImage {
    if let Some(page_mask) = page_text_mask
        && let Ok(masked) = crop_mask_from_page(page_mask, geometry, exclusions)
        && masked.pixels().any(|p| p[0] >= 127)
    {
        return masked;
    }
    derive_ink_mask_from_rgb(&crop_image.to_rgb8())
}

fn crop_mask_from_page(
    page_mask: &GrayImage,
    geometry: &Geometry,
    exclusions: &[&Geometry],
) -> Result<GrayImage> {
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
        .min(f64::from(page_mask.width())) as u32;
    let bottom = (center_y + half_height)
        .ceil()
        .max(0.0)
        .min(f64::from(page_mask.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the mask");
    }
    let mut sub = image::imageops::crop_imm(page_mask, x, y, right - x, bottom - y).to_image();
    for local_y in 0..sub.height() {
        for local_x in 0..sub.width() {
            let sample_x = f64::from(x + local_x) + 0.5;
            let sample_y = f64::from(y + local_y) + 0.5;
            let geometry_x = center_x + (sample_x - center_x) / OCR_CROP_PADDING_SCALE;
            let geometry_y = center_y + (sample_y - center_y) / OCR_CROP_PADDING_SCALE;
            if !point_in_geometry(geometry_x, geometry_y, geometry) {
                sub.put_pixel(local_x, local_y, Luma([0]));
                continue;
            }
            for exclusion in exclusions {
                if point_in_geometry(sample_x, sample_y, exclusion) {
                    sub.put_pixel(local_x, local_y, Luma([0]));
                    break;
                }
            }
        }
    }
    Ok(sub)
}

fn derive_ink_mask_from_rgb(rgb: &RgbImage) -> GrayImage {
    let (width, height) = rgb.dimensions();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    let mut lumas: Vec<u8> = rgb
        .pixels()
        .map(|px| ((u16::from(px[0]) * 77 + u16::from(px[1]) * 150 + u16::from(px[2]) * 29) >> 8) as u8)
        .collect();
    lumas.sort_unstable();
    let bg_luma = i32::from(lumas[lumas.len() * 3 / 4]);
    GrayImage::from_fn(width, height, |x, y| {
        let px = rgb.get_pixel(x, y);
        let luma =
            ((i32::from(px[0]) * 77 + i32::from(px[1]) * 150 + i32::from(px[2]) * 29) >> 8)
                as i32;
        if (bg_luma - luma).abs() >= 40 {
            Luma([u8::MAX])
        } else {
            Luma([0])
        }
    })
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
        let _ = geometry_extents(&result.geometry)
            .ok_or_else(|| anyhow!("text region {} has empty geometry", result.region))?;
        edit.set(
            result.region,
            &OcrAnalysis {
                origin: Origin::Generated(generation.clone()),
                direction: result.direction,
                confidence: result.confidence,
                line_boundaries: result.line_boundaries,
            },
        )?;
    }
    finish(edit)
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
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | ''))
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
    crop_with_scale(source, geometry, exclusions, OCR_CROP_PADDING_SCALE)
}

fn crop_with_scale(
    source: &DynamicImage,
    geometry: &Geometry,
    exclusions: &[&Geometry],
    padding_scale: f64,
) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let half_width = (max_x - min_x) * 0.5 * padding_scale;
    let half_height = (max_y - min_y) * 0.5 * padding_scale;
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
            let geometry_x = center_x + (sample_x - center_x) / padding_scale;
            let geometry_y = center_y + (sample_y - center_y) / padding_scale;
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

    #[test]
    fn projection_profile_classifies_wide_multi_column_japanese_box_as_vertical() {
        // Wide box (72x48) containing 3 distinct vertical columns of Japanese text
        let mut mask = GrayImage::new(72, 48);
        for &(col_start, col_end) in &[(8_u32, 16_u32), (30, 38), (52, 60)] {
            for y in 6..42 {
                for x in col_start..col_end {
                    mask.put_pixel(x, y, Luma([u8::MAX]));
                }
            }
        }
        let geometry = Geometry::rectangle(0.0, 0.0, 72.0, 48.0);
        let analysis = analyze_region_projection(&mask, &geometry, true, Language::Japanese);
        assert_eq!(analysis.direction, TextDirection::Vertical);
        assert_eq!(analysis.column_spans.len(), 3);

        let dummy_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(72, 48, Rgb([255, 255, 255])));
        let slices = slice_vertical_columns(&dummy_img, &analysis.column_spans);
        assert_eq!(slices.len(), 3);
    }

    #[test]
    fn japanese_ocr_inspector_prefers_rtl_over_inverted_ltr_columns() {
        // Right column: "お前がやったの" (stem + particle)
        // Left column:  "ですか！？" (auxiliary + sentence-final particles + punctuation)
        let rtl_cols = vec!["お前がやったの".to_owned(), "ですか！？".to_owned()];
        let ltr_cols = vec!["ですか！？".to_owned(), "お前がやったの".to_owned()];

        let rtl_text = rtl_cols.join("");
        let ltr_text = ltr_cols.join("");

        let rtl_report = inspect_japanese_ocr(&rtl_text, Some(&rtl_cols));
        let ltr_report = inspect_japanese_ocr(&ltr_text, Some(&ltr_cols));

        assert!(
            rtl_report.score > ltr_report.score + 0.25,
            "expected RTL score ({}) >> LTR score ({})",
            rtl_report.score,
            ltr_report.score
        );
    }

    #[test]
    fn furigana_suppression_removes_small_right_side_rubi_components() {
        let mut rgb = RgbImage::from_pixel(48, 48, Rgb([250, 250, 250]));
        let mut mask = GrayImage::new(48, 48);
        // Main kanji column at x in 14..26, y in 8..40
        for y in 8..40 {
            for x in 14..26 {
                rgb.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        // Small furigana rubi component just to the right at x in 28..31, y in 12..16
        for y in 12..16 {
            for x in 28..31 {
                rgb.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }

        let cleaned = suppress_furigana_in_crop(&DynamicImage::ImageRgb8(rgb), &mask).to_rgb8();
        // Main column remains intact
        assert_eq!(cleaned.get_pixel(20, 20), &Rgb([10, 10, 10]));
        // Small right-side furigana component is filled with background color (250, 250, 250)
        assert_eq!(cleaned.get_pixel(29, 14), &Rgb([250, 250, 250]));
    }

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
