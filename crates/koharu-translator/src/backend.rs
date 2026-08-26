use std::{io::Cursor, sync::Arc};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, ImageEncoder, RgbImage, codecs::jpeg::JpegEncoder};

use crate::Language;

const MAX_IMAGE_DIMENSION: u32 = 2048;
const JPEG_QUALITY: u8 = 88;

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    pub segments: Vec<String>,
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub instructions: Option<String>,
    pub context: Vec<TranslationContext>,
    pub image: Option<Arc<DynamicImage>>,
    pub(crate) segment_metadata: Vec<TranslationSegmentMetadata>,
    pub(crate) page_numbers: Option<Vec<usize>>,
    pub(crate) segment_ids: Vec<String>,
    pub(crate) units: Vec<TranslationUnit>,
    pub(crate) unit_policy: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OcrRequest {
    pub image: Arc<DynamicImage>,
    pub source_language: Option<Language>,
    pub instructions: Option<String>,
}

impl TranslationRequest {
    #[must_use]
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<String>>,
        target_language: Language,
    ) -> Self {
        let segments = segments.into_iter().map(Into::into).collect::<Vec<_>>();
        Self {
            segment_metadata: vec![TranslationSegmentMetadata::default(); segments.len()],
            segment_ids: (0..segments.len())
                .map(|index| format!("segment:{index}"))
                .collect(),
            segments,
            source_language: None,
            target_language,
            instructions: None,
            context: Vec::new(),
            image: None,
            page_numbers: None,
            units: Vec::new(),
            unit_policy: None,
        }
    }

    #[must_use]
    pub fn with_source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    pub fn with_segment_metadata(
        mut self,
        metadata: impl IntoIterator<Item = TranslationSegmentMetadata>,
    ) -> anyhow::Result<Self> {
        let metadata = metadata.into_iter().collect::<Vec<_>>();
        anyhow::ensure!(
            metadata.len() == self.segments.len(),
            "translation metadata count does not match segment count"
        );
        self.segment_metadata = metadata;
        self.segment_ids = self
            .segment_metadata
            .iter()
            .enumerate()
            .map(|(index, metadata)| metadata.stable_id(index))
            .collect();
        Ok(self)
    }

    pub fn with_segment_ids(
        mut self,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> anyhow::Result<Self> {
        let ids = ids.into_iter().map(Into::into).collect::<Vec<_>>();
        anyhow::ensure!(
            ids.len() == self.segments.len(),
            "translation ID count does not match segment count"
        );
        anyhow::ensure!(
            ids.iter().all(|id| !id.trim().is_empty()),
            "translation IDs cannot be empty"
        );
        let unique = ids.iter().collect::<std::collections::BTreeSet<_>>();
        anyhow::ensure!(unique.len() == ids.len(), "translation IDs must be unique");
        self.segment_ids = ids;
        Ok(self)
    }

    #[must_use]
    pub fn with_units(
        mut self,
        policy: impl Into<String>,
        units: impl IntoIterator<Item = TranslationUnit>,
    ) -> Self {
        self.unit_policy = Some(policy.into());
        self.units = units.into_iter().collect();
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    #[must_use]
    pub fn with_image(mut self, image: Arc<DynamicImage>) -> Self {
        self.image = Some(image);
        self
    }

    #[must_use]
    pub fn with_context(mut self, context: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.context = context.into_iter().collect();
        self
    }

    pub fn with_page_numbers(
        mut self,
        pages: impl IntoIterator<Item = usize>,
    ) -> anyhow::Result<Self> {
        let pages = pages.into_iter().collect::<Vec<_>>();
        anyhow::ensure!(
            pages.len() == self.segments.len(),
            "translation page count does not match segment count"
        );
        self.page_numbers = Some(pages);
        Ok(self)
    }

    pub(crate) fn prepare_image(&mut self) -> anyhow::Result<()> {
        prepare_image(&mut self.image)
    }

    pub(crate) fn page_number(&self, id: usize) -> usize {
        self.page_numbers
            .as_ref()
            .and_then(|pages| pages.get(id))
            .copied()
            .unwrap_or(1)
    }

    pub(crate) fn is_chapter(&self) -> bool {
        self.page_numbers.is_some()
    }

    pub(crate) fn chapter_split_hint(&self) -> String {
        let max_page = self
            .page_numbers
            .as_ref()
            .and_then(|pages| pages.iter().copied().max())
            .unwrap_or(0);
        let midpoint = max_page / 2;
        format!(
            "Try two chapter parts: pages 1-{midpoint} and pages {}-{max_page}.",
            midpoint.saturating_add(1)
        )
    }

    pub(crate) fn segment_metadata(&self, id: usize) -> &TranslationSegmentMetadata {
        self.segment_metadata
            .get(id)
            .expect("translation metadata is aligned with segments")
    }

    pub(crate) fn segment_id(&self, id: usize) -> &str {
        self.segment_ids
            .get(id)
            .map(String::as_str)
            .expect("translation IDs are aligned with segments")
    }

    pub(crate) fn page_id(&self, id: usize) -> String {
        self.segment_metadata(id)
            .page_id
            .clone()
            .unwrap_or_else(|| format!("page:{}", self.page_number(id)))
    }

    pub(crate) fn region_id(&self, id: usize) -> String {
        self.segment_metadata(id)
            .region_id
            .clone()
            .unwrap_or_else(|| format!("region:{id}"))
    }

    pub(crate) fn order_index(&self, id: usize) -> usize {
        self.segment_metadata(id).order_index.unwrap_or(id)
    }

    pub(crate) fn remove_image(&mut self) {
        self.image = None;
    }
}

impl OcrRequest {
    pub(crate) fn prepare_image(&mut self) -> anyhow::Result<()> {
        let mut image = Some(Arc::clone(&self.image));
        prepare_image(&mut image)?;
        self.image = image.expect("OCR image cannot be removed");
        Ok(())
    }
}

fn prepare_image(image: &mut Option<Arc<DynamicImage>>) -> anyhow::Result<()> {
    let Some(source_image) = image.as_ref() else {
        return Ok(());
    };
    let (width, height) = (source_image.width(), source_image.height());
    let longest = width.max(height);
    if longest <= MAX_IMAGE_DIMENSION {
        return Ok(());
    }
    let resized_width =
        (u64::from(width) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
    let resized_height =
        (u64::from(height) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
    let source = source_image.to_rgb8();
    let mut resized = RgbImage::new(resized_width, resized_height);
    Resizer::new()
        .resize(
            &source,
            &mut resized,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .context("failed to resize translation image")?;
    *image = Some(Arc::new(DynamicImage::ImageRgb8(resized)));
    Ok(())
}

pub(crate) struct EncodedImage {
    pub(crate) data: String,
}

impl EncodedImage {
    pub(crate) fn data_url(&self) -> String {
        format!("data:image/jpeg;base64,{}", self.data)
    }
}

pub(crate) fn encode_image(image: &DynamicImage) -> anyhow::Result<EncodedImage> {
    let rgb = image.to_rgb8();
    let mut bytes = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .context("failed to encode translation image")?;
    Ok(EncodedImage {
        data: STANDARD.encode(bytes.into_inner()),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationContext {
    pub source: String,
    pub translation: String,
}

/// Semantic hints attached to one OCR/translation segment.
///
/// The role is the existing namespaced `TextRole` value (for example
/// `dev.koharu.text.dialogue`), while `inside_balloon` comes from the existing
/// `FlowsIn` relation between a text layer and a detected bubble.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationSegmentMetadata {
    pub role: Option<String>,
    pub inside_balloon: bool,
    #[serde(default)]
    pub page_id: Option<String>,
    #[serde(default)]
    pub region_id: Option<String>,
    #[serde(default)]
    pub bubble_id: Option<String>,
    #[serde(default)]
    pub order_index: Option<usize>,
    #[serde(default)]
    pub unit_id: Option<String>,
    #[serde(default)]
    pub unit_type: Option<String>,
    #[serde(default)]
    pub is_sound_effect: bool,
}

impl TranslationSegmentMetadata {
    fn stable_id(&self, index: usize) -> String {
        match (&self.page_id, &self.region_id) {
            (Some(page), Some(region)) => format!("page:{page}::region:{region}"),
            _ => format!("segment:{index}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationUnit {
    pub id: String,
    pub page_id: String,
    pub page_index: usize,
    pub unit_type: String,
    pub region_ids: Vec<String>,
    pub segment_ids: Vec<String>,
}

impl TranslationContext {
    #[cfg(test)]
    #[must_use]
    pub fn new(source: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
        }
    }
}
