use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use koharu_renderer::{
    CopyfitFontConfig, contour, geometry_frame, measure_text_fit, split_text_for_lobes,
};
use koharu_scene::{
    Authored, BubbleRegion, EntityId, Geometry, LanguageTag, Origin, PanelRegion, Region,
    RegionSpec, SourceText, Translation, Typography,
};
use koharu_translator::{
    GenerationConfig, Language, ModelSelection, Provider, TranslationContext, TranslationRequest,
    TranslationSegmentMetadata, TranslationUnit, TranslationVariants, Translator,
};

use crate::{StageTarget, TranslationConfig, TranslationProfile, TranslationUnitPolicy};

use super::{
    StageInput, StageProcessor,
    detection::{LayoutItem, LayoutKind, reading_order},
    finish, generation,
    jev::{
        CopyfitCandidateMetrics, CopyfitJudgeInput, CopyfitVariantChoice, JevClient,
    },
};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }

    fn profile(&self, input: &StageInput) -> &TranslationProfile {
        match input.target() {
            StageTarget::Page(_) => &self.config.page,
            StageTarget::Chapter => &self.config.chapter,
        }
    }
}

#[derive(Clone)]
struct Target {
    entity: EntityId,
    source: String,
    page: usize,
    metadata: TranslationSegmentMetadata,
    bubble_polygon: Option<Arc<[[f32; 2]]>>,
    preferred_font: Option<String>,
    explicit_font_size: Option<f32>,
}

enum TranslationEntry {
    Target(Target),
    Context(TranslationContext),
}

struct OrderedEntry {
    entry: TranslationEntry,
    bounds: Option<(f64, f64, f64, f64)>,
}

/// The complete, ordered chapter translation input and its target entities.
/// Keeping request construction and patch construction together ensures that
/// network responses and manually imported files use exactly the same contract.
pub struct ChapterTranslation {
    snapshot: koharu_scene::Snapshot,
    targets: Vec<Target>,
    request: TranslationRequest,
    model: ModelSelection,
    generation: GenerationConfig,
    target_language: Language,
}

impl ChapterTranslation {
    pub fn from_snapshot(
        snapshot: koharu_scene::Snapshot,
        config: &TranslationConfig,
    ) -> Result<Self> {
        let pages: Arc<[EntityId]> = snapshot.pages().map(|page| page.id()).collect();
        let input = StageInput::chapter(snapshot, pages);
        Self::from_input(
            &input,
            &config.chapter,
            config.source_language,
            config.target_language,
        )
    }

    fn from_input(
        input: &StageInput,
        profile: &TranslationProfile,
        source_language: Language,
        target_language: Language,
    ) -> Result<Self> {
        let (mut targets, context) = collect_inputs(input, source_language)?;
        anyhow::ensure!(
            !targets.is_empty() || !context.is_empty(),
            "Translation found no OCR text. Run Detection and OCR before Translation."
        );
        let units = match profile.unit_policy {
            TranslationUnitPolicy::PageOnly => page_units(&mut targets)?,
            TranslationUnitPolicy::AdaptiveV1 => adaptive_units(&mut targets)?,
        };
        let mut request = TranslationRequest::new(
            targets.iter().map(|target| target.source.clone()),
            target_language,
        )
        .with_source_language(source_language)
        .with_segment_metadata(targets.iter().map(|target| target.metadata.clone()))?
        .with_segment_ids(
            targets
                .iter()
                .map(|target| stable_segment_id(&target.metadata))
                .collect::<Vec<_>>(),
        )?
        .with_context(context)
        .with_page_numbers(targets.iter().map(|target| target.page))?
        .with_units(
            match profile.unit_policy {
                TranslationUnitPolicy::PageOnly => "page_only",
                TranslationUnitPolicy::AdaptiveV1 => "adaptive_v1",
            },
            units,
        );
        if let Some(instructions) = profile.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        Ok(Self {
            snapshot: input.scene.clone(),
            targets,
            request,
            model: profile.model.clone(),
            generation: profile.generation,
            target_language,
        })
    }

    #[must_use]
    pub fn model(&self) -> &ModelSelection {
        &self.model
    }

    pub fn openrouter_request(&self) -> Result<serde_json::Value> {
        anyhow::ensure!(
            !self.request.segments.is_empty(),
            "chapter translation found no untranslated OCR text"
        );
        Translator::openrouter_request(&self.model, self.generation, self.request.clone())
    }

    pub fn patch_from_response(&self, response: &str) -> Result<koharu_scene::Patch> {
        let variants = koharu_translator::parse_translation_variants_response(
            "openrouter",
            response,
            &self.request,
        )?;
        let selected = select_best_copyfit_variants_sync(&self.targets, variants);
        self.patch_from_translations(selected, "openrouter-manual-import")
    }

    fn patch_from_translations(
        &self,
        translated: Vec<String>,
        provider: &str,
    ) -> Result<koharu_scene::Patch> {
        anyhow::ensure!(
            translated.len() == self.targets.len(),
            "chapter translation returned {} translations, expected {}",
            translated.len(),
            self.targets.len()
        );
        let language = koharu_scene::LanguageTag::new(self.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = self.snapshot.edit_as(generated.clone());
        for target in &self.targets {
            edit.observe::<SourceText>(target.entity)?;
            edit.observe::<Translation>(target.entity)?;
        }
        for (target, text) in self.targets.iter().zip(translated) {
            let text = if target.source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                target.entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, input: &StageInput) -> String {
        Translator::model(&self.profile(input).model).to_owned()
    }

    fn recovers_from_memory_pressure(&self, input: &StageInput) -> bool {
        self.profile(input).model.provider == Provider::Local
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        let (targets, context) = collect_inputs(input, self.config.source_language)?;
        anyhow::ensure!(
            !targets.is_empty() || !context.is_empty(),
            "Translation found no OCR text. Run Detection and OCR before Translation."
        );
        if targets.is_empty() {
            return Ok(true);
        }
        for target in targets {
            let Some(translation) = input.scene.component::<Translation>(target.entity)? else {
                return Ok(false);
            };
            if translation.text.value.trim().is_empty()
                || !matches!(translation.text.origin, Origin::Generated(_))
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    async fn load(&self, input: &StageInput) -> Result<()> {
        self.translator.load_model(&self.profile(input).model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let profile = self.profile(&input).clone();
        if input.target() == StageTarget::Chapter {
            let chapter = ChapterTranslation::from_input(
                &input,
                &profile,
                self.config.source_language,
                self.config.target_language,
            )?;
            let (provider, variants) = self
                .translator
                .translate_variants(&profile.model, profile.generation, chapter.request.clone())
                .await?;
            let resolved = resolve_copyfit_translations(
                &self.translator,
                &profile,
                self.config.source_language,
                self.config.target_language,
                &chapter.targets,
                variants,
            )
            .await;
            return chapter.patch_from_translations(resolved, provider);
        }
        let (targets, context) = collect_inputs(&input, self.config.source_language)?;
        anyhow::ensure!(
            !targets.is_empty() || !context.is_empty(),
            "Translation found no OCR text. Run Detection and OCR before Translation."
        );
        if targets.is_empty() {
            return finish(input.scene.edit());
        }
        let mut request = TranslationRequest::new(
            targets.iter().map(|target| target.source.clone()),
            self.config.target_language,
        )
        .with_source_language(self.config.source_language)
        .with_segment_metadata(targets.iter().map(|target| target.metadata.clone()))?
        .with_segment_ids(
            targets
                .iter()
                .map(|target| stable_segment_id(&target.metadata))
                .collect::<Vec<_>>(),
        )?
        .with_context(context);
        if let Some(instructions) = profile.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if Translator::supports_vision(&profile.model, &profile.generation)
            && let Some(image) = input
                .images
                .get(&input.scene, input.page()?, "source")
                .await?
        {
            request = request.with_image(image);
        }
        let (provider, variants) = self
            .translator
            .translate_variants(&profile.model, profile.generation, request)
            .await?;
        let translated = resolve_copyfit_translations(
            &self.translator,
            &profile,
            self.config.source_language,
            self.config.target_language,
            &targets,
            variants,
        )
        .await;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for target in &targets {
            edit.observe::<SourceText>(target.entity)?;
            edit.observe::<Translation>(target.entity)?;
        }
        for (target, text) in targets.into_iter().zip(translated) {
            let text = if target.source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                target.entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

fn target_font_config(target: &Target) -> CopyfitFontConfig {
    let mut cfg = CopyfitFontConfig {
        preferred_font: target.preferred_font.clone(),
        font_size: target.explicit_font_size,
        ..CopyfitFontConfig::default()
    };
    if let (Some(w), Some(h)) = (
        target.metadata.bubble_width_px,
        target.metadata.bubble_height_px,
    ) {
        let area = (w as f32) * (h as f32);
        cfg.target_font_size = if area < 4500.0 {
            12.5
        } else if area < 12000.0 {
            13.0
        } else {
            14.0
        };
    }
    cfg
}

fn evaluate_target_copyfit(
    target: &Target,
    variants: &TranslationVariants,
) -> Option<(CopyfitJudgeInput, Option<String>)> {
    if target.explicit_font_size.is_some() || target.metadata.is_sound_effect {
        return None;
    }
    let polygon = target.bubble_polygon.as_deref()?;
    if polygon.len() < 3 {
        return None;
    }

    let font_cfg = target_font_config(target);
    let full_diag = measure_text_fit(&variants.full, polygon, &font_cfg);

    // Fast path: if `full` fits cleanly at or above target_font_size with no ugly wrap
    // and is not in a multi-lobe figure-eight balloon that benefits from splitting, accept immediately.
    if !full_diag.overflow
        && full_diag.resulting_font_size >= full_diag.target_font_size
        && !full_diag.has_ugly_single_char_wrap
        && full_diag.lobe_count < 2
    {
        return None;
    }

    let compact_metrics = variants
        .compact
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|text| {
            let diag = measure_text_fit(text, polygon, &font_cfg);
            CopyfitCandidateMetrics::from_diagnostics(text, &diag)
        });

    let ultra_short_metrics = variants
        .ultra_short
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|text| {
            let diag = measure_text_fit(text, polygon, &font_cfg);
            CopyfitCandidateMetrics::from_diagnostics(text, &diag)
        });

    let (lobe_split_metrics, lobe_split_text) = if full_diag.lobe_count >= 2 {
        if let Some((part1, part2)) = split_text_for_lobes(&variants.full) {
            let joined = format!("{part1}\n\n{part2}");
            let diag = measure_text_fit(&joined, polygon, &font_cfg);
            (
                Some(CopyfitCandidateMetrics::from_diagnostics(&joined, &diag)),
                Some(joined),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let region_index = target
        .metadata
        .order_index
        .unwrap_or(0);
    let input = CopyfitJudgeInput {
        region_index,
        source_text: target.source.clone(),
        bubble_width_px: target.metadata.bubble_width_px.unwrap_or(120) as f32,
        bubble_height_px: target.metadata.bubble_height_px.unwrap_or(120) as f32,
        lobe_count: full_diag.lobe_count,
        full: CopyfitCandidateMetrics::from_diagnostics(&variants.full, &full_diag),
        compact: compact_metrics,
        ultra_short: ultra_short_metrics,
        lobe_split: lobe_split_metrics,
    };
    Some((input, lobe_split_text))
}

fn apply_verdict_choice(
    choice: CopyfitVariantChoice,
    variants: &TranslationVariants,
    lobe_split_text: Option<&str>,
) -> String {
    match choice {
        CopyfitVariantChoice::Full | CopyfitVariantChoice::NeedsRegeneration => {
            variants.full.clone()
        }
        CopyfitVariantChoice::Compact => variants
            .compact
            .clone()
            .unwrap_or_else(|| variants.full.clone()),
        CopyfitVariantChoice::UltraShort => variants
            .ultra_short
            .clone()
            .or_else(|| variants.compact.clone())
            .unwrap_or_else(|| variants.full.clone()),
        CopyfitVariantChoice::LobeSplit => lobe_split_text
            .map(str::to_owned)
            .unwrap_or_else(|| variants.full.clone()),
    }
}

fn select_best_copyfit_variants_sync(
    targets: &[Target],
    variants: Vec<TranslationVariants>,
) -> Vec<String> {
    targets
        .iter()
        .zip(variants)
        .map(|(target, variant)| {
            let Some((judge_input, lobe_split_text)) = evaluate_target_copyfit(target, &variant)
            else {
                return variant.full;
            };
            let verdict = JevClient::deterministic_copyfit_verdict(&judge_input);
            if verdict.choice == CopyfitVariantChoice::NeedsRegeneration {
                // In offline/manual import mode without an LLM retry, pick the shortest available variant.
                variant
                    .ultra_short
                    .or(variant.compact)
                    .unwrap_or(variant.full)
            } else {
                apply_verdict_choice(verdict.choice, &variant, lobe_split_text.as_deref())
            }
        })
        .collect()
}

async fn resolve_copyfit_translations(
    translator: &Translator,
    profile: &TranslationProfile,
    source_language: Language,
    target_language: Language,
    targets: &[Target],
    variants: Vec<TranslationVariants>,
) -> Vec<String> {
    let mut selected: Vec<String> = variants.iter().map(|v| v.full.clone()).collect();
    let mut batch_inputs: Vec<CopyfitJudgeInput> = Vec::new();
    let mut evaluated_map: Vec<(usize, Option<String>)> = Vec::new();

    for (idx, (target, variant)) in targets.iter().zip(variants.iter()).enumerate() {
        if let Some((mut judge_input, lobe_split_text)) = evaluate_target_copyfit(target, variant) {
            judge_input.region_index = idx;
            batch_inputs.push(judge_input);
            evaluated_map.push((idx, lobe_split_text));
        }
    }

    if batch_inputs.is_empty() {
        return selected;
    }

    let jev = JevClient::from_config();
    let verdicts = jev.judge_copyfitting_batch(&batch_inputs).await;
    let mut regen_indices: Vec<usize> = Vec::new();

    for ((target_idx, lobe_split_text), verdict) in evaluated_map.into_iter().zip(verdicts) {
        let variant = &variants[target_idx];
        if verdict.choice == CopyfitVariantChoice::NeedsRegeneration {
            // Provisionally pick the most compact non-empty candidate in case retry is skipped or fails.
            selected[target_idx] = variant
                .ultra_short
                .clone()
                .or_else(|| variant.compact.clone())
                .unwrap_or_else(|| variant.full.clone());
            regen_indices.push(target_idx);
        } else {
            selected[target_idx] =
                apply_verdict_choice(verdict.choice, variant, lobe_split_text.as_deref());
        }
    }

    // Targeted 1-pass LLM copyfitting retry for segments that still overflowed or had <11.5pt font size.
    if !regen_indices.is_empty() {
        let retry_sources: Vec<String> = regen_indices
            .iter()
            .map(|&idx| targets[idx].source.clone())
            .collect();
        let retry_metadata: Vec<TranslationSegmentMetadata> = regen_indices
            .iter()
            .map(|&idx| {
                let mut meta = targets[idx].metadata.clone();
                let current_len = selected[idx].chars().count();
                let strict_max_chars = ((current_len as f32) * 0.65).round().max(8.0) as usize;
                meta.max_chars_hint = Some(
                    meta.max_chars_hint
                        .map_or(strict_max_chars, |h| h.min(strict_max_chars)),
                );
                let width_px = meta.bubble_width_px.unwrap_or(80) as f32;
                let strict_word_len = ((width_px / 10.5).floor() as usize).clamp(4, 8);
                meta.max_word_len_hint = Some(
                    meta.max_word_len_hint
                        .map_or(strict_word_len, |w| w.min(strict_word_len)),
                );
                meta
            })
            .collect();
        let retry_ids: Vec<String> = retry_metadata
            .iter()
            .map(stable_segment_id)
            .collect();

        let retry_instructions = format!(
            "{}\nCRITICAL COPYFITTING RETRY: The previous translation overflowed a narrow speech bubble or resulted in an unreadable tiny font (<11.5pt). Rewrite each segment into an ultra-compact, natural manga phrase strictly within `max_chars_hint` characters and using short words within `max_word_len_hint` letters.",
            profile.instructions.as_deref().unwrap_or("")
        );

        if let Ok(req) = TranslationRequest::new(retry_sources, target_language)
            .with_source_language(source_language)
            .with_segment_metadata(retry_metadata)
            .and_then(|r| r.with_segment_ids(retry_ids))
        {
            let req = req.with_instructions(retry_instructions.trim());
            if let Ok((_, retry_variants)) = translator
                .translate_variants(&profile.model, profile.generation, req)
                .await
            {
                for (&target_idx, retry_var) in regen_indices.iter().zip(retry_variants) {
                    let target = &targets[target_idx];
                    if let Some(polygon) = target.bubble_polygon.as_deref() {
                        let font_cfg = target_font_config(target);
                        let mut best_text = selected[target_idx].clone();
                        let mut best_diag = measure_text_fit(&best_text, polygon, &font_cfg);

                        for candidate in [
                            Some(retry_var.full.as_str()),
                            retry_var.compact.as_deref(),
                            retry_var.ultra_short.as_deref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            let candidate = candidate.trim();
                            if candidate.is_empty() {
                                continue;
                            }
                            let diag = measure_text_fit(candidate, polygon, &font_cfg);
                            let better_overflow = best_diag.overflow && !diag.overflow;
                            let same_overflow_bigger_font = best_diag.overflow == diag.overflow
                                && (diag.resulting_font_size > best_diag.resulting_font_size + 0.25
                                    || (diag.resulting_font_size
                                        >= best_diag.resulting_font_size - 0.1
                                        && best_diag.has_ugly_single_char_wrap
                                        && !diag.has_ugly_single_char_wrap));
                            if better_overflow || same_overflow_bigger_font {
                                best_text = candidate.to_owned();
                                best_diag = diag;
                            }
                        }
                        selected[target_idx] = best_text;
                    }
                }
            }
        }
    }

    selected
}

fn collect_inputs(
    input: &StageInput,
    source_language: Language,
) -> Result<(Vec<Target>, Vec<TranslationContext>)> {
    let chapter = input.target() == StageTarget::Chapter;
    let mut targets = Vec::new();
    let mut context = Vec::new();
    for (page_index, page) in input.pages().iter().copied().enumerate() {
        let Some(group) = input.scene.page(page)?.text_group()? else {
            continue;
        };
        let mut entries = Vec::new();
        for layer in group.text_layers()? {
            if !chapter && !input.contains_entity(layer.id())? {
                continue;
            }
            let content = layer.content()?;
            let Some(source) = content.source()? else {
                continue;
            };
            if source.text.value.trim().is_empty() {
                continue;
            }
            let source_region = content.source_region()?;
            if chapter && source_region.is_none() {
                anyhow::bail!(
                    "Translation found OCR text '{}' without a source region on page {}",
                    source.text.value,
                    page
                );
            }
            let bounds = match source_region.as_ref() {
                Some(region) => crate::scope::geometry_extents(&region.geometry()?),
                None => None,
            };
            let page_id = page.to_string();
            let region_id = source_region
                .map(|region| region.id().to_string())
                .or_else(|| (!chapter).then(|| content.id().to_string()));
            let role = content.role()?.map(|role| role.role);
            let balloon_target = layer.balloon_target()?;
            let bubble_id = balloon_target
                .as_ref()
                .map(|bubble| bubble.id().to_string());
            let is_sound_effect = role
                .as_deref()
                .is_some_and(|role| role.ends_with(".onomatopoeia") || role == "onomatopoeia");
            if let Some(translation) = input.scene.component::<Translation>(content.id())?
                && matches!(translation.text.origin, Origin::User)
            {
                entries.push(OrderedEntry {
                    entry: TranslationEntry::Context(TranslationContext {
                        source: source.text.value.clone(),
                        translation: translation.text.value,
                    }),
                    bounds,
                });
                continue;
            }

            let typography = input.scene.component::<Typography>(layer.id())?;
            let preferred_font = typography
                .as_ref()
                .and_then(|t| t.preferred_font.clone());
            let explicit_font_size = typography.as_ref().and_then(|t| t.size);

            let (
                bubble_polygon,
                bubble_width_px,
                bubble_height_px,
                max_chars_hint,
                max_word_len_hint,
            ) = if let Some(bubble) = balloon_target.as_ref()
                && let Ok(geom) = bubble.geometry()
                && let Some(frame) = geometry_frame(&geom)
            {
                let poly = contour(&geom, frame);
                let w = frame.bounds.width.round().max(1.0) as u32;
                let h = frame.bounds.height.round().max(1.0) as u32;
                // Only attach copyfit constraints when the bubble is narrow or compact enough
                // that standard Russian/English expansions risk shrinking below 13pt.
                let is_constrained = w < 115 || (w * h) < 11_000;
                let max_chars = is_constrained.then(|| {
                    let usable_area = (w as f32 * h as f32) * 0.52;
                    ((usable_area / 145.0).round() as usize).clamp(8, 120)
                });
                let max_word = (w < 105).then(|| {
                    ((w as f32 / 9.5).floor() as usize).clamp(5, 10)
                });
                (Some(poly), Some(w), Some(h), max_chars, max_word)
            } else {
                (None, None, None, None, None)
            };

            entries.push(OrderedEntry {
                entry: TranslationEntry::Target(Target {
                    entity: content.id(),
                    source: source.text.value,
                    page: page_index + 1,
                    metadata: TranslationSegmentMetadata {
                        role,
                        inside_balloon: bubble_id.is_some(),
                        page_id: Some(page_id),
                        region_id,
                        bubble_id,
                        order_index: None,
                        unit_id: None,
                        unit_type: None,
                        is_sound_effect,
                        bubble_width_px,
                        bubble_height_px,
                        max_chars_hint,
                        max_word_len_hint,
                    },
                    bubble_polygon,
                    preferred_font,
                    explicit_font_size,
                }),
                bounds,
            });
        }
        order_entries(
            &mut entries,
            page_layout_containers(input, page)?,
            source_language,
        );
        let mut page_order_index = 0;
        for entry in entries {
            match entry.entry {
                TranslationEntry::Target(mut target) => {
                    target.metadata.order_index = Some(page_order_index);
                    page_order_index += 1;
                    targets.push(target);
                }
                TranslationEntry::Context(reference) => context.push(reference),
            }
        }
    }
    Ok((targets, context))
}

fn stable_segment_id(metadata: &TranslationSegmentMetadata) -> String {
    match (&metadata.page_id, &metadata.region_id) {
        (Some(page), Some(region)) => format!("page:{page}::region:{region}"),
        _ => "segment:missing-identity".to_owned(),
    }
}

fn adaptive_units(targets: &mut [Target]) -> Result<Vec<TranslationUnit>> {
    let mut units = Vec::new();
    let mut start = 0;
    while start < targets.len() {
        let page_id = targets[start]
            .metadata
            .page_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("chapter translation target has no page_id"))?;
        let end = targets[start..]
            .iter()
            .position(|target| target.metadata.page_id.as_deref() != Some(page_id.as_str()))
            .map_or(targets.len(), |offset| start + offset);
        units.extend(adaptive_page_units(&mut targets[start..end], &page_id)?);
        start = end;
    }
    Ok(units)
}

fn adaptive_page_units(targets: &mut [Target], page_id: &str) -> Result<Vec<TranslationUnit>> {
    let page_index = targets
        .first()
        .map(|target| target.page)
        .unwrap_or_default();
    let mut units = Vec::new();
    let mut active_bubble = None::<String>;
    let mut active_caption = false;
    let mut active_fallback = false;
    for target in targets {
        let role = short_role(target.metadata.role.as_deref());
        let (unit_type, reuse_active) = if target.metadata.is_sound_effect {
            // Sound effects are deliberately isolated. A role-less or detected
            // bubble must never make an SFX part of dialogue context.
            active_bubble = None;
            active_caption = false;
            active_fallback = false;
            ("page", false)
        } else if let Some(bubble) = target.metadata.bubble_id.as_deref() {
            let reuse = active_bubble.as_deref() == Some(bubble);
            active_bubble = Some(bubble.to_owned());
            active_caption = false;
            active_fallback = false;
            ("dialogue_group", reuse)
        } else if matches!(role, Some("caption" | "narration")) {
            let reuse = active_caption;
            active_bubble = None;
            active_caption = true;
            active_fallback = false;
            ("caption_block", reuse)
        } else {
            let reuse = active_fallback;
            active_bubble = None;
            active_caption = false;
            active_fallback = true;
            ("page", reuse)
        };

        let unit_index = if reuse_active {
            units.len() - 1
        } else {
            let index = units.len();
            units.push(TranslationUnit {
                id: format!("unit-page-{page_index:04}-{index:04}"),
                page_id: page_id.to_owned(),
                page_index,
                unit_type: unit_type.to_owned(),
                region_ids: Vec::new(),
                segment_ids: Vec::new(),
            });
            index
        };
        let segment_id = stable_segment_id(&target.metadata);
        let region_id = target
            .metadata
            .region_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("chapter translation target has no region_id"))?;
        let unit = &mut units[unit_index];
        unit.region_ids.push(region_id);
        unit.segment_ids.push(segment_id);
        target.metadata.unit_id = Some(unit.id.clone());
        target.metadata.unit_type = Some(unit.unit_type.clone());
    }
    Ok(units)
}

fn page_units(targets: &mut [Target]) -> Result<Vec<TranslationUnit>> {
    let mut units = Vec::new();
    let mut start = 0;
    while start < targets.len() {
        let page_id = targets[start]
            .metadata
            .page_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("chapter translation target has no page_id"))?;
        let end = targets[start..]
            .iter()
            .position(|target| target.metadata.page_id.as_deref() != Some(page_id.as_str()))
            .map_or(targets.len(), |offset| start + offset);
        let page_index = targets[start].page;
        let mut unit = TranslationUnit {
            id: format!("unit-page-{page_index:04}-0000"),
            page_id,
            page_index,
            unit_type: "page".to_owned(),
            region_ids: Vec::new(),
            segment_ids: Vec::new(),
        };
        for target in &mut targets[start..end] {
            let region_id =
                target.metadata.region_id.clone().ok_or_else(|| {
                    anyhow::anyhow!("chapter translation target has no region_id")
                })?;
            let segment_id = stable_segment_id(&target.metadata);
            unit.region_ids.push(region_id);
            unit.segment_ids.push(segment_id);
            target.metadata.unit_id = Some(unit.id.clone());
            target.metadata.unit_type = Some(unit.unit_type.clone());
        }
        units.push(unit);
        start = end;
    }
    Ok(units)
}

fn short_role(role: Option<&str>) -> Option<&str> {
    role.map(|role| role.rsplit('.').next().unwrap_or(role))
}

fn page_layout_containers(input: &StageInput, page: EntityId) -> Result<Vec<LayoutItem>> {
    let mut items = Vec::new();
    for entity in input.scene.descendants(page)? {
        if input.target() != StageTarget::Chapter && !input.contains_entity(entity.id())? {
            continue;
        }
        let Some(region) = input.scene.component::<Region>(entity.id())? else {
            continue;
        };
        let kind = if region.kind == PanelRegion::kind() {
            LayoutKind::Panel
        } else if region.kind == BubbleRegion::kind() {
            LayoutKind::Bubble
        } else {
            continue;
        };
        let Some(geometry) = input.scene.component::<Geometry>(entity.id())? else {
            continue;
        };
        let Some((left, top, right, bottom)) = crate::scope::geometry_extents(&geometry) else {
            continue;
        };
        items.push(LayoutItem {
            kind,
            bounds: [left, top, right, bottom],
        });
    }
    Ok(items)
}

fn order_entries(
    entries: &mut Vec<OrderedEntry>,
    mut layout: Vec<LayoutItem>,
    source_language: Language,
) {
    if !matches!(
        source_language,
        Language::Japanese | Language::ChineseSimplified | Language::ChineseTraditional
    ) {
        return;
    }

    let bounded = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| entry.bounds.map(|bounds| (index, bounds)))
        .collect::<Vec<_>>();
    if bounded.len() < 2 {
        return;
    }

    let first_text = layout.len();
    layout.extend(bounded.iter().map(|(_, bounds)| LayoutItem {
        kind: LayoutKind::Text,
        bounds: [bounds.0, bounds.1, bounds.2, bounds.3],
    }));
    let ordered_bounded = reading_order(&layout, source_language)
        .into_iter()
        .filter_map(|index| index.checked_sub(first_text))
        .map(|index| bounded[index].0)
        .collect::<Vec<_>>();

    let mut next = ordered_bounded.into_iter();
    let order = (0..entries.len())
        .map(|index| {
            if entries[index].bounds.is_some() {
                next.next().expect("bounded order contains each entry once")
            } else {
                index
            }
        })
        .collect::<Vec<_>>();
    let mut values = std::mem::take(entries)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    *entries = order
        .into_iter()
        .map(|index| values[index].take().expect("entry order is unique"))
        .collect();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use koharu_scene::{
        At, BubbleRegion, FlowsIn, Geometry, Inside, PageDraft, RecognizedFrom, TextLayout,
        TextLayoutKind, TextRegion,
    };

    use super::*;
    use crate::ImageCache;

    #[tokio::test]
    async fn translation_inputs_repair_old_japanese_layer_order() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();
        for (source, x, y) in [
            ("left-bottom", 10.0, 100.0),
            ("right-bottom", 80.0, 100.0),
            ("left-top", 10.0, 10.0),
            ("right-top", 80.0, 10.0),
        ] {
            add_text(&mut edit, page, source, (x, y, 20.0, 30.0));
        }
        session.commit(edit.finish().unwrap()).await.unwrap();

        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );
        let (targets, context) = collect_inputs(&input, Language::Japanese).unwrap();

        assert_eq!(
            targets
                .iter()
                .map(|target| (target.page, target.source.as_str()))
                .collect::<Vec<_>>(),
            [
                (1, "right-top"),
                (1, "right-bottom"),
                (1, "left-top"),
                (1, "left-bottom"),
            ]
        );
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn translation_inputs_repair_old_panel_aware_japanese_order() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();

        let _left_panel = edit
            .add_analysis_region::<PanelRegion>(
                page,
                At::End,
                &Geometry::rectangle(0.0, 0.0, 55.0, 180.0),
                None,
            )
            .unwrap();
        let left_bubble = edit
            .add_analysis_region::<BubbleRegion>(
                page,
                At::End,
                &Geometry::rectangle(5.0, 10.0, 40.0, 60.0),
                None,
            )
            .unwrap();
        let (_, left_region, left_layer) =
            add_text(&mut edit, page, "left-panel", (15.0, 20.0, 20.0, 30.0));
        edit.relate::<Inside>(left_region, left_bubble).unwrap();
        edit.relate::<FlowsIn>(left_layer, left_bubble).unwrap();

        let _right_panel = edit
            .add_analysis_region::<PanelRegion>(
                page,
                At::End,
                &Geometry::rectangle(65.0, 0.0, 55.0, 180.0),
                None,
            )
            .unwrap();
        let right_bubble = edit
            .add_analysis_region::<BubbleRegion>(
                page,
                At::End,
                &Geometry::rectangle(70.0, 100.0, 40.0, 60.0),
                None,
            )
            .unwrap();
        let (_, right_region, right_layer) =
            add_text(&mut edit, page, "right-panel", (80.0, 110.0, 20.0, 30.0));
        edit.relate::<Inside>(right_region, right_bubble).unwrap();
        edit.relate::<FlowsIn>(right_layer, right_bubble).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );
        let (targets, context) = collect_inputs(&input, Language::Japanese).unwrap();

        assert_eq!(
            targets
                .iter()
                .map(|target| target.source.as_str())
                .collect::<Vec<_>>(),
            ["right-panel", "left-panel"]
        );
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn translation_inputs_follow_t_shaped_japanese_panel_order() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 200.0, 200.0), At::End)
            .unwrap();
        for (x, y, width, height) in [
            (0.0, 0.0, 200.0, 80.0),
            (0.0, 100.0, 90.0, 100.0),
            (110.0, 100.0, 90.0, 100.0),
        ] {
            edit.add_analysis_region::<PanelRegion>(
                page,
                At::End,
                &Geometry::rectangle(x, y, width, height),
                None,
            )
            .unwrap();
        }
        add_text(&mut edit, page, "left-bottom", (20.0, 120.0, 50.0, 40.0));
        add_text(&mut edit, page, "right-bottom", (130.0, 120.0, 50.0, 40.0));
        add_text(&mut edit, page, "top", (80.0, 20.0, 40.0, 40.0));
        session.commit(edit.finish().unwrap()).await.unwrap();

        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );
        let (targets, context) = collect_inputs(&input, Language::Japanese).unwrap();

        assert_eq!(
            targets
                .iter()
                .map(|target| target.source.as_str())
                .collect::<Vec<_>>(),
            ["top", "right-bottom", "left-bottom"]
        );
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn region_translation_ignores_layout_containers_outside_its_scope() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 200.0, 100.0), At::End)
            .unwrap();
        for (x, width) in [(0.0, 90.0), (110.0, 90.0)] {
            edit.add_analysis_region::<PanelRegion>(
                page,
                At::End,
                &Geometry::rectangle(x, 0.0, width, 100.0),
                None,
            )
            .unwrap();
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            Some(crate::Bounds {
                x: 100.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            }),
            Arc::new(ImageCache::default()),
            None,
        );

        let containers = page_layout_containers(&input, page).unwrap();

        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].bounds, [110.0, 0.0, 200.0, 100.0]);
    }

    #[tokio::test]
    async fn chapter_translation_keeps_project_page_order() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let first_page = edit
            .add_page(PageDraft::new("first", 120.0, 180.0), At::End)
            .unwrap();
        add_text(
            &mut edit,
            first_page,
            "first-page",
            (10.0, 10.0, 20.0, 30.0),
        );
        let second_page = edit
            .add_page(PageDraft::new("second", 120.0, 180.0), At::End)
            .unwrap();
        add_text(
            &mut edit,
            second_page,
            "second-page",
            (80.0, 10.0, 20.0, 30.0),
        );
        session.commit(edit.finish().unwrap()).await.unwrap();

        let input = StageInput::chapter(session.snapshot(), Arc::from([first_page, second_page]));
        let (targets, context) = collect_inputs(&input, Language::Japanese).unwrap();

        assert_eq!(
            targets
                .iter()
                .map(|target| (target.page, target.source.as_str()))
                .collect::<Vec<_>>(),
            [(1, "first-page"), (2, "second-page")]
        );
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn manual_chapter_import_is_strict_and_preserves_user_translations() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();
        let (target, region, _) = add_text(&mut edit, page, "source", (80.0, 10.0, 20.0, 30.0));
        let (reference, _, _) =
            add_text(&mut edit, page, "manual source", (10.0, 10.0, 20.0, 30.0));
        edit.set(
            reference,
            &Translation {
                text: Authored::user("manual translation".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let chapter =
            ChapterTranslation::from_snapshot(session.snapshot(), &TranslationConfig::default())
                .unwrap();
        let response = serde_json::json!({
            "translations": [{
                "id": format!("page:{page}::region:{region}"),
                "page_id": page.to_string(),
                "region_id": region.to_string(),
                "source_text": "source",
                "translated_text": "перевод"
            }],
            "notes": ""
        });
        let patch = chapter.patch_from_response(&response.to_string()).unwrap();
        session.commit(patch).await.unwrap();

        let snapshot = session.snapshot();
        assert_eq!(
            snapshot
                .component::<Translation>(target)
                .unwrap()
                .unwrap()
                .text
                .value,
            "перевод"
        );
        let manual = snapshot
            .component::<Translation>(reference)
            .unwrap()
            .unwrap();
        assert_eq!(manual.text.value, "manual translation");
        assert!(matches!(manual.text.origin, Origin::User));

        let changed = response.to_string().replace("\"source\"", "\"changed\"");
        assert!(chapter.patch_from_response(&changed).is_err());
    }

    #[tokio::test]
    async fn completed_page_translations_are_skipped() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();
        let (generated_content, _, _) = add_text(
            &mut edit,
            page,
            "generated source",
            (80.0, 10.0, 20.0, 30.0),
        );
        let (user_content, _, _) =
            add_text(&mut edit, page, "user source", (10.0, 10.0, 20.0, 30.0));
        edit.set(
            user_content,
            &Translation {
                text: Authored::user("user translation".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        let generated = generation(PRODUCER, "test-translator").unwrap();
        let mut edit = session.snapshot().edit_as(generated.clone());
        edit.set(
            generated_content,
            &Translation {
                text: Authored::generated("generated translation".to_owned(), generated),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert!(processor().skip(&page_input(&session, page)).unwrap());
    }

    #[tokio::test]
    async fn translation_without_ocr_text_reports_an_actionable_error() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let error = processor()
            .skip(&page_input(&session, page))
            .unwrap_err()
            .to_string();

        assert!(error.contains("no OCR text"));
        assert!(error.contains("Detection and OCR"));
    }

    #[tokio::test]
    async fn manual_only_translation_context_remains_a_valid_noop() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
            .unwrap();
        let (content, _, _) = add_text(&mut edit, page, "source", (10.0, 10.0, 20.0, 30.0));
        edit.set(
            content,
            &Translation {
                text: Authored::user("manual translation".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert!(processor().skip(&page_input(&session, page)).unwrap());
    }

    #[tokio::test]
    async fn missing_or_empty_generated_page_translation_is_not_skipped() {
        for translation in [None, Some("")] {
            let mut session = koharu_scene::Session::memory().await.unwrap();
            let mut edit = session.snapshot().edit();
            let page = edit
                .add_page(PageDraft::new("page", 120.0, 180.0), At::End)
                .unwrap();
            let (content, _, _) = add_text(&mut edit, page, "source", (10.0, 10.0, 20.0, 30.0));
            session.commit(edit.finish().unwrap()).await.unwrap();
            if let Some(translation) = translation {
                let generated = generation(PRODUCER, "test-translator").unwrap();
                let mut edit = session.snapshot().edit_as(generated.clone());
                edit.set(
                    content,
                    &Translation {
                        text: Authored::generated(translation.to_owned(), generated),
                        language: None,
                    },
                )
                .unwrap();
                session.commit(edit.finish().unwrap()).await.unwrap();
            }

            if translation.is_some() {
                let stored = session
                    .snapshot()
                    .component::<Translation>(content)
                    .unwrap()
                    .unwrap();
                assert!(matches!(stored.text.origin, Origin::Generated(_)));
            }

            assert!(
                !processor().skip(&page_input(&session, page)).unwrap(),
                "target with {translation:?} translation was skipped"
            );
        }
    }

    #[tokio::test]
    async fn chapter_skip_checks_every_page() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let first_page = edit
            .add_page(PageDraft::new("first", 120.0, 180.0), At::End)
            .unwrap();
        let (first_content, _, _) = add_text(
            &mut edit,
            first_page,
            "first source",
            (10.0, 10.0, 20.0, 30.0),
        );
        let second_page = edit
            .add_page(PageDraft::new("second", 120.0, 180.0), At::End)
            .unwrap();
        add_text(
            &mut edit,
            second_page,
            "second source",
            (80.0, 10.0, 20.0, 30.0),
        );
        session.commit(edit.finish().unwrap()).await.unwrap();
        let generated = generation(PRODUCER, "test-translator").unwrap();
        let mut edit = session.snapshot().edit_as(generated.clone());
        edit.set(
            first_content,
            &Translation {
                text: Authored::generated("first translation".to_owned(), generated),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let input = StageInput::chapter(session.snapshot(), Arc::from([first_page, second_page]));
        assert!(!processor().skip(&input).unwrap());
    }

    #[test]
    fn japanese_column_tolerance_uses_region_width() {
        let mut entries = vec![
            target("right-lower", (500.0, 500.0, 510.0, 510.0)),
            target("right-upper", (450.0, 450.0, 460.0, 460.0)),
            target("left", (0.0, 290.0, 300.0, 300.0)),
        ];

        order_entries(&mut entries, Vec::new(), Language::Japanese);

        assert_eq!(sources(&entries), ["right-upper", "right-lower", "left"]);
    }

    #[test]
    fn japanese_translation_reads_real_panel_rows_right_to_left() {
        let mut entries = vec![
            target("left-top", (537.63, 430.70, 587.76, 575.53)),
            target("right-top", (1106.99, 70.47, 1189.22, 218.39)),
            target("middle", (560.91, 875.24, 613.39, 945.51)),
            target("right-bottom", (1037.99, 1350.16, 1100.18, 1493.89)),
            target("left-bottom", (152.12, 1158.30, 531.36, 1522.63)),
        ];
        let panels = [
            [103.53, 0.86, 680.85, 757.97],
            [693.52, 1.93, 1280.82, 757.30],
            [4.24, 752.93, 1278.80, 1211.98],
            [701.62, 1239.78, 1175.73, 1702.67],
            [-0.65, 1214.37, 698.22, 1804.58],
        ]
        .map(|bounds| LayoutItem {
            kind: LayoutKind::Panel,
            bounds,
        })
        .to_vec();

        order_entries(&mut entries, panels, Language::Japanese);

        assert_eq!(
            sources(&entries),
            [
                "right-top",
                "left-top",
                "middle",
                "right-bottom",
                "left-bottom",
            ]
        );
    }

    #[test]
    fn western_translation_keeps_canonical_layer_order() {
        let mut entries = vec![
            target("left-bottom", (10.0, 100.0, 30.0, 130.0)),
            target("right-top", (80.0, 10.0, 100.0, 40.0)),
        ];

        order_entries(&mut entries, Vec::new(), Language::English);

        assert_eq!(sources(&entries), ["left-bottom", "right-top"]);
    }

    #[test]
    fn adaptive_units_group_one_detected_bubble() {
        let mut targets = vec![
            adaptive_target("a", 1, "bubble-a", Some("dialogue"), false),
            adaptive_target("b", 2, "bubble-a", Some("dialogue"), false),
        ];

        let units = adaptive_units(&mut targets).unwrap();

        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, "dialogue_group");
        assert_eq!(units[0].region_ids, vec!["1".to_owned(), "2".to_owned()]);
        assert_eq!(
            units[0].segment_ids,
            vec![
                "page:page-a::region:1".to_owned(),
                "page:page-a::region:2".to_owned()
            ]
        );
        assert_eq!(targets[0].metadata.unit_id, targets[1].metadata.unit_id);
    }

    #[test]
    fn adaptive_units_group_consecutive_captions() {
        let mut targets = vec![
            adaptive_target("caption one", 1, "", Some("caption"), false),
            adaptive_target("caption two", 2, "", Some("narration"), false),
        ];

        let units = adaptive_units(&mut targets).unwrap();

        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, "caption_block");
        assert_eq!(units[0].segment_ids.len(), 2);
    }

    #[test]
    fn adaptive_units_isolate_sound_effects_from_dialogue() {
        let mut targets = vec![
            adaptive_target("hello", 1, "bubble-a", Some("dialogue"), false),
            adaptive_target("ガタッ", 2, "", Some("onomatopoeia"), true),
            adaptive_target("goodbye", 3, "bubble-a", Some("dialogue"), false),
        ];

        let units = adaptive_units(&mut targets).unwrap();

        assert_eq!(
            units
                .iter()
                .map(|unit| unit.unit_type.as_str())
                .collect::<Vec<_>>(),
            ["dialogue_group", "page", "dialogue_group",]
        );
        assert_eq!(units[1].segment_ids.len(), 1);
        assert!(targets[1].metadata.is_sound_effect);
    }

    #[test]
    fn adaptive_units_use_page_fallback_for_unknown_roles() {
        let mut targets = vec![adaptive_target("unknown", 1, "", Some("label"), false)];

        let units = adaptive_units(&mut targets).unwrap();

        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, "page");
        assert_eq!(targets[0].metadata.unit_type.as_deref(), Some("page"));
    }

    #[test]
    fn page_only_units_keep_one_unit_per_project_page() {
        let mut targets = vec![
            adaptive_target("first", 1, "", Some("dialogue"), false),
            adaptive_target("second", 2, "", Some("dialogue"), false),
            adaptive_target("third", 3, "", Some("dialogue"), false),
        ];
        targets[0].metadata.page_id = Some("page-a".to_owned());
        targets[1].metadata.page_id = Some("page-a".to_owned());
        targets[2].metadata.page_id = Some("page-b".to_owned());
        targets[0].page = 1;
        targets[1].page = 1;
        targets[2].page = 2;

        let units = page_units(&mut targets).unwrap();

        assert_eq!(units.len(), 2);
        assert_eq!(units[0].unit_type, "page");
        assert_eq!(units[0].segment_ids.len(), 2);
        assert_eq!(units[1].page_index, 2);
    }

    fn add_text(
        edit: &mut koharu_scene::Edit,
        page: EntityId,
        source: &str,
        (x, y, width, height): (f64, f64, f64, f64),
    ) -> (EntityId, EntityId, EntityId) {
        let region = edit
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(x, y, width, height),
                None,
            )
            .unwrap();
        let content = edit.add_text_content(page, At::End).unwrap();
        let layer = edit
            .add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                },
            )
            .unwrap();
        edit.set(
            content,
            &SourceText {
                text: Authored::user(source.to_owned()),
                language: Some(LanguageTag::new("ja-JP").unwrap()),
            },
        )
        .unwrap();
        edit.relate::<RecognizedFrom>(content, region).unwrap();
        (content, region, layer)
    }

    fn processor() -> Processor {
        let translator = Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        Processor::new(TranslationConfig::default(), translator)
    }

    fn page_input(session: &koharu_scene::Session, page: EntityId) -> StageInput {
        StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        )
    }

    fn target(source: &str, bounds: (f64, f64, f64, f64)) -> OrderedEntry {
        OrderedEntry {
            entry: TranslationEntry::Target(Target {
                entity: EntityId::new(),
                source: source.to_owned(),
                page: 1,
                metadata: TranslationSegmentMetadata::default(),
                bubble_polygon: None,
                preferred_font: None,
                explicit_font_size: None,
            }),
            bounds: Some(bounds),
        }
    }

    fn adaptive_target(
        source: &str,
        region: usize,
        bubble: &str,
        role: Option<&str>,
        is_sound_effect: bool,
    ) -> Target {
        Target {
            entity: EntityId::new(),
            source: source.to_owned(),
            page: 1,
            metadata: TranslationSegmentMetadata {
                role: role.map(str::to_owned),
                page_id: Some("page-a".to_owned()),
                region_id: Some(region.to_string()),
                bubble_id: (!bubble.is_empty()).then(|| bubble.to_owned()),
                is_sound_effect,
                ..Default::default()
            },
            bubble_polygon: None,
            preferred_font: None,
            explicit_font_size: None,
        }
    }

    #[test]
    fn copyfit_selector_chooses_compact_variant_for_narrow_bubble() {
        let narrow_poly: Arc<[[f32; 2]]> = Arc::from([
            [0.0, 0.0],
            [62.0, 0.0],
            [62.0, 95.0],
            [0.0, 95.0],
        ]);
        let target_item = Target {
            entity: EntityId::new(),
            source: "本当に信じられないことだ！".to_owned(),
            page: 1,
            metadata: TranslationSegmentMetadata {
                inside_balloon: true,
                bubble_width_px: Some(62),
                bubble_height_px: Some(95),
                ..Default::default()
            },
            bubble_polygon: Some(narrow_poly),
            preferred_font: None,
            explicit_font_size: None,
        };
        let variants = TranslationVariants {
            full: "На самом деле это совершенно невероятное происшествие для всех нас!".to_owned(),
            compact: Some("Это просто невероятно!".to_owned()),
            ultra_short: Some("Невероятно!".to_owned()),
        };

        let chosen = select_best_copyfit_variants_sync(&[target_item], vec![variants]);
        assert!(
            chosen[0] == "Это просто невероятно!" || chosen[0] == "Невероятно!",
            "expected compact or ultra_short variant for narrow bubble, got: {}",
            chosen[0]
        );
    }

    fn sources(entries: &[OrderedEntry]) -> Vec<&str> {
        entries
            .iter()
            .filter_map(|entry| match &entry.entry {
                TranslationEntry::Target(target) => Some(target.source.as_str()),
                TranslationEntry::Context(_) => None,
            })
            .collect()
    }
}
