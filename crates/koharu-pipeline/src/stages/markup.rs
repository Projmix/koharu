//! Scene markup: the per-page stage that labels each text with its kind,
//! speaker and continuation. Translation reads whatever markup exists and
//! never runs this stage itself.

use std::{collections::BTreeSet, sync::Arc};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut},
    rect::Rect,
};
use koharu_scene::{
    Character, CharacterId, CharacterList, Continues, EntityId, Gender, Generation, Geometry,
    ObservedText, Origin, PageMarkup, PageMarkupMissing, PageMarkupStatus, Region, RegionSpec,
    Snapshot, SpeechKind, SpeechMarkup, TextRegion, UnboxedText, UnboxedTextReview,
};
use koharu_translator::{
    CharacterGender, Language, MarkedRegion, MarkupCharacter, MarkupKind, MarkupOutcome,
    MarkupRegion, MarkupRequest, MarkupWithheld, PreviousMarkup, Translator,
};

use super::{
    StageInput, StageProcessor,
    translation::{layout_containers, reading_permutation},
};
use crate::SceneMarkupConfig;

const PRODUCER: &str = "dev.koharu.pipeline.markup";

/// How many marked texts of the previous page the model sees for continuity.
const PREVIOUS_CONTEXT: usize = 6;

/// A page with at least this many texts is retried in two halves when the
/// whole-page answer is invalid: dense pages are where a model drops or
/// misnumbers boxes.
const SPLIT_MIN_TEXTS: usize = 8;

/// `(left, top, right, bottom)` in page units.
type Bounds = (f64, f64, f64, f64);

/// Share of an unboxed-text flag a text region must cover for the lettering
/// to count as boxed. A sound effect whose region holds only its first
/// character keeps its flag.
const BOXED_SHARE: f64 = 0.3;

struct PageText {
    content: EntityId,
    source: String,
    bounds: Bounds,
}

pub(super) struct Processor {
    config: SceneMarkupConfig,
    source_language: Language,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(
        config: SceneMarkupConfig,
        source_language: Language,
        translator: Translator,
    ) -> Self {
        Self {
            config,
            source_language,
            translator,
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self, _input: &StageInput) -> String {
        model_name(&self.config)
    }

    /// Markup holds no model of its own, so a failure has nothing to unload.
    fn unload(&self) -> bool {
        false
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        skip(input, self.source_language)
    }

    async fn load(&self, _input: &StageInput) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        process(&input, &self.config, &self.translator, self.source_language).await
    }
}

fn model_name(config: &SceneMarkupConfig) -> String {
    let provider = Translator::model(&config.model);
    match config.model.model.as_deref() {
        Some(model) => format!("{provider}: {model}"),
        None => provider.to_owned(),
    }
}

/// A page is up to date when its last complete markup saw exactly its
/// current texts. A missing result is retried on the next run.
fn skip(input: &StageInput, source_language: Language) -> Result<bool> {
    let page = input.page()?;
    let texts = page_texts(&input.scene, page, source_language)?;
    if texts.is_empty() {
        return Ok(true);
    }
    Ok(input.scene.page(page)?.markup()?.is_some_and(|markup| {
        markup.status == PageMarkupStatus::Complete && markup.observed == observed(&texts)
    }))
}

/// What one page's markup reads from the scene.
struct PageContext {
    page: EntityId,
    texts: Vec<PageText>,
    previous: Vec<(EntityId, PreviousMarkup)>,
    list: Option<CharacterList>,
}

impl PageContext {
    fn read(scene: &Snapshot, page: EntityId, source_language: Language) -> Result<Self> {
        let list = scene.project_component::<CharacterList>()?;
        Ok(Self {
            page,
            texts: page_texts(scene, page, source_language)?,
            previous: previous_markup(scene, page, source_language, list.as_ref())?,
            list,
        })
    }
}

/// Marks up one page. A withheld, invalid or failed answer does not fail the
/// run: the page records why its markup is missing, the next run retries it,
/// and translation runs without it.
async fn process(
    input: &StageInput,
    config: &SceneMarkupConfig,
    translator: &Translator,
    source_language: Language,
) -> Result<koharu_scene::Patch> {
    let context = PageContext::read(&input.scene, input.page()?, source_language)?;
    let every = (0..context.texts.len()).collect::<Vec<_>>();
    let outcome = match request(
        input,
        config,
        translator,
        source_language,
        &context,
        &every,
        &[],
        &[],
    )
    .await
    {
        Err(error)
            if invalid_markup(&error)
                && context.texts.len() >= SPLIT_MIN_TEXTS
                && !input.stop.stopped() =>
        {
            tracing::info!(page = %context.page, texts = context.texts.len(), "retrying invalid scene markup in two halves");
            request_in_halves(input, config, translator, source_language, &context).await
        }
        outcome => outcome,
    };
    // Stopping discards the uncommitted result instead of recording a miss.
    let outcome = match outcome {
        Err(error) if input.stop.stopped() => return Err(error),
        outcome => outcome,
    };
    let generated = super::generation(PRODUCER, &model_name(config))?;
    page_patch(&input.scene, &context, &generated, outcome)
}

fn invalid_markup(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<koharu_translator::Error>()
        .is_some_and(|error| matches!(error, koharu_translator::Error::InvalidMarkup { .. }))
}

/// Marks up the first half of the page, then the second half with the first
/// half's answers as earlier context, and joins them in page order.
async fn request_in_halves(
    input: &StageInput,
    config: &SceneMarkupConfig,
    translator: &Translator,
    source_language: Language,
    context: &PageContext,
) -> Result<MarkupOutcome> {
    let middle = context.texts.len() / 2;
    let first = (0..middle).collect::<Vec<_>>();
    let second = (middle..context.texts.len()).collect::<Vec<_>>();
    let (regions, new_characters, unboxed) = match request(
        input,
        config,
        translator,
        source_language,
        context,
        &first,
        &[],
        &[],
    )
    .await?
    {
        MarkupOutcome::Marked {
            regions,
            new_characters,
            unboxed,
        } => (regions, new_characters, unboxed),
        withheld => return Ok(withheld),
    };
    let earlier = earlier_markup(&context.texts, &regions);
    let rest = request(
        input,
        config,
        translator,
        source_language,
        context,
        &second,
        &earlier,
        &new_characters,
    )
    .await?;
    Ok(join_halves(regions, new_characters, unboxed, rest))
}

/// The first half's answers, shown to the second half like the previous page.
fn earlier_markup(texts: &[PageText], regions: &[MarkedRegion]) -> Vec<PreviousMarkup> {
    texts
        .iter()
        .zip(regions)
        .map(|(text, region)| PreviousMarkup {
            id: region.id.clone(),
            text: text.source.clone(),
            kind: region.kind,
            speaker: region.speaker.clone(),
        })
        .collect()
}

/// Each half saw only its own boxes, so its unboxed lettering includes the
/// other half's texts; the page patch drops whatever a text region covers.
fn join_halves(
    mut regions: Vec<MarkedRegion>,
    mut new_characters: Vec<MarkupCharacter>,
    mut unboxed: Vec<koharu_translator::UnboxedText>,
    rest: MarkupOutcome,
) -> MarkupOutcome {
    match rest {
        MarkupOutcome::Marked {
            regions: more,
            new_characters: more_characters,
            unboxed: more_unboxed,
        } => {
            regions.extend(more);
            new_characters.extend(more_characters);
            unboxed.extend(more_unboxed);
            MarkupOutcome::Marked {
                regions,
                new_characters,
                unboxed,
            }
        }
        withheld => withheld,
    }
}

/// Marks up the texts at `indices`, keeping their page numbers. `earlier`
/// and `cast` add this page's already answered texts and characters.
#[allow(clippy::too_many_arguments)]
async fn request(
    input: &StageInput,
    config: &SceneMarkupConfig,
    translator: &Translator,
    source_language: Language,
    context: &PageContext,
    indices: &[usize],
    earlier: &[PreviousMarkup],
    cast: &[MarkupCharacter],
) -> Result<MarkupOutcome> {
    let scene = &input.scene;
    let size = scene.page(context.page)?.page()?;
    let image = input
        .images
        .page_image(scene, context.page)
        .await?
        .context("scene markup needs the page source image")?;
    let boxes = indices
        .iter()
        .map(|&index| (index + 1, context.texts[index].bounds))
        .collect::<Vec<_>>();
    let scale = (
        f64::from(image.width()) / size.width,
        f64::from(image.height()) / size.height,
    );
    let image = tokio_rayon::spawn(move || draw_regions(&image, &boxes, scale)).await;
    let request = MarkupRequest {
        image: Arc::new(DynamicImage::ImageRgb8(image)),
        source_language: Some(source_language),
        regions: indices
            .iter()
            .map(|&index| MarkupRegion {
                id: region_id(index),
                text: context.texts[index].source.clone(),
                bbox: normalized(context.texts[index].bounds, size.width, size.height),
            })
            .collect(),
        previous: context
            .previous
            .iter()
            .map(|(_, markup)| markup.clone())
            .chain(earlier.iter().cloned())
            .collect(),
        characters: context
            .list
            .iter()
            .flat_map(|list| &list.characters)
            .map(|character| MarkupCharacter {
                id: character.id.to_string(),
                name: character.name.clone(),
                gender: translator_gender(character.gender),
                notes: character.notes.clone(),
            })
            .chain(cast.iter().cloned())
            .collect(),
        instructions: config.instructions.clone(),
    };
    let (_, outcome) = translator
        .mark_up(&config.model, config.generation, request)
        .await?;
    Ok(outcome)
}

fn page_patch(
    scene: &Snapshot,
    context: &PageContext,
    generated: &Generation,
    outcome: Result<MarkupOutcome>,
) -> Result<koharu_scene::Patch> {
    let page = context.page;
    let mut edit = scene.edit_as(generated.clone());
    edit.observe_subtree(page)?;
    edit.observe_project::<CharacterList>()?;
    for (content, _) in &context.previous {
        edit.observe::<SpeechMarkup>(*content)?;
    }
    let size = scene.page(page)?.page()?;
    let mut found = Vec::new();
    let status = match outcome {
        Ok(MarkupOutcome::Marked {
            regions,
            new_characters,
            unboxed,
        }) => {
            apply(
                &mut edit,
                scene,
                generated,
                context,
                regions,
                new_characters,
            )?;
            found = unboxed
                .into_iter()
                .map(|text| page_unboxed(&text, size.width, size.height))
                .collect();
            PageMarkupStatus::Complete
        }
        Ok(MarkupOutcome::Withheld(reason)) => {
            tracing::warn!(%page, ?reason, "scene markup was withheld");
            PageMarkupStatus::Missing(match reason {
                MarkupWithheld::Refused => PageMarkupMissing::Refused,
                MarkupWithheld::ContentFilter => PageMarkupMissing::ContentFilter,
                MarkupWithheld::Empty => PageMarkupMissing::Empty,
            })
        }
        Err(error) => {
            tracing::warn!(%page, error = %format!("{error:#}"), "scene markup failed");
            PageMarkupStatus::Missing(if invalid_markup(&error) {
                PageMarkupMissing::Invalid
            } else {
                PageMarkupMissing::Failed
            })
        }
    };
    edit.set(
        page,
        &PageMarkup {
            origin: Origin::Generated(generated.clone()),
            status,
            observed: observed(&context.texts),
            unboxed: uncovered(scene, page, found)?,
        },
    )?;
    super::finish(edit)
}

/// The page's unboxed-text flags that no text region covers now and the
/// user has not dismissed. Adding a region over the lettering resolves its
/// flag without editing the markup.
pub fn open_unboxed_text(scene: &Snapshot, page: EntityId) -> Result<Vec<UnboxedText>> {
    let Some(markup) = scene.page(page)?.markup()? else {
        return Ok(Vec::new());
    };
    let dismissed = scene
        .component::<UnboxedTextReview>(page)?
        .map(|review| review.dismissed)
        .unwrap_or_default();
    let mut flags = markup.unboxed;
    flags.retain(|flag| !dismissed.contains(flag));
    uncovered(scene, page, flags)
}

/// Drops flags a text region covers and repeats of an earlier flag.
fn uncovered(
    scene: &Snapshot,
    page: EntityId,
    flags: Vec<UnboxedText>,
) -> Result<Vec<UnboxedText>> {
    let mut regions = Vec::new();
    for entity in scene.descendants(page)? {
        if !entity
            .component::<Region>()?
            .is_some_and(|region| region.kind == TextRegion::kind())
        {
            continue;
        }
        if let Some(bounds) = entity
            .component::<Geometry>()?
            .as_ref()
            .and_then(crate::scope::geometry_extents)
        {
            regions.push(bounds);
        }
    }
    let mut open: Vec<UnboxedText> = Vec::new();
    for flag in flags {
        let bounds = flag_bounds(&flag);
        let area = bounds_area(bounds);
        let boxed = regions
            .iter()
            .any(|region| overlap(bounds, *region) >= BOXED_SHARE * area);
        let repeated = open.iter().any(|kept| {
            let kept = flag_bounds(kept);
            overlap(bounds, kept) >= 0.5 * area.min(bounds_area(kept))
        });
        if !boxed && !repeated {
            open.push(flag);
        }
    }
    Ok(open)
}

fn page_unboxed(text: &koharu_translator::UnboxedText, width: f64, height: f64) -> UnboxedText {
    let [left, top, right, bottom] = text.bbox.map(f64::from);
    UnboxedText {
        x: left / 1000.0 * width,
        y: top / 1000.0 * height,
        width: (right - left) / 1000.0 * width,
        height: (bottom - top) / 1000.0 * height,
        kind: scene_kind(text.kind),
    }
}

fn flag_bounds(flag: &UnboxedText) -> Bounds {
    (flag.x, flag.y, flag.x + flag.width, flag.y + flag.height)
}

fn bounds_area((left, top, right, bottom): Bounds) -> f64 {
    (right - left).max(0.0) * (bottom - top).max(0.0)
}

fn overlap(a: Bounds, b: Bounds) -> f64 {
    bounds_area((a.0.max(b.0), a.1.max(b.1), a.2.min(b.2), a.3.min(b.3)))
}

fn apply(
    edit: &mut koharu_scene::Edit,
    scene: &Snapshot,
    generated: &Generation,
    context: &PageContext,
    regions: Vec<MarkedRegion>,
    new_characters: Vec<MarkupCharacter>,
) -> Result<()> {
    let texts = &context.texts;
    // A user-owned cast is the user's decision; markup only assigns the
    // speakers it already names.
    let user_owned = context
        .list
        .as_ref()
        .is_some_and(|list| matches!(list.origin, Origin::User));
    let mut cast = context
        .list
        .as_ref()
        .map(|list| list.characters.clone())
        .unwrap_or_default();
    if !user_owned && !new_characters.is_empty() {
        for character in new_characters {
            cast.push(Character {
                id: CharacterId::new(character.id)?,
                name: character.name.trim().to_owned(),
                gender: scene_gender(character.gender),
                notes: character.notes,
            });
        }
        edit.set_project(&CharacterList {
            origin: Origin::Generated(generated.clone()),
            characters: cast.clone(),
        })?;
    }
    let speakers = cast
        .iter()
        .map(|character| character.id.as_str())
        .collect::<BTreeSet<_>>();

    for (index, (text, region)) in texts.iter().zip(regions).enumerate() {
        debug_assert_eq!(region.id, region_id(index));
        if scene
            .component::<SpeechMarkup>(text.content)?
            .is_some_and(|markup| matches!(markup.origin, Origin::User))
        {
            continue;
        }
        let speaker = region
            .speaker
            .filter(|speaker| speakers.contains(speaker.as_str()))
            .map(CharacterId::new)
            .transpose()?;
        edit.set(
            text.content,
            &SpeechMarkup {
                origin: Origin::Generated(generated.clone()),
                kind: scene_kind(region.kind),
                speaker,
            },
        )?;

        let target = region.continues.as_deref().and_then(|id| {
            texts
                .iter()
                .enumerate()
                .find(|(index, _)| region_id(*index) == id)
                .map(|(_, text)| text.content)
                .or_else(|| {
                    context
                        .previous
                        .iter()
                        .find(|(_, previous)| previous.id == id)
                        .map(|(content, _)| *content)
                })
        });
        match scene.relation_from::<Continues>(text.content)? {
            Some(relation) if matches!(relation.value().origin, Origin::User) => {}
            Some(relation) if Some(relation.value().target) == target => {}
            existing => {
                if let Some(relation) = existing {
                    edit.remove_relation(relation.id())?;
                }
                if let Some(target) = target {
                    edit.relate::<Continues>(text.content, target)?;
                }
            }
        }
    }
    Ok(())
}

/// The page's texts with a located source region, in reading order. Their
/// position in this list is the number drawn on the page.
fn page_texts(
    scene: &Snapshot,
    page: EntityId,
    source_language: Language,
) -> Result<Vec<PageText>> {
    let Some(group) = scene.page(page)?.text_group()? else {
        return Ok(Vec::new());
    };
    let mut seen = BTreeSet::new();
    let mut texts = Vec::new();
    for layer in group.text_layers()? {
        let content = layer.content()?;
        if !seen.insert(content.id()) {
            continue;
        }
        let Some(source) = content.source()? else {
            continue;
        };
        if source.text.value.trim().is_empty() {
            continue;
        }
        let Some(region) = content.source_region()? else {
            continue;
        };
        let Some(bounds) = crate::scope::geometry_extents(&region.geometry()?) else {
            continue;
        };
        texts.push(PageText {
            content: content.id(),
            source: source.text.value,
            bounds,
        });
    }
    let layout = layout_containers(scene, page)?
        .into_iter()
        .map(|(_, item)| item)
        .collect();
    let order = reading_permutation(
        &texts
            .iter()
            .map(|text| Some(text.bounds))
            .collect::<Vec<_>>(),
        layout,
        source_language,
    );
    let mut texts = texts.into_iter().map(Some).collect::<Vec<_>>();
    Ok(order
        .into_iter()
        .map(|index| texts[index].take().expect("reading order is a permutation"))
        .collect())
}

/// The last marked texts of the page before this one in project order.
fn previous_markup(
    scene: &Snapshot,
    page: EntityId,
    source_language: Language,
    list: Option<&CharacterList>,
) -> Result<Vec<(EntityId, PreviousMarkup)>> {
    let pages = scene.pages().map(|page| page.id()).collect::<Vec<_>>();
    let Some(previous) = pages
        .iter()
        .position(|candidate| *candidate == page)
        .and_then(|index| index.checked_sub(1))
        .map(|index| pages[index])
    else {
        return Ok(Vec::new());
    };
    let mut marked = Vec::new();
    for text in page_texts(scene, previous, source_language)? {
        if let Some(markup) = scene.component::<SpeechMarkup>(text.content)? {
            marked.push((text, markup));
        }
    }
    let skip = marked.len().saturating_sub(PREVIOUS_CONTEXT);
    Ok(marked
        .into_iter()
        .skip(skip)
        .enumerate()
        .map(|(index, (text, markup))| {
            (
                text.content,
                PreviousMarkup {
                    id: format!("p{}", index + 1),
                    text: text.source,
                    kind: translator_kind(markup.kind),
                    speaker: markup
                        .speaker
                        .filter(|speaker| {
                            list.is_some_and(|list| list.character(speaker).is_some())
                        })
                        .map(|speaker| speaker.to_string()),
                },
            )
        })
        .collect())
}

fn observed(texts: &[PageText]) -> Vec<ObservedText> {
    texts
        .iter()
        .map(|text| ObservedText {
            content: text.content,
            source: text.source.clone(),
        })
        .collect()
}

fn region_id(index: usize) -> String {
    (index + 1).to_string()
}

fn normalized(
    (left, top, right, bottom): (f64, f64, f64, f64),
    width: f64,
    height: f64,
) -> [u32; 4] {
    let scale = |value: f64, size: f64| (value / size * 1000.0).round().clamp(0.0, 1000.0) as u32;
    [
        scale(left, width),
        scale(top, height),
        scale(right, width),
        scale(bottom, height),
    ]
}

const BOX_COLOR: Rgb<u8> = Rgb([230, 0, 40]);
const LABEL_BACKGROUND: Rgb<u8> = Rgb([255, 255, 255]);

/// 3x5 digit glyphs, one row per byte with the low three bits lit left to right.
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

/// Draws each region's box and its 1-based page number on a copy of the page.
fn draw_regions(image: &DynamicImage, boxes: &[(usize, Bounds)], scale: (f64, f64)) -> RgbImage {
    let mut canvas = image.to_rgb8();
    let (width, height) = canvas.dimensions();
    let shortest = width.min(height).max(1);
    let thickness = (shortest / 400).max(2);
    let cell = (shortest / 250).max(2);
    for (number, (left, top, right, bottom)) in boxes.iter().copied() {
        let x0 = (left * scale.0).round().clamp(0.0, f64::from(width - 1)) as i32;
        let y0 = (top * scale.1).round().clamp(0.0, f64::from(height - 1)) as i32;
        let x1 = (right * scale.0).round().clamp(0.0, f64::from(width - 1)) as i32;
        let y1 = (bottom * scale.1).round().clamp(0.0, f64::from(height - 1)) as i32;
        for inset in 0..thickness as i32 {
            let box_width = (x1 - x0 - 2 * inset).max(1) as u32;
            let box_height = (y1 - y0 - 2 * inset).max(1) as u32;
            draw_hollow_rect_mut(
                &mut canvas,
                Rect::at(x0 + inset, y0 + inset).of_size(box_width, box_height),
                BOX_COLOR,
            );
        }

        let digits = number
            .to_string()
            .bytes()
            .map(|digit| DIGITS[usize::from(digit - b'0')])
            .collect::<Vec<_>>();
        let label_width = (digits.len() as u32 * 4 + 1) * cell;
        let label_height = 7 * cell;
        let label_x = x0.min(width.saturating_sub(label_width) as i32).max(0);
        let label_y = y0.min(height.saturating_sub(label_height) as i32).max(0);
        draw_filled_rect_mut(
            &mut canvas,
            Rect::at(label_x, label_y).of_size(label_width, label_height),
            LABEL_BACKGROUND,
        );
        for (position, glyph) in digits.iter().enumerate() {
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..3 {
                    if bits & (0b100 >> column) == 0 {
                        continue;
                    }
                    draw_filled_rect_mut(
                        &mut canvas,
                        Rect::at(
                            label_x + ((position as u32 * 4 + 1 + column) * cell) as i32,
                            label_y + ((row as u32 + 1) * cell) as i32,
                        )
                        .of_size(cell, cell),
                        BOX_COLOR,
                    );
                }
            }
        }
    }
    canvas
}

pub(super) fn translator_kind(kind: SpeechKind) -> MarkupKind {
    match kind {
        SpeechKind::Speech => MarkupKind::Speech,
        SpeechKind::Thought => MarkupKind::Thought,
        SpeechKind::Narration => MarkupKind::Narration,
        SpeechKind::SoundEffect => MarkupKind::SoundEffect,
        SpeechKind::Sign => MarkupKind::Sign,
        SpeechKind::Aside => MarkupKind::Aside,
    }
}

fn scene_kind(kind: MarkupKind) -> SpeechKind {
    match kind {
        MarkupKind::Speech => SpeechKind::Speech,
        MarkupKind::Thought => SpeechKind::Thought,
        MarkupKind::Narration => SpeechKind::Narration,
        MarkupKind::SoundEffect => SpeechKind::SoundEffect,
        MarkupKind::Sign => SpeechKind::Sign,
        MarkupKind::Aside => SpeechKind::Aside,
    }
}

pub(super) fn translator_gender(gender: Gender) -> CharacterGender {
    match gender {
        Gender::Male => CharacterGender::Male,
        Gender::Female => CharacterGender::Female,
        Gender::Unknown => CharacterGender::Unknown,
    }
}

fn scene_gender(gender: CharacterGender) -> Gender {
    match gender {
        CharacterGender::Male => Gender::Male,
        CharacterGender::Female => Gender::Female,
        CharacterGender::Unknown => Gender::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use koharu_scene::{
        At, Authored, Geometry, LanguageTag, PageDraft, RecognizedFrom, SourceText, TextLayout,
        TextLayoutKind, TextRegion,
    };
    use koharu_translator::MarkedRegion;

    use super::*;
    use crate::ImageCache;

    async fn page_with_texts(sources: &[&str]) -> (koharu_scene::Session, EntityId, Vec<EntityId>) {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let page = edit
            .add_page(PageDraft::new("page", 100.0, 200.0), At::End)
            .unwrap();
        let mut contents = Vec::new();
        for (index, source) in sources.iter().enumerate() {
            let region = edit
                .add_analysis_region::<TextRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0 + 50.0 * index as f64, 20.0, 30.0),
                    None,
                )
                .unwrap();
            let content = edit.add_text_content(page, At::End).unwrap();
            edit.add_text_layer(
                page,
                At::End,
                content,
                &TextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )
            .unwrap();
            edit.set(
                content,
                &SourceText {
                    text: Authored::user((*source).to_owned()),
                    language: Some(LanguageTag::new("ja").unwrap()),
                },
            )
            .unwrap();
            edit.relate::<RecognizedFrom>(content, region).unwrap();
            contents.push(content);
        }
        session.commit(edit.finish().unwrap()).await.unwrap();
        (session, page, contents)
    }

    fn generated() -> Generation {
        super::super::generation(PRODUCER, "test-markup").unwrap()
    }

    async fn commit(
        session: &mut koharu_scene::Session,
        page: EntityId,
        outcome: Result<MarkupOutcome>,
    ) {
        let snapshot = session.snapshot();
        let context = PageContext::read(&snapshot, page, Language::English).unwrap();
        let patch = page_patch(&snapshot, &context, &generated(), outcome).unwrap();
        session.commit(patch).await.unwrap();
    }

    fn skips(session: &koharu_scene::Session, page: EntityId) -> bool {
        let input = StageInput::new(
            session.snapshot(),
            page,
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
        );
        skip(&input, Language::English).unwrap()
    }

    fn region(
        id: &str,
        kind: MarkupKind,
        speaker: Option<&str>,
        continues: Option<&str>,
    ) -> MarkedRegion {
        MarkedRegion {
            id: id.to_owned(),
            kind,
            speaker: speaker.map(str::to_owned),
            continues: continues.map(str::to_owned),
            confidence: 0.9,
        }
    }

    fn character(id: &str, name: &str, gender: CharacterGender) -> MarkupCharacter {
        MarkupCharacter {
            id: id.to_owned(),
            name: name.to_owned(),
            gender,
            notes: String::new(),
        }
    }

    #[tokio::test]
    async fn withheld_page_records_why_and_is_retried_until_complete() {
        let (mut session, page, contents) = page_with_texts(&["一", "二"]).await;
        assert!(!skips(&session, page));

        commit(
            &mut session,
            page,
            Ok(MarkupOutcome::Withheld(MarkupWithheld::Refused)),
        )
        .await;
        let markup = session
            .snapshot()
            .page(page)
            .unwrap()
            .markup()
            .unwrap()
            .unwrap();
        assert_eq!(
            markup.status,
            PageMarkupStatus::Missing(PageMarkupMissing::Refused)
        );
        assert_eq!(
            markup
                .observed
                .iter()
                .map(|text| text.content)
                .collect::<Vec<_>>(),
            contents
        );
        assert!(!skips(&session, page));

        commit(&mut session, page, Err(anyhow::anyhow!("network down"))).await;
        let markup = session
            .snapshot()
            .page(page)
            .unwrap()
            .markup()
            .unwrap()
            .unwrap();
        assert_eq!(
            markup.status,
            PageMarkupStatus::Missing(PageMarkupMissing::Failed)
        );

        commit(
            &mut session,
            page,
            Err(koharu_translator::Error::InvalidMarkup {
                provider: "test",
                message: "region 1 appears twice".to_owned(),
            }
            .into()),
        )
        .await;
        let markup = session
            .snapshot()
            .page(page)
            .unwrap()
            .markup()
            .unwrap()
            .unwrap();
        assert_eq!(
            markup.status,
            PageMarkupStatus::Missing(PageMarkupMissing::Invalid)
        );

        commit(
            &mut session,
            page,
            Ok(MarkupOutcome::Marked {
                regions: vec![
                    region("1", MarkupKind::Narration, None, None),
                    region("2", MarkupKind::Narration, None, Some("1")),
                ],
                new_characters: Vec::new(),
                unboxed: Vec::new(),
            }),
        )
        .await;
        assert!(skips(&session, page));
        let snapshot = session.snapshot();
        let second = snapshot.text_content(contents[1]).unwrap();
        assert_eq!(
            second.continues().unwrap().map(|content| content.id()),
            Some(contents[0])
        );

        let mut edit = session.snapshot().edit();
        edit.set(
            contents[1],
            &SourceText {
                text: Authored::user("三".to_owned()),
                language: None,
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        assert!(!skips(&session, page));
    }

    #[tokio::test]
    async fn user_markup_and_a_user_cast_survive_a_rerun() {
        let (mut session, page, contents) = page_with_texts(&["僕は", "行く", "よ"]).await;
        let mut edit = session.snapshot().edit();
        edit.set_project(&CharacterList {
            origin: Origin::User,
            characters: vec![Character {
                id: CharacterId::new("hero").unwrap(),
                name: "Hero".to_owned(),
                gender: Gender::Male,
                notes: String::new(),
            }],
        })
        .unwrap();
        let user_markup = SpeechMarkup {
            origin: Origin::User,
            kind: SpeechKind::Speech,
            speaker: Some(CharacterId::new("hero").unwrap()),
        };
        edit.set(contents[0], &user_markup).unwrap();
        edit.relate::<Continues>(contents[2], contents[0]).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        commit(
            &mut session,
            page,
            Ok(MarkupOutcome::Marked {
                regions: vec![
                    region("1", MarkupKind::Narration, None, None),
                    region("2", MarkupKind::Speech, Some("yui"), Some("1")),
                    region("3", MarkupKind::Thought, Some("hero"), None),
                ],
                new_characters: vec![character("yui", "Yui", CharacterGender::Female)],
                unboxed: Vec::new(),
            }),
        )
        .await;

        let snapshot = session.snapshot();
        assert_eq!(
            snapshot.component::<SpeechMarkup>(contents[0]).unwrap(),
            Some(user_markup)
        );
        let second = snapshot
            .component::<SpeechMarkup>(contents[1])
            .unwrap()
            .unwrap();
        assert_eq!(second.kind, SpeechKind::Speech);
        assert_eq!(second.speaker, None);
        assert!(matches!(second.origin, Origin::Generated(_)));
        let third = snapshot
            .component::<SpeechMarkup>(contents[2])
            .unwrap()
            .unwrap();
        assert_eq!(third.speaker, Some(CharacterId::new("hero").unwrap()));
        let user_continues = snapshot
            .relation_from::<Continues>(contents[2])
            .unwrap()
            .unwrap();
        assert!(matches!(user_continues.value().origin, Origin::User));
        assert_eq!(user_continues.value().target, contents[0]);
        let list = snapshot
            .project_component::<CharacterList>()
            .unwrap()
            .unwrap();
        assert!(matches!(list.origin, Origin::User));
        assert_eq!(list.characters.len(), 1);
    }

    #[tokio::test]
    async fn markup_grows_a_generated_cast_and_links_the_previous_page() {
        let (mut session, first_page, first) = page_with_texts(&["昨日"]).await;
        commit(
            &mut session,
            first_page,
            Ok(MarkupOutcome::Marked {
                regions: vec![region("1", MarkupKind::Narration, Some("narrator"), None)],
                new_characters: vec![character("narrator", "Narrator", CharacterGender::Male)],
                unboxed: Vec::new(),
            }),
        )
        .await;
        let mut edit = session.snapshot().edit();
        let second_page = edit
            .add_page(PageDraft::new("next", 100.0, 200.0), At::End)
            .unwrap();
        let text_region = edit
            .add_analysis_region::<TextRegion>(
                second_page,
                At::End,
                &Geometry::rectangle(10.0, 10.0, 20.0, 30.0),
                None,
            )
            .unwrap();
        let content = edit.add_text_content(second_page, At::End).unwrap();
        edit.add_text_layer(
            second_page,
            At::End,
            content,
            &TextLayout {
                origin: Origin::User,
                kind: TextLayoutKind::Paragraph,
                angle_degrees: None,
            },
        )
        .unwrap();
        edit.set(
            content,
            &SourceText {
                text: Authored::user("行った".to_owned()),
                language: None,
            },
        )
        .unwrap();
        edit.relate::<RecognizedFrom>(content, text_region).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let snapshot = session.snapshot();
        let context = PageContext::read(&snapshot, second_page, Language::English).unwrap();
        assert_eq!(context.previous.len(), 1);
        assert_eq!(context.previous[0].1.id, "p1");
        assert_eq!(context.previous[0].1.speaker.as_deref(), Some("narrator"));
        let patch = page_patch(
            &snapshot,
            &context,
            &generated(),
            Ok(MarkupOutcome::Marked {
                regions: vec![region("1", MarkupKind::Narration, Some("yui"), Some("p1"))],
                new_characters: vec![character("yui", "Yui", CharacterGender::Female)],
                unboxed: Vec::new(),
            }),
        )
        .unwrap();
        session.commit(patch).await.unwrap();

        let snapshot = session.snapshot();
        let list = snapshot
            .project_component::<CharacterList>()
            .unwrap()
            .unwrap();
        assert_eq!(
            list.characters
                .iter()
                .map(|character| (character.id.as_str(), character.gender))
                .collect::<Vec<_>>(),
            [("narrator", Gender::Male), ("yui", Gender::Female)]
        );
        assert_eq!(
            snapshot
                .text_content(content)
                .unwrap()
                .continues()
                .unwrap()
                .map(|content| content.id()),
            Some(first[0])
        );
    }

    #[test]
    fn the_second_half_sees_the_first_half_and_the_halves_join_in_page_order() {
        let text = |source: &str| PageText {
            content: EntityId::new(),
            source: source.to_owned(),
            bounds: (0.0, 0.0, 10.0, 10.0),
        };
        let texts = [text("一"), text("二"), text("三"), text("四")];
        let first = vec![
            region("1", MarkupKind::Speech, Some("hero"), None),
            region("2", MarkupKind::SoundEffect, None, None),
        ];
        let earlier = earlier_markup(&texts[..2], &first);
        assert_eq!(earlier[0].id, "1");
        assert_eq!(earlier[0].text, "一");
        assert_eq!(earlier[0].speaker.as_deref(), Some("hero"));
        assert_eq!(earlier[1].kind, MarkupKind::SoundEffect);

        let hero = character("hero", "Hero", CharacterGender::Male);
        let joined = join_halves(
            first.clone(),
            vec![hero.clone()],
            Vec::new(),
            MarkupOutcome::Marked {
                regions: vec![
                    region("3", MarkupKind::Speech, Some("hero"), Some("1")),
                    region("4", MarkupKind::Thought, None, None),
                ],
                new_characters: Vec::new(),
                unboxed: Vec::new(),
            },
        );
        let MarkupOutcome::Marked {
            regions,
            new_characters,
            ..
        } = joined
        else {
            panic!("expected joined markup");
        };
        assert_eq!(
            regions
                .iter()
                .map(|region| region.id.as_str())
                .collect::<Vec<_>>(),
            ["1", "2", "3", "4"]
        );
        assert_eq!(new_characters, [hero]);
        assert_eq!(
            join_halves(
                first,
                Vec::new(),
                Vec::new(),
                MarkupOutcome::Withheld(MarkupWithheld::ContentFilter)
            ),
            MarkupOutcome::Withheld(MarkupWithheld::ContentFilter)
        );
    }

    #[tokio::test]
    async fn unboxed_text_is_flagged_until_covered_or_dismissed() {
        let (mut session, page, _) = page_with_texts(&["一"]).await;
        let unboxed = |bbox| koharu_translator::UnboxedText {
            bbox,
            kind: MarkupKind::SoundEffect,
        };
        let marked = || {
            Ok(MarkupOutcome::Marked {
                regions: vec![region("1", MarkupKind::Speech, None, None)],
                new_characters: Vec::new(),
                unboxed: vec![
                    // Inside the existing text region.
                    unboxed([100, 50, 300, 200]),
                    unboxed([500, 500, 700, 600]),
                    // The same lettering again.
                    unboxed([510, 505, 700, 600]),
                    unboxed([100, 800, 300, 900]),
                ],
            })
        };
        commit(&mut session, page, marked()).await;
        let open = open_unboxed_text(&session.snapshot(), page).unwrap();
        let flag = UnboxedText {
            x: 50.0,
            y: 100.0,
            width: 20.0,
            height: 20.0,
            kind: SpeechKind::SoundEffect,
        };
        assert_eq!(open.len(), 2);
        assert_eq!(open[0], flag);

        // A dismissal is the user's own record; markup can still rerun.
        let mut edit = session.snapshot().edit();
        edit.set(
            page,
            &UnboxedTextReview {
                origin: Origin::User,
                dismissed: vec![open[1]],
            },
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        commit(&mut session, page, marked()).await;
        assert_eq!(
            open_unboxed_text(&session.snapshot(), page).unwrap(),
            [flag]
        );

        let mut edit = session.snapshot().edit();
        edit.add_analysis_region::<TextRegion>(
            page,
            At::End,
            &Geometry::rectangle(48.0, 98.0, 24.0, 24.0),
            None,
        )
        .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();
        assert!(
            open_unboxed_text(&session.snapshot(), page)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn numbered_boxes_are_drawn_inside_the_page() {
        let image = DynamicImage::new_rgb8(400, 800);
        let drawn = draw_regions(
            &image,
            &[
                (1, (10.0, 10.0, 60.0, 90.0)),
                (2, (190.0, 380.0, 200.0, 400.0)),
            ],
            (2.0, 2.0),
        );
        assert_eq!(drawn.dimensions(), (400, 800));
        assert_eq!(*drawn.get_pixel(20, 100), BOX_COLOR);
        assert_eq!(*drawn.get_pixel(380, 790), BOX_COLOR);
        assert_eq!(*drawn.get_pixel(200, 400), Rgb([0, 0, 0]));
        assert_eq!(
            normalized((10.0, 50.0, 100.0, 200.0), 100.0, 200.0),
            [100, 250, 1000, 1000]
        );
    }
}
