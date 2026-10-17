//! Text layout and portable glyph recording.

use std::sync::{Arc, LazyLock};

use anyhow::Result;
use fontique::Attributes;
use koharu_rasterizer::{
    PreparedGlyph, PreparedGlyphRun, PreparedResource, PreparedScene, PreparedSceneCommand,
    ResourceId,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use vello::kurbo::Affine;

use crate::{
    Error, FontStyle, HyphenationPolicy, LayoutRun, RenderBounds, RenderDiagnostic,
    Result as RenderResult, TextAlign, TextLayout, TypesettingConfig, WritingMode,
    bubble::{
        LayoutBox, contour, detect_contour_lobes, geometry_frame, polygon_area_px,
    },
    fonts::{Font, FontSystem, Fonts, font_key},
    script::{fontique_scripts, is_chinese_or_japanese_text},
    segment::contains_cyrillic,
};

static COPYFIT_FONT_SYSTEM: LazyLock<Mutex<FontSystem>> =
    LazyLock::new(|| Mutex::new(FontSystem::new()));

/// Configuration for dry-run copyfitting measurement (`measure_text_fit`).
#[derive(Clone, Debug, PartialEq)]
pub struct CopyfitFontConfig {
    pub font_families: Vec<String>,
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_style: Option<FontStyle>,
    pub language: Option<String>,
    pub minimum_font_size: f32,
    pub target_font_size: f32,
    pub line_height: f32,
    pub text_inset: [f32; 4],
    pub writing_mode: WritingMode,
    /// Safe white-margin relaxation in pixels (`0.0..=5.0`) available inside the bubble border.
    pub safe_margin_relax_px: f32,
}

impl Default for CopyfitFontConfig {
    fn default() -> Self {
        Self {
            font_families: TypesettingConfig::default().font_families,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            language: None,
            minimum_font_size: 9.0,
            target_font_size: 13.0,
            line_height: 1.2,
            text_inset: [4.0; 4],
            writing_mode: WritingMode::Horizontal,
            safe_margin_relax_px: 0.0,
        }
    }
}

/// Geometric and typographic diagnostics produced by `measure_text_fit` before rasterization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopyfitDiagnostics {
    pub resulting_font_size: f32,
    pub target_font_size: f32,
    pub overflow: bool,
    pub line_count: usize,
    pub lines: Vec<String>,
    pub max_word_char_len: usize,
    pub has_ugly_single_char_wrap: bool,
    pub area_utilization: f32,
    pub relaxed_inset: [f32; 4],
    pub lobe_count: usize,
    pub recommended_max_chars: usize,
    pub recommended_max_word_len: usize,
}

/// Fast dry-run measurement of how `text` fits inside `bubble_polygon` without GPU rasterization.
///
/// Uses contour-aware diamond/barrel horizontal cross-sections, syllabic hyphenation for narrow
/// bubbles, two-lobe neck detection, and automatic contour padding relaxation when
/// `resulting_font_size < 12.0` and `safe_margin_relax_px >= 2.0`.
pub fn measure_text_fit(
    text: &str,
    bubble_polygon: &koharu_scene::Geometry,
    font_config: &CopyfitFontConfig,
) -> RenderResult<CopyfitDiagnostics> {
    let trimmed = text.trim();
    let Some(frame) = geometry_frame(bubble_polygon) else {
        return Err(Error::invalid("invalid bubble polygon geometry for copyfit"));
    };
    let raw_contour = contour(bubble_polygon, frame);
    let lobes = detect_contour_lobes(&raw_contour);
    let lobe_count = lobes.len().max(1);
    let usable_area = polygon_area_px(&raw_contour)
        .max(frame.bounds.width * frame.bounds.height * 0.72)
        .max(1.0);
    let target_font_size = (usable_area.sqrt() * 0.14)
        .clamp(12.5, 14.0)
        .max(font_config.target_font_size);

    let max_word_char_len = max_alphabetic_word_len(trimmed);
    let effective_inset = font_config.text_inset;
    let max_inset = effective_inset.into_iter().fold(0.0f32, f32::max);
    let usable_width = (frame.bounds.width - max_inset * 2.0).max(16.0);
    let usable_height = (frame.bounds.height - max_inset * 2.0).max(16.0);
    let avg_char_px = (target_font_size * 0.56).max(6.5);
    let recommended_max_word_len = ((usable_width * 0.88) / avg_char_px)
        .floor()
        .clamp(5.0, 18.0) as usize;
    let max_target_lines = (usable_height / (target_font_size * font_config.line_height))
        .floor()
        .max(1.0);
    let recommended_max_chars = ((usable_width * 0.76 / avg_char_px) * max_target_lines)
        .floor()
        .clamp(8.0, 220.0) as usize;

    if trimmed.is_empty() {
        return Ok(CopyfitDiagnostics {
            resulting_font_size: target_font_size,
            target_font_size,
            overflow: false,
            line_count: 0,
            lines: Vec::new(),
            max_word_char_len: 0,
            has_ugly_single_char_wrap: false,
            area_utilization: 0.0,
            relaxed_inset: effective_inset,
            lobe_count,
            recommended_max_chars,
            recommended_max_word_len,
        });
    }

    let resolved_fonts = resolve_copyfit_fonts(trimmed, font_config);
    let mut best_diag = measure_with_inset(
        trimmed,
        frame.bounds,
        &raw_contour,
        font_config,
        effective_inset,
        target_font_size,
        usable_area,
        max_word_char_len,
        lobe_count,
        recommended_max_chars,
        recommended_max_word_len,
        resolved_fonts.as_deref(),
    );

    // Contour Padding Relaxation: if font size < 12.0pt or overflowed, and there is a safe
    // white margin inside the bubble contour (or default 4px inset can be relaxed by 2..5px),
    // try relaxing the inset without touching the dark frame.
    let relax_px = if font_config.safe_margin_relax_px >= 2.0 {
        font_config.safe_margin_relax_px.clamp(2.0, 5.0)
    } else if max_inset >= 4.0 {
        2.5
    } else {
        0.0
    };
    if (best_diag.resulting_font_size < 12.0 || best_diag.overflow) && relax_px >= 2.0 {
        let relaxed_inset = effective_inset.map(|v| (v - relax_px).max(1.0));
        let candidate = measure_with_inset(
            trimmed,
            frame.bounds,
            &raw_contour,
            font_config,
            relaxed_inset,
            target_font_size,
            usable_area,
            max_word_char_len,
            lobe_count,
            recommended_max_chars,
            recommended_max_word_len,
            resolved_fonts.as_deref(),
        );
        if (!candidate.overflow && best_diag.overflow)
            || (candidate.overflow == best_diag.overflow
                && candidate.resulting_font_size > best_diag.resulting_font_size + 0.25)
        {
            best_diag = candidate;
        }
    }

    Ok(best_diag)
}

#[allow(clippy::too_many_arguments)]
fn measure_with_inset(
    text: &str,
    frame_bounds: LayoutBox,
    raw_contour: &[(f32, f32)],
    font_config: &CopyfitFontConfig,
    text_inset: [f32; 4],
    target_font_size: f32,
    usable_area: f32,
    max_word_char_len: usize,
    lobe_count: usize,
    recommended_max_chars: usize,
    recommended_max_word_len: usize,
    fonts: Option<&[Font]>,
) -> CopyfitDiagnostics {
    let bounds = inset(frame_bounds, text_inset);
    if bounds.width <= 1.0 || bounds.height <= 1.0 {
        return CopyfitDiagnostics {
            resulting_font_size: font_config.minimum_font_size,
            target_font_size,
            overflow: true,
            line_count: 1,
            lines: vec![text.to_owned()],
            max_word_char_len,
            has_ugly_single_char_wrap: false,
            area_utilization: 1.0,
            relaxed_inset: text_inset,
            lobe_count,
            recommended_max_chars,
            recommended_max_word_len,
        };
    }

    let [top, _, _, left] = text_inset;
    let shifted_contour: Vec<(f32, f32)> = raw_contour
        .iter()
        .map(|&(x, y)| (x - left, y - top))
        .collect();
    let min_air = text_inset.into_iter().fold(0.0f32, f32::max);

    if let Some(fonts) = fonts.filter(|f| !f.is_empty()) {
        let maximum = if font_config.writing_mode.is_vertical() {
            bounds.height
        } else {
            bounds.width
        }
        .max(font_config.minimum_font_size);
        let minimum = font_config.minimum_font_size.min(maximum);
        let mut layout = TextLayout::new(&fonts[0])
            .with_fallback_fonts(&fonts[1..])
            .with_writing_mode(font_config.writing_mode)
            .with_alignment(TextAlign::Center)
            .with_line_height(font_config.line_height)
            .with_max_width(bounds.width)
            .with_max_height(bounds.height)
            .with_comic_balloon(bounds.width, bounds.height, shifted_contour.clone(), min_air)
            .with_max_font_size(maximum)
            .with_min_font_size(minimum);

        if let Some(lang) = &font_config.language {
            layout = layout.with_hyphenation_language_tag(lang);
        } else if contains_cyrillic(text) {
            layout = layout.with_hyphenation_language_tag("ru");
        }
        if font_config.writing_mode == WritingMode::Horizontal {
            layout = layout.with_hyphenation_policy(HyphenationPolicy::LastResort);
        }

        if let Ok(run) = layout.run(text) {
            let lines: Vec<String> = run
                .lines
                .iter()
                .map(|line| text.get(line.range.clone()).unwrap_or("").trim().to_owned())
                .filter(|line| !line.is_empty())
                .collect();
            let has_ugly = detect_ugly_line_wraps(text, &lines);
            let text_area = (run.width * run.height).max(0.0);
            return CopyfitDiagnostics {
                resulting_font_size: run.font_size,
                target_font_size,
                overflow: run.overflowed(),
                line_count: lines.len(),
                lines,
                max_word_char_len,
                has_ugly_single_char_wrap: has_ugly,
                area_utilization: (text_area / usable_area).clamp(0.0, 2.0),
                relaxed_inset: text_inset,
                lobe_count,
                recommended_max_chars,
                recommended_max_word_len,
            };
        }
    }

    measure_analytical_balloon_fit(
        text,
        bounds,
        &shifted_contour,
        font_config,
        text_inset,
        target_font_size,
        usable_area,
        max_word_char_len,
        lobe_count,
        recommended_max_chars,
        recommended_max_word_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_analytical_balloon_fit(
    text: &str,
    bounds: LayoutBox,
    contour: &[(f32, f32)],
    font_config: &CopyfitFontConfig,
    text_inset: [f32; 4],
    target_font_size: f32,
    usable_area: f32,
    max_word_char_len: usize,
    lobe_count: usize,
    recommended_max_chars: usize,
    recommended_max_word_len: usize,
) -> CopyfitDiagnostics {
    let words: Vec<&str> = text.split_whitespace().collect();
    let max_size = bounds.width.min(bounds.height).clamp(9.0, 48.0);
    let min_size = font_config.minimum_font_size;
    let mut chosen_size = min_size;
    let mut chosen_lines: Vec<String> = Vec::new();
    let mut chosen_overflow = true;

    let span_width_at_y = |y: f32| -> f32 {
        if contour.len() < 3 {
            return bounds.width;
        }
        let mut xs = Vec::new();
        for idx in 0..contour.len() {
            let a = contour[idx];
            let b = contour[(idx + 1) % contour.len()];
            if (a.1 <= y && b.1 > y) || (b.1 <= y && a.1 > y) {
                let t = (y - a.1) / (b.1 - a.1);
                xs.push(a.0 + (b.0 - a.0) * t);
            }
        }
        xs.sort_by(f32::total_cmp);
        xs.as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[1] - pair[0]).max(0.0))
            .fold(0.0f32, f32::max)
            .max(bounds.width * 0.45)
    };

    let mut probe_size = max_size;
    while probe_size >= min_size - 0.01 {
        let size = probe_size.max(min_size);
        let char_w = size * 0.56;
        let line_h = size * font_config.line_height;
        let max_lines = ((bounds.height - size * 0.8) / line_h).floor().max(1.0) as usize;
        let mut lines = Vec::new();
        let mut current = String::new();
        let mut word_idx = 0usize;
        let mut overflow = false;

        while word_idx < words.len() {
            let line_idx = lines.len();
            let y_frac = (line_idx as f32 + 0.5) / (max_lines.max(2) as f32);
            let y = (bounds.height * y_frac).clamp(size * 0.5, (bounds.height - size * 0.5).max(size * 0.5));
            let avail_w = (span_width_at_y(y) - size * 1.1).max(size * 1.5);
            let max_chars_line = (avail_w / char_w).floor().max(1.0) as usize;

            let word = words[word_idx];
            let candidate_len = if current.is_empty() {
                word.chars().count()
            } else {
                current.chars().count() + 1 + word.chars().count()
            };
            if candidate_len <= max_chars_line {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
                word_idx += 1;
            } else if current.is_empty() {
                if bounds.width < 95.0 && word.chars().count() >= 9 && max_chars_line >= 4 {
                    let split_at = (max_chars_line.saturating_sub(1)).clamp(3, word.chars().count() - 3);
                    let prefix: String = word.chars().take(split_at).collect();
                    let suffix: String = word.chars().skip(split_at).collect();
                    lines.push(format!("{prefix}-"));
                    current = suffix;
                    word_idx += 1;
                } else {
                    overflow = true;
                    lines.push(word.to_owned());
                    word_idx += 1;
                }
            } else {
                lines.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
        if lines.len() > max_lines {
            overflow = true;
        }
        chosen_size = size;
        chosen_lines = lines;
        chosen_overflow = overflow;
        if !overflow {
            break;
        }
        if (probe_size - min_size).abs() < 0.05 {
            break;
        }
        probe_size = (probe_size - 0.75).max(min_size);
    }

    let max_line_chars = chosen_lines
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1) as f32;
    let est_w = (max_line_chars * chosen_size * 0.56).min(bounds.width);
    let est_h = (chosen_lines.len() as f32 * chosen_size * font_config.line_height).min(bounds.height);
    let has_ugly = detect_ugly_line_wraps(text, &chosen_lines);

    CopyfitDiagnostics {
        resulting_font_size: chosen_size,
        target_font_size,
        overflow: chosen_overflow,
        line_count: chosen_lines.len(),
        lines: chosen_lines,
        max_word_char_len,
        has_ugly_single_char_wrap: has_ugly,
        area_utilization: ((est_w * est_h) / usable_area).clamp(0.0, 2.0),
        relaxed_inset: text_inset,
        lobe_count,
        recommended_max_chars,
        recommended_max_word_len,
    }
}

fn resolve_copyfit_fonts(text: &str, config: &CopyfitFontConfig) -> Option<Vec<Font>> {
    let mut families = Vec::new();
    if let Some(preferred) = &config.preferred_font
        && !preferred.trim().is_empty()
    {
        families.push(preferred.clone());
    }
    for family in &config.font_families {
        if !families.iter().any(|existing| existing.eq_ignore_ascii_case(family)) {
            families.push(family.clone());
        }
    }
    let scripts = fontique_scripts(text);
    let mut system = COPYFIT_FONT_SYSTEM.lock();
    system
        .resolve(
            &families,
            Attributes::default(),
            &scripts,
            config.language.as_deref(),
        )
        .ok()
}

fn max_alphabetic_word_len(text: &str) -> usize {
    text.split(|ch: char| !ch.is_alphabetic())
        .map(|word| word.chars().count())
        .max()
        .unwrap_or(0)
}

/// Detects ugly single-character line wraps, hanging 1–2 letter Russian/English prepositions at line
/// ends, or mid-word breaks without a hyphen.
#[must_use]
pub fn detect_ugly_line_wraps(full_text: &str, lines: &[String]) -> bool {
    if lines.len() < 2 {
        return false;
    }
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        let word_count = trimmed.split_whitespace().count();

        // Single-letter line in a multi-line layout (orphan character/pronoun).
        if alpha_count == 1 && word_count == 1 && lines.len() >= 2 {
            return true;
        }

        // Check hanging short preposition/conjunction at the end of a non-final line.
        if idx + 1 < lines.len() {
            let last_token = trimmed
                .rsplit(|c: char| !c.is_alphabetic())
                .find(|w| !w.is_empty())
                .unwrap_or("")
                .to_lowercase();
            if word_count >= 2
                && matches!(
                    last_token.as_str(),
                    "в" | "к"
                        | "с"
                        | "у"
                        | "о"
                        | "и"
                        | "а"
                        | "я"
                        | "не"
                        | "ни"
                        | "на"
                        | "по"
                        | "за"
                        | "из"
                        | "от"
                        | "до"
                        | "об"
                        | "со"
                        | "ко"
                        | "во"
                        | "a"
                        | "an"
                        | "to"
                        | "in"
                        | "on"
                        | "at"
                        | "of"
                )
            {
                return true;
            }
        }
    }

    // Also check if any word in full_text was broken across two lines without a hyphen `-`.
    for pair in lines.windows(2) {
        let first = pair[0].trim_end();
        let second = pair[1].trim_start();
        if !first.ends_with('-')
            && let (Some(last_ch), Some(first_ch)) = (first.chars().next_back(), second.chars().next())
            && last_ch.is_alphabetic()
            && first_ch.is_alphabetic()
        {
            let joined_probe = format!(
                "{}{}",
                first.rsplit(|c: char| !c.is_alphabetic()).next().unwrap_or(""),
                second.split(|c: char| !c.is_alphabetic()).next().unwrap_or("")
            );
            if !joined_probe.is_empty() && full_text.contains(&joined_probe) {
                return true;
            }
        }
    }

    false
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextNodeDescriptor {
    pub(crate) entity: koharu_scene::EntityId,
    pub(crate) text: String,
    pub(crate) language: Option<koharu_scene::LanguageTag>,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) balloon_contour: Option<Vec<(f32, f32)>>,
    pub(crate) flow_contour: Option<Vec<(f32, f32)>>,
    pub(crate) preferred_font: Option<String>,
    pub(crate) font_families: Vec<String>,
    pub(crate) font_weight: Option<u16>,
    pub(crate) font_style: Option<FontStyle>,
    pub(crate) font_size: Option<f32>,
    pub(crate) minimum_font_size: f32,
    pub(crate) auto_fit: bool,
    pub(crate) alignment: TextAlign,
    pub(crate) writing_mode: WritingMode,
    pub(crate) foreground_color: [u8; 4],
    pub(crate) stroke: Option<StrokeOptions>,
    pub(crate) line_height: f32,
    pub(crate) letter_spacing: f32,
    pub(crate) word_spacing: f32,
    pub(crate) text_inset: [f32; 4],
    pub(crate) point_text: bool,
}

pub(crate) struct RenderedTextNode {
    pub(crate) scene: Arc<PreparedScene>,
    pub(crate) resources: Arc<[PreparedResource]>,
    pub(crate) local_bounds: RenderBounds,
    pub(crate) metadata: RenderedTextMetadata,
    pub(crate) diagnostics: Vec<RenderDiagnostic>,
}

pub(crate) struct RenderedTextMetadata {
    pub(crate) rendered_bounds: RenderBounds,
    pub(crate) layout_bounds: RenderBounds,
    pub(crate) post_script_fonts: Vec<String>,
    pub(crate) font_size: f32,
    pub(crate) color: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StrokeOptions {
    pub color: [u8; 4],
    pub width_px: f32,
}

/// Paint options used when recording one laid-out text run into a Vello scene.
#[derive(Debug, Clone)]
pub(crate) struct TextRenderOptions {
    pub color: [u8; 4],
    pub hint_glyphs: bool,
    pub padding: f32,
    pub baseline_shift: f32,
    pub stroke: Option<StrokeOptions>,
}

impl Default for TextRenderOptions {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 255],
            hint_glyphs: true,
            padding: 0.0,
            baseline_shift: 0.0,
            stroke: None,
        }
    }
}

/// Shapes text and records the resulting glyphs into vector scenes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TextRenderer;

impl TextRenderer {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub(crate) fn layout<'a>(&self, builder: &TextLayout<'a>, text: &str) -> Result<LayoutRun<'a>> {
        builder.run(text)
    }

    pub(crate) fn render(
        &self,
        scene: &mut PreparedScene,
        resources: &mut Vec<PreparedResource>,
        layout: &LayoutRun<'_>,
        writing_mode: WritingMode,
        options: &TextRenderOptions,
        transform: Affine,
    ) {
        // The border is drawn first as an outward-only dilation of the glyph outline (see
        // `draw_layout`), then the ordinary fill is drawn on top in the foreground color. Unlike a
        // centered stroke, a dilation never grows inward, so it can't punch a hole through small
        // features (e.g. dots) once the border width exceeds their radius.
        if let Some(stroke) = options
            .stroke
            .filter(|stroke| stroke.width_px > 0.0 && stroke.color[3] > 0)
        {
            draw_layout(
                scene,
                resources,
                layout,
                writing_mode,
                options,
                transform,
                GlyphPaint {
                    color: stroke.color,
                    dilation_px: stroke.width_px,
                },
            );
        }
        draw_layout(
            scene,
            resources,
            layout,
            writing_mode,
            options,
            transform,
            GlyphPaint {
                color: options.color,
                dilation_px: 0.0,
            },
        );
    }

    pub(crate) fn render_descriptor(
        &self,
        descriptor: &TextNodeDescriptor,
        fonts: &Fonts,
    ) -> RenderResult<RenderedTextNode> {
        let is_bubble_text = descriptor.balloon_contour.is_some();
        let frame = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: descriptor.width,
            height: descriptor.height,
        };
        let mut bounds = if is_bubble_text {
            inset(frame, descriptor.text_inset)
        } else {
            frame
        };
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return Err(Error::invalid(format!(
                "text inset leaves no layout area for entity {}",
                descriptor.entity
            )));
        }
        let fonts = fonts
            .resolve(
                descriptor.preferred_font.as_deref(),
                descriptor.font_weight,
                descriptor.font_style,
                &descriptor.font_families,
                &descriptor.text,
                descriptor
                    .language
                    .as_ref()
                    .map(koharu_scene::LanguageTag::as_str),
            )
            .map_err(|source| Error::Font {
                entity: descriptor.entity,
                source,
            })?;

        let build_layout_for_inset = |active_inset: [f32; 4], active_bounds: LayoutBox| {
            let automatic_maximum = || automatic_maximum(descriptor, active_bounds, is_bubble_text);
            let maximum = if descriptor.auto_fit {
                automatic_maximum()
            } else {
                descriptor.font_size.unwrap_or_else(automatic_maximum)
            };
            let minimum = descriptor.minimum_font_size.min(maximum);
            let mut layout = TextLayout::new(&fonts[0])
                .with_fallback_fonts(&fonts[1..])
                .with_writing_mode(descriptor.writing_mode)
                .with_alignment(descriptor.alignment)
                .with_line_height(descriptor.line_height)
                .with_spacing(descriptor.letter_spacing, descriptor.word_spacing)
                .with_cjk_punctuation_layout(
                    is_chinese_or_japanese_text(&descriptor.text)
                        || descriptor
                            .language
                            .as_ref()
                            .is_some_and(|language| is_chinese_or_japanese_language(language.as_str())),
                );
            if !descriptor.point_text {
                layout = layout
                    .with_max_width(active_bounds.width)
                    .with_max_height(active_bounds.height);
            }
            if let Some(contour) = &descriptor.balloon_contour {
                let [top, _, _, left] = active_inset;
                let contour = contour.iter().map(|&(x, y)| (x - left, y - top)).collect();
                if let Some(flow_contour) = &descriptor.flow_contour {
                    layout = layout.with_comic_balloon_constraints(
                        active_bounds.width,
                        active_bounds.height,
                        vec![
                            contour,
                            flow_contour
                                .iter()
                                .map(|&(x, y)| (x - left, y - top))
                                .collect(),
                        ],
                        active_inset.into_iter().fold(0.0, f32::max),
                    );
                } else {
                    layout = layout.with_comic_balloon(
                        active_bounds.width,
                        active_bounds.height,
                        contour,
                        active_inset.into_iter().fold(0.0, f32::max),
                    );
                }
            }
            if let Some(language) = &descriptor.language {
                layout = layout.with_hyphenation_language_tag(language.as_str());
            } else if contains_cyrillic(&descriptor.text) {
                layout = layout.with_hyphenation_language_tag("ru");
            }
            if is_bubble_text && descriptor.writing_mode == WritingMode::Horizontal {
                layout = layout.with_hyphenation_policy(HyphenationPolicy::LastResort);
            }
            if descriptor.auto_fit && !descriptor.point_text {
                layout
                    .with_max_font_size(maximum)
                    .with_min_font_size(minimum)
            } else {
                layout.with_font_size(descriptor.font_size.unwrap_or(maximum))
            }
        };

        let initial_builder = build_layout_for_inset(descriptor.text_inset, bounds);
        let mut layout = self
            .layout(&initial_builder, &descriptor.text)
            .map_err(|source| Error::Layout {
                entity: descriptor.entity,
                source,
            })?;

        // Safe Contour Padding Relaxation during auto-fit when text is pinched below 12pt.
        if is_bubble_text
            && descriptor.auto_fit
            && !descriptor.point_text
            && (layout.font_size < 12.0 || layout.overflowed())
            && descriptor.text_inset.iter().any(|&v| v >= 3.5)
        {
            let relaxed_inset = descriptor.text_inset.map(|v| (v - 2.5).max(1.5));
            let relaxed_bounds = inset(frame, relaxed_inset);
            if relaxed_bounds.width > 0.0 && relaxed_bounds.height > 0.0 {
                let relaxed_builder = build_layout_for_inset(relaxed_inset, relaxed_bounds);
                if let Ok(relaxed_layout) = self.layout(&relaxed_builder, &descriptor.text)
                    && ((!relaxed_layout.overflowed() && layout.overflowed())
                        || (relaxed_layout.overflowed() == layout.overflowed()
                            && relaxed_layout.font_size > layout.font_size + 0.25))
                {
                    layout = relaxed_layout;
                    bounds = relaxed_bounds;
                }
            }
        }

        let (mut x, mut y) = if descriptor.point_text {
            (bounds.x, bounds.y)
        } else {
            placement(bounds, layout.width, layout.height)
        };
        x += layout.placement_offset_x();
        y += layout.placement_offset_y();
        let transform = Affine::translate((f64::from(x), f64::from(y)));
        let color = descriptor.foreground_color;
        let mut options = TextRenderOptions {
            color,
            stroke: None,
            ..TextRenderOptions::default()
        };
        let mut scene = PreparedScene::default();
        let mut resources = Vec::new();
        if let Some(stroke) = descriptor.stroke {
            options.stroke = Some(stroke);
        }
        self.render(
            &mut scene,
            &mut resources,
            &layout,
            descriptor.writing_mode,
            &options,
            transform,
        );
        let mut diagnostics = Vec::new();
        if layout.font_size + f32::EPSILON < descriptor.minimum_font_size {
            diagnostics.push(RenderDiagnostic::TextBelowReadableSize {
                entity: descriptor.entity,
                font_size: layout.font_size,
                minimum_font_size: descriptor.minimum_font_size,
            });
        }
        if layout.overflowed() {
            diagnostics.push(RenderDiagnostic::TextOverflow {
                entity: descriptor.entity,
                available: bounds.into(),
                actual_width: layout.width,
                actual_height: layout.height,
                font_size: layout.font_size,
            });
        }
        let rendered_bounds = RenderBounds {
            x,
            y,
            width: layout.width,
            height: layout.height,
        };
        let stroke_padding = descriptor
            .stroke
            .map_or(0.0, |stroke| stroke.width_px.max(0.0));
        Ok(RenderedTextNode {
            scene: Arc::new(scene),
            resources: resources.into(),
            local_bounds: RenderBounds {
                x: rendered_bounds.x - stroke_padding,
                y: rendered_bounds.y - stroke_padding,
                width: rendered_bounds.width + stroke_padding * 2.0,
                height: rendered_bounds.height + stroke_padding * 2.0,
            },
            metadata: RenderedTextMetadata {
                rendered_bounds,
                layout_bounds: if descriptor.point_text {
                    rendered_bounds
                } else {
                    bounds.into()
                },
                post_script_fonts: fonts
                    .iter()
                    .map(|font| font.post_script_name().to_owned())
                    .collect(),
                font_size: layout.font_size,
                color,
            },
            diagnostics,
        })
    }
}

fn automatic_maximum(
    descriptor: &TextNodeDescriptor,
    bounds: LayoutBox,
    is_bubble_text: bool,
) -> f32 {
    if is_bubble_text {
        if descriptor.writing_mode.is_vertical() {
            bounds.height
        } else {
            bounds.width
        }
    } else {
        24.0
    }
}

fn is_chinese_or_japanese_language(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| matches!(primary.to_ascii_lowercase().as_str(), "ja" | "zh"))
}

fn inset(rect: LayoutBox, [top, right, bottom, left]: [f32; 4]) -> LayoutBox {
    LayoutBox {
        x: rect.x + left,
        y: rect.y + top,
        width: (rect.width - left - right).max(0.0),
        height: (rect.height - top - bottom).max(0.0),
    }
}

fn placement(rect: LayoutBox, width: f32, height: f32) -> (f32, f32) {
    let x = rect.x + (rect.width - width) * 0.5;
    let remaining = rect.height - height;
    let y = rect.y + remaining * 0.5;
    (x, y)
}

/// Color and outward dilation for one glyph draw pass (border or ordinary fill).
struct GlyphPaint {
    color: [u8; 4],
    dilation_px: f32,
}

fn draw_layout(
    scene: &mut PreparedScene,
    resources: &mut Vec<PreparedResource>,
    layout: &LayoutRun<'_>,
    writing_mode: WritingMode,
    options: &TextRenderOptions,
    transform: Affine,
    paint: GlyphPaint,
) {
    for line in &layout.lines {
        let (baseline_x, baseline_y) = match writing_mode {
            WritingMode::Horizontal | WritingMode::VerticalRl => line.baseline,
        };
        let mut pen_x = 0.0;
        let mut pen_y = 0.0;
        let mut start = 0;

        while start < line.glyphs.len() {
            let font = line.glyphs[start].font;
            let key = font_key(font);
            let mut end = start + 1;
            while end < line.glyphs.len() && font_key(line.glyphs[end].font) == key {
                end += 1;
            }

            let mut glyphs = Vec::with_capacity(end - start);
            for glyph in &line.glyphs[start..end] {
                glyphs.push(PreparedGlyph {
                    id: glyph.glyph_id,
                    x: options.padding + baseline_x + pen_x + glyph.x_offset,
                    y: options.padding + baseline_y + pen_y
                        - glyph.y_offset
                        - options.baseline_shift,
                });
                pen_x += glyph.x_advance;
                pen_y -= glyph.y_advance;
            }

            let font_id = ResourceId::for_font(font.bytes());
            if !resources.iter().any(|candidate| candidate.id() == font_id) {
                resources.push(PreparedResource::font_shared(font.shared_bytes()));
            }
            let glyph_transform = font
                .synthetic_skew()
                .map(|angle| Affine::skew(-(angle.to_radians().tan() as f64), 0.0).as_coeffs());
            let synthetic_bold = if font.synthetic_bold() { 1.0 } else { 0.0 };
            scene
                .commands
                .push(PreparedSceneCommand::GlyphRun(PreparedGlyphRun {
                    font: font_id,
                    font_index: font.index(),
                    font_size: layout.font_size,
                    normalized_coords: font.normalized_coords().to_vec(),
                    transform: transform.as_coeffs(),
                    glyph_transform,
                    // Hinting folds a uniform zoom into `font_size` for crisp fill text, but a
                    // border needs the real transform so its dilation amount stays proportional
                    // to that same zoom instead of being rendered at a fixed pixel size.
                    hint: options.hint_glyphs && paint.dilation_px == 0.0,
                    embolden: [synthetic_bold + paint.dilation_px; 2],
                    color: paint.color,
                    glyphs,
                }));
            start = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_rasterizer::{
        Bounds, CompositionCommand, LayerId, LayerKind as PreparedLayerKind, Point,
        PreparedContent, PreparedFrame, PreparedFrameBundle, PreparedFrameManifest, PreparedLayer,
        PreparedResourcePacket, PreparedResourceStore, Presentation, Revision,
    };
    use koharu_scene::EntityId;

    use super::*;
    use crate::fonts::FontSystem;

    #[test]
    fn automatic_size_preserves_free_text_default_and_balloon_extent() {
        let descriptor = TextNodeDescriptor {
            entity: EntityId::new(),
            text: "Hi".to_owned(),
            language: None,
            width: 240.0,
            height: 120.0,
            balloon_contour: None,
            flow_contour: None,
            preferred_font: Some("Arial".to_owned()),
            font_families: vec!["Arial".to_owned()],
            font_weight: None,
            font_style: None,
            font_size: Some(6.0),
            minimum_font_size: 9.0,
            auto_fit: true,
            alignment: TextAlign::Center,
            writing_mode: WritingMode::Horizontal,
            foreground_color: [0, 0, 0, 255],
            stroke: None,
            line_height: 1.2,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            text_inset: [0.0; 4],
            point_text: false,
        };
        let bounds = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 240.0,
            height: 120.0,
        };

        assert_eq!(automatic_maximum(&descriptor, bounds, false), 24.0);
        assert_eq!(automatic_maximum(&descriptor, bounds, true), 240.0);
    }

    #[test]
    fn rendered_text_survives_prepared_packet_round_trip() {
        let font = FontSystem::new().first_font().unwrap();
        let layout = TextLayout::new(&font)
            .with_font_size(24.0)
            .run("Koharu")
            .unwrap();
        let mut scene = PreparedScene::default();
        let mut resources = Vec::new();
        TextRenderer::new().render(
            &mut scene,
            &mut resources,
            &layout,
            WritingMode::Horizontal,
            &TextRenderOptions::default(),
            Affine::IDENTITY,
        );
        assert_eq!(resources.len(), 1);
        assert!(matches!(
            &resources[0],
            PreparedResource::Font { bytes, .. } if !bytes.is_empty()
        ));

        let layer = LayerId::from_bytes([2; 16]);
        let bundle = PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(7),
                page: LayerId::from_bytes([1; 16]),
                width: 160,
                height: 48,
                origin: (0, 0),
                normalization: Affine::IDENTITY.as_coeffs(),
                layers: vec![PreparedLayer {
                    id: layer,
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 160.0, y: 0.0 },
                        Point { x: 160.0, y: 48.0 },
                        Point { x: 0.0, y: 48.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 160.0,
                        height: 48.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 160.0,
                        height: 48.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: PreparedLayerKind::Text,
                    placement: Affine::IDENTITY.as_coeffs(),
                    content: PreparedContent::Vector(scene),
                    element_frame: None,
                }],
            },
            resources,
        };

        let encoded = bundle.manifest().unwrap().encode().unwrap();
        let manifest = PreparedFrameManifest::decode(&encoded).unwrap();
        let mut resources = PreparedResourceStore::default();
        for reference in manifest.required_resources() {
            let encoded = bundle
                .resource_packet(reference.id)
                .unwrap()
                .encode()
                .unwrap();
            resources.insert(PreparedResourcePacket::decode(&encoded).unwrap());
        }
        let compiled = manifest.compile(&resources).unwrap();
        assert_eq!(compiled.revision(), Revision::new(7));
        assert_eq!(compiled.layers()[0].id(), layer);
        assert!(matches!(
            compiled.composition_commands(1).as_slice(),
            [CompositionCommand::Vector(_)]
        ));
    }
}
