use std::time::Duration;

use anyhow::Result;
use koharu_renderer::CopyfitDiagnostics;
use koharu_secrets::ExposeSecret as _;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_JEV_ENDPOINT: &str = "https://api.typesafe.ai/v1/system-one/evaluate";
const JEV_TIMEOUT: Duration = Duration::from_secs(6);

/// TypeSafe System One binary probability primitive (`Noul`):
/// represents a calibrated probability `p in [0.0, 1.0]` for a yes/no proposition.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Noul(pub f64);

impl Noul {
    #[must_use]
    pub fn probability(self) -> f64 {
        self.0.clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn is_affirmative(self, threshold: f64) -> bool {
        self.probability() >= threshold
    }
}

/// TypeSafe System One closed-set selection primitive (`Choice`):
/// selects strictly one option from a fixed list of candidates without generating free text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice<T> {
    pub selected: T,
    pub confidence: f64,
}

/// TypeSafe System One rubric evaluation primitive (`Score`):
/// normalized quality score in `[0.0, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Score(pub f64);

impl Score {
    #[must_use]
    pub fn value(self) -> f64 {
        self.0.clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrReadingOrderChoice {
    Rtl,
    Ltr,
    Corrupted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrRetryStrategyChoice {
    Accept,
    StripFuriganaAndRetry,
    FlipOrientationAndRetry,
    SwitchOcrModel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcrJudgeInput {
    pub region_id: String,
    pub candidate_rtl: String,
    pub candidate_ltr: Option<String>,
    pub rtl_inspector_score: f64,
    pub ltr_inspector_score: Option<f64>,
    pub likely_furigana_noise: bool,
    pub aspect_ratio: f64,
    pub column_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OcrJudgeVerdict {
    pub reading_order: Choice<OcrReadingOrderChoice>,
    pub is_valid_japanese: Noul,
    pub retry_strategy: Choice<OcrRetryStrategyChoice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CopyfitVariantChoice {
    Full,
    Compact,
    UltraShort,
    LobeSplit,
    NeedsRegeneration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopyfitCandidateMetrics {
    pub variant: CopyfitVariantChoice,
    pub text: String,
    pub diagnostics: CopyfitDiagnostics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopyfitJudgeInput {
    pub region_id: String,
    pub source_text: String,
    pub bubble_width: f32,
    pub bubble_height: f32,
    pub lobe_count: usize,
    pub candidates: Vec<CopyfitCandidateMetrics>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopyfitJudgeVerdict {
    pub region_id: String,
    /// System One `Choice`: selects the best fitting variant or requests LLM regeneration.
    pub selected_variant: Choice<CopyfitVariantChoice>,
    /// System One `Noul`: probability that meaning, character gender, and emotional tone are preserved.
    pub meaning_preserved: Noul,
    /// System One `Score`: overall readability and typesetting rubric score in `[0.0, 1.0]`.
    pub readability_score: Score,
}

#[derive(Clone, Default)]
pub struct JevClient {
    endpoint: String,
}

impl JevClient {
    #[must_use]
    pub fn new() -> Self {
        let endpoint = std::env::var("TYPESAFE_API_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_JEV_ENDPOINT.to_owned());
        Self { endpoint }
    }

    fn api_key() -> Option<String> {
        for secret_name in ["typesafe", "jev"] {
            if let Ok(Some(secret)) = koharu_secrets::get(secret_name) {
                let exposed = secret.expose_secret().trim().to_owned();
                if !exposed.is_empty() {
                    return Some(exposed);
                }
            }
        }
        for env_name in ["TYPESAFE_API_KEY", "JEV_API_KEY"] {
            if let Ok(value) = std::env::var(env_name) {
                let trimmed = value.trim().to_owned();
                if !trimmed.is_empty() {
                    return Some(trimmed);
                }
            }
        }
        None
    }

    /// Evaluates a batch of OCR regions using TypeSafe System One (`Choice` + `Noul`),
    /// falling back to the deterministic Kinsoku Shori / orthography scores when offline
    /// or when no `TYPESAFE_API_KEY` is configured.
    pub async fn judge_ocr_batch(&self, inputs: &[OcrJudgeInput]) -> Vec<OcrJudgeVerdict> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if let Some(key) = Self::api_key()
            && let Ok(verdicts) = self.request_ocr_batch(&key, inputs).await
            && verdicts.len() == inputs.len()
        {
            return verdicts;
        }
        inputs.iter().map(deterministic_ocr_verdict).collect()
    }

    /// Evaluates a batch of translation copyfitting candidates (`Full`, `Compact`, `UltraShort`,
    /// `LobeSplit`) alongside their `CopyfitDiagnostics` using TypeSafe System One (`Choice`, `Noul`, `Score`),
    /// falling back to `deterministic_copyfit_verdict` when offline or when no API key is configured.
    pub async fn judge_copyfitting_batch(
        &self,
        inputs: &[CopyfitJudgeInput],
    ) -> Vec<CopyfitJudgeVerdict> {
        if inputs.is_empty() {
            return Vec::new();
        }
        if let Some(key) = Self::api_key()
            && let Ok(verdicts) = self.request_copyfitting_batch(&key, inputs).await
            && verdicts.len() == inputs.len()
        {
            return verdicts;
        }
        inputs.iter().map(deterministic_copyfit_verdict).collect()
    }

    async fn request_ocr_batch(
        &self,
        key: &str,
        inputs: &[OcrJudgeInput],
    ) -> Result<Vec<OcrJudgeVerdict>> {
        let client = koharu_runtime::http_client()?;
        let items = inputs
            .iter()
            .map(|item| {
                json!({
                    "region_id": item.region_id,
                    "candidate_rtl": item.candidate_rtl,
                    "candidate_ltr": item.candidate_ltr,
                    "aspect_ratio": item.aspect_ratio,
                    "column_count": item.column_count,
                    "schema": {
                        "reading_order": {
                            "type": "Choice",
                            "options": ["rtl", "ltr", "corrupted"]
                        },
                        "is_valid_japanese": {
                            "type": "Noul"
                        },
                        "retry_strategy": {
                            "type": "Choice",
                            "options": [
                                "accept",
                                "strip_furigana_and_retry",
                                "flip_orientation_and_retry",
                                "switch_ocr_model"
                            ]
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        let response = client
            .post(&self.endpoint)
            .bearer_auth(key)
            .timeout(JEV_TIMEOUT)
            .json(&json!({
                "task": "ocr_validation_v1",
                "items": items,
            }))
            .send()
            .await?
            .error_for_status()?;

        let payload: OcrBatchResponse = response.json().await?;
        Ok(payload
            .verdicts
            .into_iter()
            .map(|item| OcrJudgeVerdict {
                reading_order: item.reading_order,
                is_valid_japanese: item.is_valid_japanese,
                retry_strategy: item.retry_strategy,
            })
            .collect())
    }

    async fn request_copyfitting_batch(
        &self,
        key: &str,
        inputs: &[CopyfitJudgeInput],
    ) -> Result<Vec<CopyfitJudgeVerdict>> {
        let client = koharu_runtime::http_client()?;
        let items = inputs
            .iter()
            .map(|item| {
                json!({
                    "region_id": item.region_id,
                    "source_text": item.source_text,
                    "bubble_width": item.bubble_width,
                    "bubble_height": item.bubble_height,
                    "lobe_count": item.lobe_count,
                    "candidates": item.candidates,
                    "schema": {
                        "selected_variant": {
                            "type": "Choice",
                            "options": [
                                "full",
                                "compact",
                                "ultra_short",
                                "lobe_split",
                                "needs_regeneration"
                            ]
                        },
                        "meaning_preserved": { "type": "Noul" },
                        "readability_score": { "type": "Score" }
                    }
                })
            })
            .collect::<Vec<_>>();

        let response = client
            .post(&self.endpoint)
            .bearer_auth(key)
            .timeout(JEV_TIMEOUT)
            .json(&json!({
                "task": "copyfit_selection_batch_v1",
                "items": items,
            }))
            .send()
            .await?
            .error_for_status()?;

        let payload: CopyfitBatchResponse = response.json().await?;
        Ok(payload
            .verdicts
            .into_iter()
            .zip(inputs.iter())
            .map(|(item, input)| CopyfitJudgeVerdict {
                region_id: item.region_id.unwrap_or_else(|| input.region_id.clone()),
                selected_variant: item.selected_variant,
                meaning_preserved: item.meaning_preserved,
                readability_score: item.readability_score,
            })
            .collect())
    }
}

#[derive(Deserialize)]
struct OcrBatchResponse {
    verdicts: Vec<OcrVerdictItem>,
}

#[derive(Deserialize)]
struct OcrVerdictItem {
    reading_order: Choice<OcrReadingOrderChoice>,
    is_valid_japanese: Noul,
    retry_strategy: Choice<OcrRetryStrategyChoice>,
}

#[derive(Deserialize)]
struct CopyfitBatchResponse {
    verdicts: Vec<CopyfitVerdictItem>,
}

#[derive(Deserialize)]
struct CopyfitVerdictItem {
    #[serde(default)]
    region_id: Option<String>,
    selected_variant: Choice<CopyfitVariantChoice>,
    meaning_preserved: Noul,
    readability_score: Score,
}

pub fn deterministic_ocr_verdict(input: &OcrJudgeInput) -> OcrJudgeVerdict {
    let rtl_score = input.rtl_inspector_score.clamp(0.0, 1.0);
    let ltr_score = input.ltr_inspector_score.map(|s| s.clamp(0.0, 1.0));

    let (best_order, best_score, margin) = match ltr_score {
        Some(ltr) if ltr > rtl_score + 0.08 => (
            OcrReadingOrderChoice::Ltr,
            ltr,
            (ltr - rtl_score).clamp(0.05, 1.0),
        ),
        Some(ltr) => (
            OcrReadingOrderChoice::Rtl,
            rtl_score,
            (rtl_score - ltr).abs().max(0.15).clamp(0.0, 1.0),
        ),
        None => (OcrReadingOrderChoice::Rtl, rtl_score, rtl_score),
    };

    if best_score < 0.42 || input.candidate_rtl.trim().is_empty() {
        let strategy = if input.likely_furigana_noise {
            OcrRetryStrategyChoice::StripFuriganaAndRetry
        } else if input.aspect_ratio >= 0.80 && input.aspect_ratio <= 1.35 {
            OcrRetryStrategyChoice::FlipOrientationAndRetry
        } else {
            OcrRetryStrategyChoice::SwitchOcrModel
        };
        return OcrJudgeVerdict {
            reading_order: Choice {
                selected: OcrReadingOrderChoice::Corrupted,
                confidence: (1.0 - best_score).clamp(0.5, 1.0),
            },
            is_valid_japanese: Noul(best_score),
            retry_strategy: Choice {
                selected: strategy,
                confidence: 0.85,
            },
        };
    }

    let retry_strategy = if input.likely_furigana_noise && best_score < 0.68 {
        OcrRetryStrategyChoice::StripFuriganaAndRetry
    } else {
        OcrRetryStrategyChoice::Accept
    };

    OcrJudgeVerdict {
        reading_order: Choice {
            selected: best_order,
            confidence: (0.5 + margin * 0.5).clamp(0.5, 0.99),
        },
        is_valid_japanese: Noul(best_score),
        retry_strategy: Choice {
            selected: retry_strategy,
            confidence: best_score,
        },
    }
}

/// Deterministic offline fallback for Jev Copyfitting Judge (`judge_copyfitting_batch`).
///
/// Objective function:
/// 1. Must not overflow (`!diagnostics.overflow`);
/// 2. `resulting_font_size >= 12.0` (or closest to `target_font_size`);
/// 3. No `has_ugly_single_char_wrap`;
/// 4. Minimal length loss relative to `Full` (with a structural bonus for `LobeSplit` when
///    `lobe_count >= 2`).
/// If all candidates overflow or fail minimum readability (`< 10.5pt`), returns
/// `CopyfitVariantChoice::NeedsRegeneration`.
#[must_use]
pub fn deterministic_copyfit_verdict(input: &CopyfitJudgeInput) -> CopyfitJudgeVerdict {
    const MIN_COMFORTABLE_FONT_SIZE: f32 = 12.0;

    let full_len = input
        .candidates
        .iter()
        .find(|c| c.variant == CopyfitVariantChoice::Full)
        .map(|c| c.text.chars().count().max(1))
        .unwrap_or(1) as f64;

    // Fast path: if `Full` fits without overflow, achieves `>= 12.0pt`, has no ugly single-char
    // wrap, and this is not a multi-lobe bubble where `LobeSplit` improves font size, keep `Full`.
    if let Some(full) = input
        .candidates
        .iter()
        .find(|c| c.variant == CopyfitVariantChoice::Full)
    {
        let lobe_prefers_split = input.lobe_count >= 2
            && input
                .candidates
                .iter()
                .find(|c| c.variant == CopyfitVariantChoice::LobeSplit)
                .is_some_and(|lobe| {
                    !lobe.diagnostics.overflow
                        && !lobe.diagnostics.has_ugly_single_char_wrap
                        && lobe.diagnostics.resulting_font_size
                            >= full.diagnostics.resulting_font_size
                });

        if !full.diagnostics.overflow
            && full.diagnostics.resulting_font_size >= MIN_COMFORTABLE_FONT_SIZE
            && !full.diagnostics.has_ugly_single_char_wrap
            && !lobe_prefers_split
        {
            return CopyfitJudgeVerdict {
                region_id: input.region_id.clone(),
                selected_variant: Choice {
                    selected: CopyfitVariantChoice::Full,
                    confidence: 0.96,
                },
                meaning_preserved: Noul(0.98),
                readability_score: Score(0.95),
            };
        }
    }

    let mut best_candidate: Option<(&CopyfitCandidateMetrics, f64, f64)> = None;
    for candidate in &input.candidates {
        if candidate.text.trim().is_empty() {
            continue;
        }
        let diag = &candidate.diagnostics;
        let cand_len = candidate.text.chars().count() as f64;
        let length_retention = (cand_len / full_len).clamp(0.35, 1.0);

        let meaning_prob = match candidate.variant {
            CopyfitVariantChoice::Full | CopyfitVariantChoice::LobeSplit => 0.98,
            CopyfitVariantChoice::Compact => (0.86 + 0.12 * length_retention).clamp(0.82, 0.96),
            CopyfitVariantChoice::UltraShort => (0.72 + 0.20 * length_retention).clamp(0.68, 0.90),
            CopyfitVariantChoice::NeedsRegeneration => 0.50,
        };

        let mut score = 0.0f64;
        if !diag.overflow {
            score += 0.45;
        } else {
            score -= 0.45;
        }

        let size_ratio =
            f64::from(diag.resulting_font_size / diag.target_font_size.max(MIN_COMFORTABLE_FONT_SIZE))
                .clamp(0.5, 1.15);
        score += (size_ratio - 0.5) * 0.55;

        if diag.resulting_font_size >= MIN_COMFORTABLE_FONT_SIZE {
            score += 0.18;
        } else if diag.resulting_font_size < 10.5 {
            score -= 0.22;
        }

        if diag.has_ugly_single_char_wrap {
            score -= 0.16;
        }

        // Prefer minimal length loss relative to `Full`.
        score += length_retention * 0.16;

        if candidate.variant == CopyfitVariantChoice::LobeSplit && input.lobe_count >= 2 {
            score += 0.08;
        }

        if best_candidate
            .as_ref()
            .is_none_or(|(_, best_score, _)| score > *best_score)
        {
            best_candidate = Some((candidate, score, meaning_prob));
        }
    }

    let Some((chosen, raw_score, meaning_prob)) = best_candidate else {
        return CopyfitJudgeVerdict {
            region_id: input.region_id.clone(),
            selected_variant: Choice {
                selected: CopyfitVariantChoice::NeedsRegeneration,
                confidence: 0.90,
            },
            meaning_preserved: Noul(0.30),
            readability_score: Score(0.20),
        };
    };

    if chosen.diagnostics.overflow || chosen.diagnostics.resulting_font_size < 10.2 {
        return CopyfitJudgeVerdict {
            region_id: input.region_id.clone(),
            selected_variant: Choice {
                selected: CopyfitVariantChoice::NeedsRegeneration,
                confidence: 0.88,
            },
            meaning_preserved: Noul(meaning_prob),
            readability_score: Score(raw_score.clamp(0.15, 0.48)),
        };
    }

    CopyfitJudgeVerdict {
        region_id: input.region_id.clone(),
        selected_variant: Choice {
            selected: chosen.variant,
            confidence: raw_score.clamp(0.60, 0.98),
        },
        meaning_preserved: Noul(meaning_prob),
        readability_score: Score(raw_score.clamp(0.0, 1.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_diag(
        font_size: f32,
        overflow: bool,
        ugly_wrap: bool,
        lines: &[&str],
    ) -> CopyfitDiagnostics {
        CopyfitDiagnostics {
            resulting_font_size: font_size,
            target_font_size: 13.0,
            overflow,
            line_count: lines.len(),
            lines: lines.iter().map(|s| (*s).to_owned()).collect(),
            max_word_char_len: 8,
            has_ugly_single_char_wrap: ugly_wrap,
            area_utilization: 0.65,
            relaxed_inset: [4.0; 4],
            lobe_count: 1,
            recommended_max_chars: 42,
            recommended_max_word_len: 9,
        }
    }

    #[test]
    fn deterministic_copyfit_prefers_full_when_readable_and_clean() {
        let input = CopyfitJudgeInput {
            region_id: "reg-1".to_owned(),
            source_text: "本当ですか？".to_owned(),
            bubble_width: 90.0,
            bubble_height: 120.0,
            lobe_count: 1,
            candidates: vec![
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::Full,
                    text: "Ты говоришь правду?".to_owned(),
                    diagnostics: sample_diag(13.5, false, false, &["Ты говоришь", "правду?"]),
                },
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::Compact,
                    text: "Это правда?".to_owned(),
                    diagnostics: sample_diag(15.0, false, false, &["Это", "правда?"]),
                },
            ],
        };

        let verdict = deterministic_copyfit_verdict(&input);
        assert_eq!(verdict.selected_variant.selected, CopyfitVariantChoice::Full);
        assert!(verdict.meaning_preserved.is_affirmative(0.90));
        assert!(verdict.readability_score.value() >= 0.85);
    }

    #[test]
    fn deterministic_copyfit_selects_compact_when_full_pinches_font_size() {
        let input = CopyfitJudgeInput {
            region_id: "reg-2".to_owned(),
            source_text: "どうしてそんなことを言うの？".to_owned(),
            bubble_width: 68.0,
            bubble_height: 130.0,
            lobe_count: 1,
            candidates: vec![
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::Full,
                    text: "Почему ты вообще говоришь такие вещи в подобный момент?".to_owned(),
                    diagnostics: sample_diag(9.5, false, true, &["Почему ты", "в", "подобный"]),
                },
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::Compact,
                    text: "Зачем ты такое говоришь?".to_owned(),
                    diagnostics: sample_diag(12.8, false, false, &["Зачем ты", "такое", "говоришь?"]),
                },
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::UltraShort,
                    text: "Зачем так?".to_owned(),
                    diagnostics: sample_diag(14.5, false, false, &["Зачем", "так?"]),
                },
            ],
        };

        let verdict = deterministic_copyfit_verdict(&input);
        assert_eq!(
            verdict.selected_variant.selected,
            CopyfitVariantChoice::Compact
        );
        assert!(verdict.meaning_preserved.is_affirmative(0.85));
    }

    #[test]
    fn deterministic_copyfit_requests_regeneration_when_all_variants_overflow() {
        let input = CopyfitJudgeInput {
            region_id: "reg-3".to_owned(),
            source_text: "無理だ".to_owned(),
            bubble_width: 44.0,
            bubble_height: 50.0,
            lobe_count: 1,
            candidates: vec![
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::Full,
                    text: "Это совершенно невозможно сделать!".to_owned(),
                    diagnostics: sample_diag(9.0, true, true, &["Это совершенно"]),
                },
                CopyfitCandidateMetrics {
                    variant: CopyfitVariantChoice::UltraShort,
                    text: "Это невозможно сделать!".to_owned(),
                    diagnostics: sample_diag(9.0, true, false, &["Это невозможно"]),
                },
            ],
        };

        let verdict = deterministic_copyfit_verdict(&input);
        assert_eq!(
            verdict.selected_variant.selected,
            CopyfitVariantChoice::NeedsRegeneration
        );
    }
}
