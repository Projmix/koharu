use std::collections::BTreeMap;

use anyhow::{Context, bail, ensure};
use indoc::indoc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Language, OcrRequest, TranslationContext, TranslationRequest};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    let segments = request
        .segments
        .iter()
        .enumerate()
        .map(|(id, source_text)| {
            let metadata = request.segment_metadata(id);
            TranslationInputSegment {
                id: request.segment_id(id).to_owned(),
                page_id: request.page_id(id),
                region_id: request.region_id(id),
                order_index: request.order_index(id),
                page_index: request.page_number(id),
                source_text,
                role: metadata.role.as_deref(),
                inside_balloon: metadata.inside_balloon,
                bubble_id: metadata.bubble_id.as_deref(),
                unit_id: metadata.unit_id.as_deref(),
                unit_type: metadata.unit_type.as_deref(),
                is_sound_effect: metadata.is_sound_effect,
            }
        })
        .collect::<Vec<_>>();
    let pages = request.page_numbers.as_ref().map(|page_numbers| {
        // Keep the caller's project order. A map keyed by UUID would reorder pages
        // lexicographically and break the manga reading sequence in the prompt.
        let mut pages = Vec::<(String, usize, Vec<TranslationInputSegment<'_>>)>::new();
        for (segment, page_index) in segments.iter().cloned().zip(page_numbers) {
            let page = (segment.page_id.clone(), *page_index);
            if let Some((_, _, page_segments)) = pages
                .iter_mut()
                .find(|(id, index, _)| *id == page.0 && *index == page.1)
            {
                page_segments.push(segment);
            } else {
                pages.push((page.0, page.1, vec![segment]));
            }
        }
        pages
            .into_iter()
            .map(|(page_id, page_index, segments)| {
                let units = if request.units.is_empty() {
                    // A chapter request normally receives units from the pipeline. Keep the
                    // lower-level translator safe for direct callers too: never serialize a
                    // chapter page with no translatable segments.
                    vec![TranslationInputUnit {
                        id: format!("unit-page-{page_index:04}-0000"),
                        unit_type: "page".to_owned(),
                        region_ids: segments
                            .iter()
                            .map(|segment| segment.region_id.clone())
                            .collect(),
                        segments,
                    }]
                } else {
                    request
                        .units
                        .iter()
                        .filter(|unit| unit.page_id == page_id)
                        .map(|unit| TranslationInputUnit {
                            id: unit.id.clone(),
                            unit_type: unit.unit_type.clone(),
                            region_ids: unit.region_ids.clone(),
                            segments: unit
                                .segment_ids
                                .iter()
                                .filter_map(|id| {
                                    segments.iter().find(|segment| &segment.id == id).cloned()
                                })
                                .collect(),
                        })
                        .collect::<Vec<_>>()
                };
                TranslationInputPage {
                    page_id,
                    page_index,
                    units,
                }
            })
            .collect::<Vec<_>>()
    });
    let input = TranslationInput {
        mode: request
            .page_numbers
            .as_ref()
            .map(|_| "chapter")
            .unwrap_or("page"),
        unit_policy: request
            .unit_policy
            .as_deref()
            .or_else(|| request.is_chapter().then_some("page_only")),
        source_language: request.source_language,
        target_language: request.target_language,
        references: &request.context,
        segments: pages.is_none().then_some(segments),
        pages,
    };
    let user = serde_json::to_string(&input).context("failed to serialize translation input")?;
    Ok((translation_system_prompt(request), user))
}

pub(crate) fn translations(
    provider: &str,
    text: &str,
    request: &TranslationRequest,
) -> anyhow::Result<Vec<String>> {
    let TranslationOutput {
        translations: output_translations,
        _notes: _,
    } = serde_json::from_str::<TranslationOutput>(text)
        .with_context(|| format!("{provider} returned invalid translation JSON"))?;
    ensure!(
        output_translations.len() == request.segments.len(),
        "{provider} returned {} translations, expected {}",
        output_translations.len(),
        request.segments.len()
    );
    let positions = request
        .segment_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut translations = vec![None; request.segments.len()];

    for translation in output_translations {
        let Some(&index) = positions.get(translation.id.as_str()) else {
            bail!(
                "{provider} returned unknown translation ID {}",
                translation.id
            );
        };
        if translations[index].is_some() {
            bail!(
                "{provider} returned duplicate translation ID {}",
                translation.id
            );
        }
        ensure!(
            translation.page_id == request.page_id(index),
            "{provider} returned page {} for translation ID {}, expected {}",
            translation.page_id,
            translation.id,
            request.page_id(index)
        );
        ensure!(
            translation.region_id == request.region_id(index),
            "{provider} returned region {} for translation ID {}, expected {}",
            translation.region_id,
            translation.id,
            request.region_id(index)
        );
        ensure!(
            translation.source_text == request.segments[index],
            "{provider} changed source text for translation ID {}",
            translation.id
        );
        translations[index] = Some(translation.translated_text);
    }

    let missing = request
        .segment_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| translations[index].is_none().then_some(id.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "{provider} omitted translation IDs: {}",
        missing.join(", ")
    );
    Ok(translations
        .into_iter()
        .map(|translation| translation.expect("missing IDs were rejected above"))
        .collect())
}

pub(crate) fn output_schema(expected: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The stable ID copied from the corresponding input segment."
                        },
                        "page_id": {
                            "type": "string",
                            "description": "The page_id copied exactly from the corresponding input page."
                        },
                        "region_id": {
                            "type": "string",
                            "description": "The region_id copied exactly from the corresponding input segment."
                        },
                        "source_text": {
                            "type": "string",
                            "description": "The source text copied exactly from the corresponding input segment."
                        },
                        "translated_text": {
                            "type": "string",
                            "description": "The translation of the input segment with this ID."
                        }
                    },
                    "required": ["id", "page_id", "region_id", "source_text", "translated_text"],
                    "additionalProperties": false
                }
            },
            "notes": {
                "type": "string",
                "description": "Leave empty unless a short non-actionable note is necessary."
            }
        },
        "required": ["translations", "notes"],
        "additionalProperties": false
    })
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    let source = request
        .source_language
        .map(|language| language.to_string())
        .unwrap_or_else(|| "the detected source language".to_owned());
    let mut prompt = format!(
        indoc! {"
            You are a professional manga translator.

            Translation requirements:
            - Translate every input segment from {source} into natural {target}.
            - Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects.
            - Localize idioms and sound effects naturally while keeping wording concise enough for speech bubbles.
            - Use surrounding segments only for disambiguation and continuity; never merge or split segments.
            - Each segment may include semantic metadata: `role` is the existing scene text role, and `inside_balloon` indicates the existing detected dialogue-balloon relation. Use these hints to distinguish dialogue, captions, and sound effects, but never alter segment boundaries.
            - Write every `translated_text` value only in {target}; do not include notes, explanations, or alternatives.
            - Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {target} conventions.

            Output requirements:
            - Each input segment has a stable string `id`, `page_id`, and `region_id`.
            - In chapter mode, the stable ID format is exactly `page:<page_uuid>::region:<region_uuid>`.
            - Return only a JSON object with `translations` and `notes`. The translations array contains one object with `id`, `page_id`, `region_id`, `source_text`, and `translated_text` for every input segment.
            - Copy every input ID, page_id, region_id, and source text exactly; translate only `translated_text` and leave notes empty.
            - Every input ID must appear exactly once; order does not matter.
            - Never merge, split, omit, duplicate, or add segments.
            - The response is imported automatically. A changed ID, page, or source text invalidates the entire response.
        "},
        source = source,
        target = request.target_language,
    )
    .trim_end()
    .to_owned();

    if request.is_chapter() {
        prompt.push_str("\n\n");
        prompt.push_str(
            indoc! {"
                Chapter requirements:
                - Translate the supplied pages as one coherent chapter; the supplied page order is the manga reading sequence.
                - Keep character names, terminology, forms of address, voice, and recurring sound-effect localization consistent across all pages.
                - Use `unit_type`, `bubble_id`, `role`, `inside_balloon`, and `is_sound_effect` to distinguish dialogue groups, captions, labels, and onomatopoeia. Translate each according to its function while preserving its exact stable segment ID.
                - Units are context only: return one translation object for every segment, never one result per unit.
                - Do not move text between pages, combine nearby balloons, or attach a translation to a different segment even when that would read more naturally.
            "}
            .trim_end(),
        );
    }

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context only to preserve terminology, character voice, and dialogue continuity.
            Do not translate or return the context entries.
        "}.trim_end());
    }

    if request.image.is_some() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            Use the attached original page image as visual context for speaker identity, tone, layout, and ambiguous OCR.
            Translate only the supplied segments; do not add text seen in the image that is absent from the input segments.
        "}.trim_end());
    }

    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        prompt.push_str("\n\nAdditional instructions:\n");
        prompt.push_str(instructions);
    }
    prompt
}

#[derive(Serialize)]
struct TranslationInput<'a> {
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_policy: Option<&'a str>,
    source_language: Option<Language>,
    target_language: Language,
    references: &'a [TranslationContext],
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Vec<TranslationInputSegment<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pages: Option<Vec<TranslationInputPage<'a>>>,
}

pub(crate) fn ocr_prompts(request: &OcrRequest) -> (String, String) {
    let source = request
        .source_language
        .map(|language| language.to_string())
        .unwrap_or_else(|| "the configured source language".to_owned());
    let mut system = indoc! {"
        You are an OCR engine for manga and comics.
        Transcribe every visible character in the attached cropped text region exactly as written.
        Preserve the original language, punctuation, repeated characters, ruby/furigana, and meaningful line breaks.
        Do not translate, explain, normalize, autocorrect, censor, or infer text that is not visible.
        Return only a JSON object with one string field named `text`.
    "}
    .trim_end()
    .to_owned();
    system.push_str("\n\nExpected source language: ");
    system.push_str(&source);
    system.push_str(". Do not translate from this language; transcribe it exactly.");
    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        system.push_str("\n\nAdditional instructions:\n");
        system.push_str(instructions);
    }
    (
        system,
        format!(
            "Transcribe the attached manga text crop in {source}. Return only {{\"text\":\"...\"}}."
        ),
    )
}

pub(crate) fn ocr_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Exact transcription of the attached image crop."
            }
        },
        "required": ["text"],
        "additionalProperties": false
    })
}

pub(crate) fn ocr_text(provider: &str, text: &str) -> anyhow::Result<String> {
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<OcrOutput>();
    let output = values
        .next()
        .transpose()
        .with_context(|| format!("{provider} returned invalid OCR JSON"))?
        .with_context(|| format!("{provider} returned empty OCR JSON"))?;
    Ok(output.text)
}

#[derive(Clone, Serialize)]
struct TranslationInputSegment<'a> {
    id: String,
    page_id: String,
    region_id: String,
    page_index: usize,
    order_index: usize,
    source_text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    inside_balloon: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bubble_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_type: Option<&'a str>,
    is_sound_effect: bool,
}

#[derive(Serialize)]
struct TranslationInputPage<'a> {
    page_id: String,
    page_index: usize,
    units: Vec<TranslationInputUnit<'a>>,
}

#[derive(Serialize)]
struct TranslationInputUnit<'a> {
    id: String,
    unit_type: String,
    region_ids: Vec<String>,
    segments: Vec<TranslationInputSegment<'a>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
    #[serde(rename = "notes")]
    _notes: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TranslationOutputSegment {
    id: String,
    page_id: String,
    region_id: String,
    source_text: String,
    translated_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OcrOutput {
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TranslationSegmentMetadata, TranslationUnit};

    #[test]
    fn parses_plain_json() {
        let request = TranslationRequest::new(["one", "two"], Language::English);
        let expected = vec!["hello".to_owned(), "world".to_owned()];
        let response = r#"{"translations":[{"id":"segment:0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"hello"},{"id":"segment:1","page_id":"page:1","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert_eq!(translations("test", response, &request).unwrap(), expected);
    }

    #[test]
    fn rejects_invalid_json() {
        let request = TranslationRequest::new(["one", "two"], Language::English);
        for response in [
            r#"{translations: [{id: 0, translated_text: 'hello'}]}"#,
            r#"Here is the result: {"translations": []}"#,
            "{\"translations\":[{\"id\":0",
            "```json\n{\"translations\":[]}\n```",
            r#"{"translations":[{"id":"0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"hello"}],"notes":""}"#,
        ] {
            assert!(translations("test", response, &request).is_err());
        }
    }

    #[test]
    fn restores_input_order_from_ids() {
        let request = TranslationRequest::new(["one", "two"], Language::English);
        let response = r#"{"translations":[{"id":"segment:1","page_id":"page:1","region_id":"region:1","source_text":"two","translated_text":"world"},{"id":"segment:0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"hello"}],"notes":""}"#;
        assert_eq!(
            translations("test", response, &request).unwrap(),
            ["hello", "world"]
        );
    }

    #[test]
    fn rejects_duplicate_missing_and_out_of_range_ids() {
        let request = TranslationRequest::new(["one", "two"], Language::English);
        let short = r#"{"translations":[{"id":"segment:1","page_id":"page:1","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert!(translations("test", short, &request).is_err());

        let duplicate = r#"{"translations":[{"id":"segment:0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"hello"},{"id":"segment:0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"duplicate"}],"notes":""}"#;
        assert!(translations("test", duplicate, &request).is_err());

        let unknown = r#"{"translations":[{"id":"segment:0","page_id":"page:1","region_id":"region:0","source_text":"one","translated_text":"hello"},{"id":"segment:9","page_id":"page:1","region_id":"region:9","source_text":"extra","translated_text":"extra"}],"notes":""}"#;
        assert!(translations("test", unknown, &request).is_err());
    }

    #[test]
    fn rejects_changed_page_or_source_text() {
        let request = TranslationRequest::new(["one", "two"], Language::English)
            .with_page_numbers([2, 3])
            .unwrap();
        let wrong_page = r#"{"translations":[{"id":"segment:0","page_id":"page:2","region_id":"region:0","source_text":"one","translated_text":"hello"},{"id":"segment:1","page_id":"page:2","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert!(translations("test", wrong_page, &request).is_err());

        let changed_source = r#"{"translations":[{"id":"segment:0","page_id":"page:2","region_id":"region:0","source_text":"ONE","translated_text":"hello"},{"id":"segment:1","page_id":"page:3","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert!(translations("test", changed_source, &request).is_err());

        let changed_region = r#"{"translations":[{"id":"segment:0","page_id":"page:2","region_id":"region:wrong","source_text":"one","translated_text":"hello"},{"id":"segment:1","page_id":"page:3","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert!(translations("test", changed_region, &request).is_err());

        let unknown_field = r#"{"translations":[{"id":"segment:0","page_id":"page:2","region_id":"region:0","source_text":"one","translated_text":"hello","extra":true},{"id":"segment:1","page_id":"page:3","region_id":"region:1","source_text":"two","translated_text":"world"}],"notes":""}"#;
        assert!(translations("test", unknown_field, &request).is_err());
    }

    #[test]
    fn prompt_payload_contains_ordered_context() {
        let request = TranslationRequest::new(["new"], Language::English)
            .with_context([TranslationContext::new("old", "previous")]);
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["references"][0]["source"], "old");
        assert_eq!(input["references"][0]["translation"], "previous");
        assert_eq!(input["segments"][0]["id"], "segment:0");
        assert_eq!(input["segments"][0]["page_index"], 1);
        assert_eq!(input["segments"][0]["source_text"], "new");
    }

    #[test]
    fn prompt_payload_carries_scene_semantics_without_changing_identity() {
        let request = TranslationRequest::new(["sound"], Language::Russian)
            .with_segment_metadata([TranslationSegmentMetadata {
                role: Some("dev.koharu.text.onomatopoeia".to_owned()),
                inside_balloon: false,
                ..Default::default()
            }])
            .unwrap();
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["segments"][0]["id"], "segment:0");
        assert_eq!(input["segments"][0]["source_text"], "sound");
        assert_eq!(input["segments"][0]["role"], "dev.koharu.text.onomatopoeia");
        assert_eq!(input["segments"][0]["inside_balloon"], false);
    }

    #[test]
    fn chapter_payload_groups_pages_with_global_ids() {
        let request = TranslationRequest::new(["first", "second", "third"], Language::English)
            .with_segment_metadata([
                TranslationSegmentMetadata {
                    page_id: Some("page-z".to_owned()),
                    region_id: Some("region-a".to_owned()),
                    ..Default::default()
                },
                TranslationSegmentMetadata {
                    page_id: Some("page-z".to_owned()),
                    region_id: Some("region-b".to_owned()),
                    ..Default::default()
                },
                TranslationSegmentMetadata {
                    page_id: Some("page-a".to_owned()),
                    region_id: Some("region-c".to_owned()),
                    ..Default::default()
                },
            ])
            .unwrap()
            .with_page_numbers([1, 1, 2])
            .unwrap()
            .with_units(
                "adaptive_v1",
                [
                    TranslationUnit {
                        id: "unit-page-0001-0000".to_owned(),
                        page_id: "page-z".to_owned(),
                        page_index: 1,
                        unit_type: "dialogue_group".to_owned(),
                        region_ids: vec!["region-a".to_owned(), "region-b".to_owned()],
                        segment_ids: vec![
                            "page:page-z::region:region-a".to_owned(),
                            "page:page-z::region:region-b".to_owned(),
                        ],
                    },
                    TranslationUnit {
                        id: "unit-page-0002-0000".to_owned(),
                        page_id: "page-a".to_owned(),
                        page_index: 2,
                        unit_type: "page".to_owned(),
                        region_ids: vec!["region-c".to_owned()],
                        segment_ids: vec!["page:page-a::region:region-c".to_owned()],
                    },
                ],
            );
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();

        assert!(input.get("segments").is_none());
        assert_eq!(input["unit_policy"], "adaptive_v1");
        assert_eq!(input["pages"][0]["page_id"], "page-z");
        assert_eq!(input["pages"][0]["page_index"], 1);
        assert_eq!(input["pages"][0]["units"][0]["id"], "unit-page-0001-0000");
        assert_eq!(
            input["pages"][0]["units"][0]["segments"][0]["id"],
            "page:page-z::region:region-a"
        );
        assert_eq!(
            input["pages"][0]["units"][0]["segments"][1]["id"],
            "page:page-z::region:region-b"
        );
        assert_eq!(input["pages"][1]["page_id"], "page-a");
        assert_eq!(input["pages"][1]["page_index"], 2);
        assert_eq!(
            input["pages"][1]["units"][0]["segments"][0]["id"],
            "page:page-a::region:region-c"
        );
    }

    #[test]
    fn chapter_system_prompt_requires_coherent_role_aware_translation() {
        let request = TranslationRequest::new(["dialogue", "sound"], Language::Russian)
            .with_source_language(Language::Japanese)
            .with_segment_metadata([
                TranslationSegmentMetadata {
                    role: Some("dev.koharu.text.dialogue".to_owned()),
                    inside_balloon: true,
                    ..Default::default()
                },
                TranslationSegmentMetadata {
                    role: Some("dev.koharu.text.onomatopoeia".to_owned()),
                    inside_balloon: false,
                    ..Default::default()
                },
            ])
            .unwrap()
            .with_page_numbers([1, 2])
            .unwrap();

        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("Chapter requirements:"));
        assert!(prompt.contains("one coherent chapter"));
        assert!(prompt.contains("manga reading sequence"));
        assert!(prompt.contains("Do not move text between pages"));
        assert!(prompt.contains("character names, terminology"));
    }

    #[test]
    fn default_system_prompt_uses_source_and_target_languages() {
        let request = TranslationRequest::new(["こんにちは"], Language::Russian)
            .with_source_language(Language::Japanese);
        let prompt = translation_system_prompt(&request);

        assert!(prompt.contains("from Japanese into natural Russian"));
        assert!(prompt.contains("Return only a JSON object"));
        assert!(prompt.contains("translated_text"));
        assert!(!prompt.contains("Additional instructions"));
    }

    #[test]
    fn system_prompt_encodes_invariants_and_custom_instructions() {
        let request = TranslationRequest::new(["hello"], Language::Korean)
            .with_source_language(Language::Japanese)
            .with_instructions("Use informal speech.");
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("from Japanese into natural Korean"));
        assert!(
            prompt.contains("Copy every input ID, page_id, region_id, and source text exactly")
        );
        assert!(prompt.contains("Use informal speech."));
    }

    #[test]
    fn schema_requires_the_complete_import_identity() {
        let schema = output_schema(3);
        let translations = &schema["properties"]["translations"];
        assert_eq!(translations["minItems"], 3);
        assert_eq!(translations["maxItems"], 3);
        assert_eq!(
            translations["items"]["required"],
            json!([
                "id",
                "page_id",
                "region_id",
                "source_text",
                "translated_text"
            ])
        );
        assert_eq!(translations["items"]["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn empty_custom_instructions_are_ignored() {
        let request = TranslationRequest::new(["hello"], Language::English).with_instructions("  ");
        assert!(!translation_system_prompt(&request).contains("Additional instructions"));
    }

    #[test]
    fn context_is_reference_only() {
        let request = TranslationRequest::new(["Where is she?"], Language::Japanese)
            .with_context([TranslationContext::new("I saw Alice.", "アリスを見た。")]);
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("dialogue continuity"));
        assert!(prompt.contains("Do not translate or return the context"));
    }

    #[test]
    fn image_context_does_not_expand_the_translation_scope() {
        let request = TranslationRequest::new(["text"], Language::English)
            .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("attached original page image"));
        assert!(prompt.contains("Translate only the supplied segments"));
    }

    #[test]
    fn ocr_prompt_and_schema_require_exact_transcription() {
        let request = OcrRequest {
            image: std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)),
            source_language: Some(Language::Japanese),
            instructions: Some("Read furigana before the base text.".to_owned()),
        };
        let (system, user) = ocr_prompts(&request);
        assert!(system.contains("Do not translate"));
        assert!(system.contains("Expected source language: Japanese"));
        assert!(system.contains("Read furigana before the base text."));
        assert!(user.contains("attached manga text crop"));
        let schema = ocr_output_schema();
        assert_eq!(schema["required"], json!(["text"]));
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn ocr_response_uses_the_first_strict_json_value() {
        assert_eq!(
            ocr_text("test", r#"{"text":"待って…"}"#).unwrap(),
            "待って…"
        );
        assert_eq!(
            ocr_text("test", "{\"text\":\"待って…\"}\nOCR complete.").unwrap(),
            "待って…"
        );
        assert!(ocr_text("test", r#"{"text":"hello","translation":"привет"}"#).is_err());
        assert!(ocr_text("test", "```json\n{\"text\":\"hello\"}\n```").is_err());
    }
}
