//! Flags translations that read milder than a swearing or insulting source.
//!
//! Hosted LLMs tend to soften profanity even when told not to. A source line
//! with a profanity marker whose translation has none is a candidate. When a
//! hosted machine translation provider is configured, the same sources are
//! translated there: a candidate is flagged only when that reference keeps the
//! harshness, and the reference is stored next to the flag. Without a
//! reference every candidate is flagged. The translation itself is never
//! replaced; machine translation is usually worse at tone and context.

use koharu_scene::TranslationReviewReason;
use koharu_translator::{
    GenerationConfig, Language, ModelSelection, Provider, TranslationRequest, Translator,
};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Finding {
    pub(super) reason: TranslationReviewReason,
    pub(super) reference: Option<String>,
}

/// Review every `(source, translation)` pair, in order.
pub(super) async fn review(
    translator: &Translator,
    source_language: Language,
    target_language: Language,
    pairs: &[(&str, &str)],
) -> Vec<Option<Finding>> {
    let candidates = pairs
        .iter()
        .enumerate()
        .filter(|(_, (source, translation))| {
            softened(source_language, target_language, source, translation)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut findings = vec![None; pairs.len()];
    if candidates.is_empty() {
        return findings;
    }
    let references = reference_translations(
        translator,
        source_language,
        target_language,
        candidates.iter().map(|&index| pairs[index].0.to_owned()),
    )
    .await;
    for (position, index) in candidates.into_iter().enumerate() {
        let reference = references
            .as_ref()
            .map(|references| references[position].clone());
        // A reference that is no harsher than the translation means the source
        // marker was mild in context; flagging it would only add noise.
        if reference
            .as_deref()
            .is_some_and(|reference| !harsh(target_language, reference))
        {
            continue;
        }
        findings[index] = Some(Finding {
            reason: TranslationReviewReason::SoftenedProfanity,
            reference,
        });
    }
    findings
}

fn softened(
    source_language: Language,
    target_language: Language,
    source: &str,
    translation: &str,
) -> bool {
    !target_markers(target_language).is_empty()
        && source_markers(source_language)
            .iter()
            .any(|marker| source.to_lowercase().contains(marker))
        && !harsh(target_language, translation)
}

/// Whether `text` contains a target-language profanity or insult. Markers
/// match at the start of a word so that, for example, «процедура» does not
/// count as «дура».
fn harsh(language: Language, text: &str) -> bool {
    let text = text.to_lowercase();
    let markers = target_markers(language);
    text.char_indices()
        .filter(|&(index, _)| {
            text[..index]
                .chars()
                .next_back()
                .is_none_or(|previous| !previous.is_alphanumeric())
        })
        .any(|(index, _)| {
            markers
                .iter()
                .any(|marker| text[index..].starts_with(marker))
        })
}

/// Hosted machine translation of the candidate sources, or `None` when no
/// provider is configured or the request fails. Hosted machine translation
/// fails immediately (`AGENTS.md`), so a failure only drops the reference.
async fn reference_translations(
    translator: &Translator,
    source_language: Language,
    target_language: Language,
    sources: impl Iterator<Item = String>,
) -> Option<Vec<String>> {
    let provider = [Provider::DeepL, Provider::GoogleCloudTranslation]
        .into_iter()
        .find(|provider| {
            koharu_secrets::get(provider.into()).is_ok_and(|secret| secret.is_some())
        })?;
    let selection = ModelSelection {
        provider,
        model: None,
        quantization: None,
        vision: false,
        reasoning: false,
        reasoning_required: false,
    };
    let request =
        TranslationRequest::new(sources, target_language).with_source_language(source_language);
    match translator
        .translate(&selection, GenerationConfig::default(), request)
        .await
    {
        Ok((_, translations)) => Some(translations),
        Err(error) => {
            tracing::warn!(%provider, error = %format!("{error:#}"), "profanity reference translation failed");
            None
        }
    }
}

fn source_markers(language: Language) -> &'static [&'static str] {
    match language {
        Language::Japanese => &[
            "くそ",
            "クソ",
            "糞",
            "ちくしょう",
            "チクショウ",
            "畜生",
            "てめえ",
            "てめぇ",
            "テメエ",
            "テメェ",
            "死ね",
            "殺す",
            "ぶっ殺",
            "ぶっころ",
            "くたばれ",
            "ざけんな",
            "ふざけんな",
            "ふざけるな",
            "黙れ",
            "だまれ",
            "うるせえ",
            "うるせぇ",
            "バカ",
            "馬鹿",
            "アホ",
            "クズ",
            "野郎",
            "ブス",
            "ビッチ",
            "カス",
            "雑魚",
            "ザコ",
            "貴様",
            "きさま",
        ],
        Language::Korean => &[
            "씨발",
            "시발",
            "개새끼",
            "병신",
            "꺼져",
            "닥쳐",
            "죽어",
            "젠장",
            "새끼",
        ],
        Language::ChineseSimplified | Language::ChineseTraditional => &[
            "他妈",
            "他媽",
            "妈的",
            "媽的",
            "混蛋",
            "王八蛋",
            "去死",
            "傻逼",
            "该死",
            "該死",
            "混账",
            "混帳",
        ],
        Language::English => &[
            "fuck", "shit", "bitch", "bastard", "asshole", "damn", "cunt", "dick",
        ],
        _ => &[],
    }
}

fn target_markers(language: Language) -> &'static [&'static str] {
    match language {
        Language::Russian => &[
            "бля",
            "сука",
            "суки",
            "сучк",
            "сучар",
            "хуй",
            "хуё",
            "хуе",
            "хер",
            "пизд",
            "ебан",
            "ебат",
            "ебал",
            "ебу",
            "ёб",
            "заеб",
            "уеб",
            "выеб",
            "наху",
            "мудак",
            "мудил",
            "говн",
            "дерьм",
            "срань",
            "жоп",
            "черт",
            "чёрт",
            "проклят",
            "сволоч",
            "ублюд",
            "гад",
            "урод",
            "тварь",
            "мраз",
            "идиот",
            "дебил",
            "кретин",
            "придур",
            "дура",
            "дурак",
            "козёл",
            "козел",
            "засран",
            "заткн",
            "сдохн",
            "убью",
            "прибью",
            "катись",
            "отвали",
            "падл",
            "скотин",
            "отброс",
        ],
        Language::English => &[
            "fuck", "shit", "bitch", "bastard", "asshole", "damn", "dick", "cunt", "crap", "piss",
            "hell", "idiot", "moron", "scum", "jerk", "shut up", "die", "kill",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_markers_match_at_word_start_only() {
        assert!(harsh(Language::Russian, "Ах ты сука!"));
        assert!(harsh(Language::Russian, "Чёрт возьми"));
        assert!(!harsh(Language::Russian, "Это обычная процедура."));
        assert!(!harsh(Language::Russian, "Под небом голубым"));
    }

    #[test]
    fn a_softened_japanese_insult_is_a_candidate() {
        assert!(softened(
            Language::Japanese,
            Language::Russian,
            "てめえ、ぶっ殺すぞ！",
            "Эй, я тебя побью!"
        ));
        assert!(!softened(
            Language::Japanese,
            Language::Russian,
            "てめえ、ぶっ殺すぞ！",
            "Ах ты гад, убью!"
        ));
        assert!(!softened(
            Language::Japanese,
            Language::Russian,
            "おはようございます",
            "Доброе утро"
        ));
    }

    #[test]
    fn languages_without_target_markers_are_not_checked() {
        assert!(!softened(
            Language::Japanese,
            Language::German,
            "クソ！",
            "Mist!"
        ));
    }
}
