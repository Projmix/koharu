use koharu_scene::SpeechKind;
use serde::{Deserialize, Serialize};
use specta::Type;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Type)]
#[serde(default)]
pub struct TypesettingConfig {
    pub font_families: Vec<String>,
    /// Font used ahead of `font_families` for text of one scene markup kind.
    /// A layer's own preferred font still wins.
    pub kind_fonts: KindFonts,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize, Type)]
#[serde(default)]
pub struct KindFonts {
    pub speech: Option<String>,
    pub thought: Option<String>,
    pub narration: Option<String>,
    pub sound_effect: Option<String>,
    pub sign: Option<String>,
    pub aside: Option<String>,
}

impl KindFonts {
    pub fn get(&self, kind: SpeechKind) -> Option<&str> {
        match kind {
            SpeechKind::Speech => &self.speech,
            SpeechKind::Thought => &self.thought,
            SpeechKind::Narration => &self.narration,
            SpeechKind::SoundEffect => &self.sound_effect,
            SpeechKind::Sign => &self.sign,
            SpeechKind::Aside => &self.aside,
        }
        .as_deref()
        .map(str::trim)
        .filter(|family| !family.is_empty())
    }
}

impl Default for TypesettingConfig {
    fn default() -> Self {
        Self {
            font_families: vec!["CCWildWords".to_owned(), "Adobe 黑体 Std".to_owned()],
            kind_fonts: KindFonts::default(),
        }
    }
}

impl TypesettingConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("typesetting")
    }
}
