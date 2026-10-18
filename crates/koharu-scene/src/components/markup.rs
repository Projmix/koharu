use std::{collections::HashSet, fmt, str::FromStr};

use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    EntityId, Error, Result,
    component::{Component, ValidationContext},
};

use super::Origin;

const MAX_CHARACTER_ID_BYTES: usize = 64;
const MAX_CHARACTER_NAME_BYTES: usize = 256;
const MAX_CHARACTER_NOTES_BYTES: usize = 4096;
const MAX_CHARACTERS: usize = 1024;
const MAX_OBSERVED_TEXTS: usize = 100_000;
const MAX_OBSERVED_TEXT_BYTES: usize = 16 * 1024 * 1024;

/// A project-wide identifier of a character in the [`CharacterList`].
#[revisioned(revision = 1)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub struct CharacterId(String);

impl CharacterId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = Self(value.into());
        value.validate()?;
        Ok(value)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        let valid = !self.0.is_empty()
            && self.0.len() <= MAX_CHARACTER_ID_BYTES
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
        if valid {
            Ok(())
        } else {
            Err(Error::invalid(format!("invalid character id: {}", self.0)))
        }
    }
}

impl FromStr for CharacterId {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl fmt::Display for CharacterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// What a piece of text is in the story, independent from its detected role
/// and from how it is presented.
#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum SpeechKind {
    Speech,
    Thought,
    Narration,
    SoundEffect,
    Sign,
    Aside,
}

/// Scene markup of one text content entity: its kind and who says it.
///
/// `speaker` names a character of the project [`CharacterList`]. The list may
/// later drop that character, so readers resolve the id and treat an unknown
/// one as no speaker instead of the schema rejecting the list edit.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct SpeechMarkup {
    pub origin: Origin,
    pub kind: SpeechKind,
    pub speaker: Option<CharacterId>,
}

impl Component for SpeechMarkup {
    const KIND: &'static str = "dev.koharu.text.speech-markup";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if let Some(speaker) = &self.speaker {
            speaker.validate()?;
        }
        self.origin.validate()
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum PageMarkupMissing {
    /// The model declined the request.
    Refused,
    /// The provider's content filter stopped the answer.
    ContentFilter,
    /// The model answered with no content.
    Empty,
    /// The answer broke the markup contract and was discarded whole.
    Invalid,
    /// The request itself failed, for example on a network or provider error.
    Failed,
}

#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "status", content = "reason", rename_all = "kebab-case")]
pub enum PageMarkupStatus {
    Complete,
    Missing(PageMarkupMissing),
}

/// One source text the page markup saw.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub struct ObservedText {
    pub content: EntityId,
    pub source: String,
}

/// The outcome of scene markup for one page and the texts it was based on.
///
/// `observed` lets a rerun skip a page whose texts are unchanged. The ids are
/// observation keys, not references: a removed text only makes the record
/// stale, so they are not declared as record references.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct PageMarkup {
    pub origin: Origin,
    pub status: PageMarkupStatus,
    pub observed: Vec<ObservedText>,
}

impl Component for PageMarkup {
    const KIND: &'static str = "dev.koharu.page.markup";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        let mut contents = HashSet::with_capacity(self.observed.len());
        if self.observed.len() > MAX_OBSERVED_TEXTS
            || self.observed.iter().any(|text| {
                text.source.len() > MAX_OBSERVED_TEXT_BYTES
                    || text.source.contains('\0')
                    || !contents.insert(text.content)
            })
        {
            return Err(Error::invalid("page markup observations are invalid"));
        }
        self.origin.validate()
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "kebab-case")]
pub enum Gender {
    Male,
    Female,
    #[default]
    Unknown,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub struct Character {
    pub id: CharacterId,
    pub name: String,
    pub gender: Gender,
    pub notes: String,
}

/// The project's cast, shared by every page.
///
/// Ownership follows the scene authorship rules: scene markup creates and
/// extends a generated list, but once the user edits it the list is
/// user-owned and the pipeline can no longer change it. Markup then only
/// assigns speakers that are already in the list.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct CharacterList {
    pub origin: Origin,
    pub characters: Vec<Character>,
}

impl CharacterList {
    #[must_use]
    pub fn character(&self, id: &CharacterId) -> Option<&Character> {
        self.characters.iter().find(|character| &character.id == id)
    }
}

impl Component for CharacterList {
    const KIND: &'static str = "dev.koharu.project.characters";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if self.characters.len() > MAX_CHARACTERS {
            return Err(Error::invalid("character list is too long"));
        }
        let mut ids = HashSet::with_capacity(self.characters.len());
        let mut names = HashSet::with_capacity(self.characters.len());
        for character in &self.characters {
            character.id.validate()?;
            let name = character.name.trim();
            if name.is_empty()
                || character.name.len() > MAX_CHARACTER_NAME_BYTES
                || character.name.contains('\0')
                || character.notes.len() > MAX_CHARACTER_NOTES_BYTES
                || character.notes.contains('\0')
            {
                return Err(Error::invalid(format!(
                    "character {} has an invalid name or notes",
                    character.id
                )));
            }
            if !ids.insert(&character.id) || !names.insert(name) {
                return Err(Error::invalid(format!(
                    "character {} repeats an id or a name",
                    character.id
                )));
            }
        }
        self.origin.validate()
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}
