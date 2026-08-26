use serde::{Deserialize, Serialize};
use specta::Type;

use koharu_scene::EntityId;

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    Type,
    strum::Display,
    strum::EnumIter,
    strum::EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case", ascii_case_insensitive)]
pub enum Stage {
    Detection,
    Ocr,
    Translation,
    Inpainting,
}

impl Stage {
    pub const ALL: [Self; 4] = [
        Self::Detection,
        Self::Ocr,
        Self::Translation,
        Self::Inpainting,
    ];
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(tag = "target", content = "value", rename_all = "snake_case")]
pub enum StageTarget {
    Page(EntityId),
    Chapter,
}

impl StageTarget {
    #[must_use]
    pub const fn page(self) -> Option<EntityId> {
        match self {
            Self::Page(page) => Some(page),
            Self::Chapter => None,
        }
    }
}

impl std::fmt::Display for StageTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Page(page) => write!(formatter, "page {page}"),
            Self::Chapter => formatter.write_str("chapter"),
        }
    }
}
