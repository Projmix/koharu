use serde::{Deserialize, Serialize};
use specta::Type;

use koharu_scene::EntityId;

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum OcrPass {
    Local,
    #[default]
    Final,
}

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
    Enhancement,
    Detection,
    Ocr,
    Markup,
    Translation,
    Inpainting,
}

impl Stage {
    pub const ALL: [Self; 6] = [
        Self::Enhancement,
        Self::Detection,
        Self::Ocr,
        Self::Markup,
        Self::Translation,
        Self::Inpainting,
    ];

    /// The stages whose results this stage reads. A stage cannot start on a
    /// page before these are done there or run earlier in the same request.
    /// Translation uses markup when it exists but does not need it.
    #[must_use]
    pub const fn requires(self) -> &'static [Self] {
        match self {
            // Enhancement is optional: detection reads the source without it.
            Self::Enhancement | Self::Detection => &[],
            Self::Ocr | Self::Inpainting => &[Self::Detection],
            Self::Markup | Self::Translation => &[Self::Ocr],
        }
    }
}

/// How many pages of a project a stage has nothing left to do on.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Type)]
pub struct StageProgress {
    pub stage: Stage,
    #[specta(type = f64)]
    pub done: usize,
    #[specta(type = f64)]
    pub total: usize,
    pub requires: Vec<Stage>,
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
