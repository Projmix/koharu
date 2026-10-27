use std::collections::BTreeMap;

use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    Error, Result,
    component::{Component, ValidationContext},
    id::validate_namespaced,
};

use super::{Origin, Point};

#[revisioned(revision = 2)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RasterLayerKind {
    Cleanup,
    Paint,
    /// An opaque replacement of the page source made by scan enhancement;
    /// later stages read it instead of the source while it is visible.
    #[revision(start = 2)]
    Enhancement,
}

/// A full-page pixel layer composited above the page source: transparent for
/// cleanup and paint, opaque for scan enhancement.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct RasterLayer {
    pub origin: Origin,
    pub name: String,
    pub kind: RasterLayerKind,
}

/// The user marked a page's cleanup as finished. Automatic cleaning leaves
/// the page as it is, so a reviewed local fill and the user's fixes to it are
/// kept and no cloud edit is paid for; manual removal and restoration still
/// work. User-owned: only the user sets or clears it.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct CleanupDone {
    pub origin: Origin,
}

impl Component for CleanupDone {
    const KIND: &'static str = "dev.koharu.page.cleanup-done";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
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

impl Component for RasterLayer {
    const KIND: &'static str = "dev.koharu.layer.raster";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        self.origin.validate()?;
        if self.name.is_empty() || self.name.len() > 4096 || self.name.contains('\0') {
            return Err(Error::invalid("raster layer name is invalid"));
        }
        Ok(())
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TextLayoutKind {
    Point,
    Paragraph,
}

#[revisioned(revision = 3)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct TextLayout {
    pub origin: Origin,
    pub kind: TextLayoutKind,
    /// Overrides the axes inferred from geometry when a frame is transformed.
    #[revision(start = 2)]
    pub angle_degrees: Option<f32>,
    /// Lettering printed on a surface seen at an angle (a newspaper, a phone
    /// screen, a board): the text is laid out in its frame as usual, then
    /// drawn mapped onto this quadrilateral. Its top-left, top-right,
    /// bottom-right and bottom-left corners are in units of the frame's width
    /// and height, from its top-left corner, so moving, resizing or turning
    /// the frame carries the distortion with it.
    #[revision(start = 3)]
    pub perspective: Option<Vec<Point>>,
}

/// The farthest a perspective corner may sit from its frame, in frame sizes.
const PERSPECTIVE_REACH: f64 = 8.0;

/// Whether `corners` are four finite corners of a convex quadrilateral turning
/// the same way as the frame, within reach of the frame.
#[must_use]
pub fn valid_perspective(corners: &[Point]) -> bool {
    let [a, b, c, d] = corners else {
        return false;
    };
    if corners.iter().any(|point| {
        !point.x.is_finite()
            || !point.y.is_finite()
            || point.x.abs() > PERSPECTIVE_REACH
            || point.y.abs() > PERSPECTIVE_REACH
    }) {
        return false;
    }
    let turn =
        |o: &Point, p: &Point, q: &Point| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    [turn(a, b, c), turn(b, c, d), turn(c, d, a), turn(d, a, b)]
        .into_iter()
        .all(|value| value > 1e-6)
}

impl Component for TextLayout {
    const KIND: &'static str = "dev.koharu.layer.text";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if self.angle_degrees.is_some_and(|angle| !angle.is_finite()) {
            return Err(Error::invalid("text layout angle must be finite"));
        }
        if self
            .perspective
            .as_ref()
            .is_some_and(|corners| !valid_perspective(corners))
        {
            return Err(Error::invalid(
                "text perspective must be four corners of a convex quadrilateral",
            ));
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum TextAlignment {
    Start,
    Center,
    End,
    Justify,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum WritingMode {
    Horizontal,
    Vertical,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

/// Where text may break across lines; `None` on the typography lets the
/// renderer hyphenate a word as a last resort.
#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LineBreaking {
    /// Wrap between words only, never inside one.
    WholeWords,
    /// Keep the author's lines: break only at line breaks in the text.
    Manual,
}

#[revisioned(revision = 2)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct Typography {
    pub origin: Origin,
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_style: Option<FontStyle>,
    pub size: Option<f32>,
    pub auto_fit: bool,
    pub color: Option<[u8; 4]>,
    pub stroke_color: Option<[u8; 4]>,
    pub stroke_width: Option<f32>,
    pub alignment: Option<TextAlignment>,
    pub writing_mode: Option<WritingMode>,
    pub extensions: BTreeMap<String, String>,
    #[revision(start = 2)]
    pub line_breaking: Option<LineBreaking>,
}

impl Component for Typography {
    const KIND: &'static str = "dev.koharu.text.typography";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if self
            .preferred_font
            .as_ref()
            .is_some_and(|font| font.len() > 4096)
            || self
                .font_weight
                .is_some_and(|weight| !(1..=1000).contains(&weight))
            || self
                .size
                .is_some_and(|size| !size.is_finite() || size <= 0.0)
            || (!self.auto_fit && self.size.is_none())
            || self
                .stroke_width
                .is_some_and(|width| !width.is_finite() || width < 0.0)
        {
            return Err(Error::invalid("typography intent is invalid"));
        }
        self.origin.validate()?;
        if self.extensions.len() > 1024
            || self.extensions.iter().any(|(key, value)| {
                validate_namespaced(key, "typography extension").is_err()
                    || value.len() > 64 * 1024
                    || value.contains('\0')
            })
        {
            return Err(Error::invalid("typography extensions are invalid"));
        }
        Ok(())
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}
