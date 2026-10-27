//! Lettering drawn onto a surface seen at an angle.
//!
//! A text layer with a perspective is laid out in its frame as any other
//! text, then every glyph outline is mapped by the projective transform that
//! takes the frame's rectangle onto the layer's quadrilateral. Glyph runs can
//! only carry an affine transform, so warped text is recorded as filled paths.

use koharu_rasterizer::{FillRule, PathElement, PreparedPath, PreparedSceneCommand};
use skrifa::outline::OutlinePen;
use vello::kurbo::{
    Affine, BezPath, Cap, Join, PathEl, Point, Stroke, StrokeOpts, flatten, stroke,
};

use crate::RenderBounds;

/// Curves are cut into lines this close to them before the mapping, which
/// does not keep Bézier curves.
const FLATTEN_TOLERANCE: f64 = 0.05;

/// The projective map from a `width` x `height` frame onto corners given in
/// units of that frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Perspective {
    width: f64,
    height: f64,
    /// Unit square to quadrilateral, after Heckbert's closed form:
    /// `x = (a u + b v + c) / (g u + h v + 1)`, `y = (d u + e v + f) / (...)`.
    coefficients: [f64; 8],
}

impl Perspective {
    pub(crate) fn new(width: f32, height: f32, corners: [(f32, f32); 4]) -> Self {
        let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] =
            corners.map(|(x, y)| (f64::from(x), f64::from(y)));
        let (dx1, dy1) = (x1 - x2, y1 - y2);
        let (dx2, dy2) = (x3 - x2, y3 - y2);
        let (dx3, dy3) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
        let denominator = dx1 * dy2 - dx2 * dy1;
        let (g, h) = if denominator.abs() <= f64::EPSILON {
            (0.0, 0.0)
        } else {
            (
                (dx3 * dy2 - dx2 * dy3) / denominator,
                (dx1 * dy3 - dx3 * dy1) / denominator,
            )
        };
        Self {
            width: f64::from(width),
            height: f64::from(height),
            coefficients: [
                x1 - x0 + g * x1,
                x3 - x0 + h * x3,
                x0,
                y1 - y0 + g * y1,
                y3 - y0 + h * y3,
                y0,
                g,
                h,
            ],
        }
    }

    pub(crate) fn map(&self, point: Point) -> Point {
        let [a, b, c, d, e, f, g, h] = self.coefficients;
        let (u, v) = (point.x / self.width, point.y / self.height);
        let w = g * u + h * v + 1.0;
        Point::new(
            (a * u + b * v + c) / w * self.width,
            (d * u + e * v + f) / w * self.height,
        )
    }

    /// The bounds of `bounds` once mapped; a convex quadrilateral's corners
    /// hold its whole image.
    pub(crate) fn map_bounds(&self, bounds: RenderBounds) -> RenderBounds {
        let corners = [
            (bounds.x, bounds.y),
            (bounds.x + bounds.width, bounds.y),
            (bounds.x + bounds.width, bounds.y + bounds.height),
            (bounds.x, bounds.y + bounds.height),
        ]
        .map(|(x, y)| self.map(Point::new(f64::from(x), f64::from(y))));
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for point in corners {
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
        }
        RenderBounds {
            x: min_x as f32,
            y: min_y as f32,
            width: (max_x - min_x) as f32,
            height: (max_y - min_y) as f32,
        }
    }

    /// One filled path of `path` in frame coordinates, mapped.
    fn fill(&self, path: &BezPath, color: [u8; 4]) -> Option<PreparedSceneCommand> {
        let mut elements = Vec::new();
        flatten(path, FLATTEN_TOLERANCE, |element| match element {
            PathEl::MoveTo(point) => {
                let point = self.map(point);
                elements.push(PathElement::MoveTo([point.x, point.y]));
            }
            PathEl::LineTo(point) => {
                let point = self.map(point);
                elements.push(PathElement::LineTo([point.x, point.y]));
            }
            PathEl::ClosePath => elements.push(PathElement::Close),
            // `flatten` emits lines only.
            PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
        });
        (!elements.is_empty()).then(|| {
            PreparedSceneCommand::FillPath(PreparedPath {
                fill: FillRule::NonZero,
                transform: Affine::IDENTITY.as_coeffs(),
                color,
                elements,
            })
        })
    }

    /// One glyph outline, `transform`ed into frame coordinates, as filled
    /// paths: grown outward by `dilation` (an outline drawn around the glyph,
    /// as `embolden` grows a glyph run) and then mapped.
    pub(crate) fn glyph(
        &self,
        commands: &mut Vec<PreparedSceneCommand>,
        outline: &BezPath,
        transform: Affine,
        dilation: f64,
        color: [u8; 4],
    ) {
        let path = transform * outline.clone();
        if dilation > 0.0 {
            // One stroke per contour: the bands of a glyph's outer and inner
            // contours wind opposite ways and would cancel in one fill.
            let style = Stroke::new(dilation * 2.0)
                .with_join(Join::Round)
                .with_caps(Cap::Round);
            for contour in contours(&path) {
                let band = stroke(&contour, &style, &StrokeOpts::default(), FLATTEN_TOLERANCE);
                commands.extend(self.fill(&band, color));
            }
        }
        commands.extend(self.fill(&path, color));
    }
}

fn contours(path: &BezPath) -> Vec<BezPath> {
    let mut contours = Vec::<BezPath>::new();
    for element in path.elements() {
        if matches!(element, PathEl::MoveTo(_)) || contours.is_empty() {
            contours.push(BezPath::new());
        }
        contours
            .last_mut()
            .expect("a contour was started")
            .push(*element);
    }
    contours
}

/// Records a glyph outline, in font units scaled to the drawn size, as a path.
pub(crate) struct OutlinePath<'a>(pub(crate) &'a mut BezPath);

impl OutlinePen for OutlinePath<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((f64::from(x), f64::from(y)));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((f64::from(x), f64::from(y)));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0.quad_to(
            (f64::from(cx0), f64::from(cy0)),
            (f64::from(x), f64::from(y)),
        );
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            (f64::from(cx0), f64::from(cy0)),
            (f64::from(cx1), f64::from(cy1)),
            (f64::from(x), f64::from(y)),
        );
    }

    fn close(&mut self) {
        self.0.close_path();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: Point, right: (f64, f64)) -> bool {
        (left.x - right.0).abs() < 1e-9 && (left.y - right.1).abs() < 1e-9
    }

    #[test]
    fn the_frame_corners_land_on_the_quadrilateral() {
        let corners = [(0.1, 0.0), (0.9, 0.2), (1.0, 0.8), (0.0, 1.0)];
        let perspective = Perspective::new(200.0, 100.0, corners);
        for ((x, y), (u, v)) in [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
            .into_iter()
            .zip(corners)
        {
            let mapped = perspective.map(Point::new(x, y));
            assert!(
                close(mapped, (f64::from(u) * 200.0, f64::from(v) * 100.0)),
                "{mapped:?}"
            );
        }
    }

    #[test]
    fn the_unit_square_is_the_identity() {
        let perspective =
            Perspective::new(30.0, 40.0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert!(close(perspective.map(Point::new(12.5, 7.0)), (12.5, 7.0)));
    }

    #[test]
    fn a_receding_surface_draws_the_far_half_smaller() {
        // The top edge is half as wide: the top half of the text shrinks more.
        let perspective = Perspective::new(
            100.0,
            100.0,
            [(0.25, 0.0), (0.75, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let middle = perspective.map(Point::new(50.0, 50.0));
        assert!((middle.x - 50.0).abs() < 1e-9);
        assert!(middle.y < 50.0, "{middle:?}");
    }
}
