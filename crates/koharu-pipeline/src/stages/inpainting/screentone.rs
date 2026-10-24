//! Rescreening: an inpainting model continues lines and the overall tone of a
//! page well, but turns a regular screentone into grey mush with dots off the
//! screen's grid. Around each masked area the screen is measured from the
//! known pixels (its lattice, phase, paper and ink) and its dots are drawn
//! again inside the area at the tone the model predicted, so they continue
//! the surrounding dots pixel for pixel and follow a gradient. The pattern is
//! first drawn over the known ring and compared with it; where it does not
//! match (no screen, another screen, artwork), the model's pixels stay.

use image::{GrayImage, Rgb, RgbImage};

use super::mask_components;

/// Largest lattice lag measured, in pixels: screens up to 16 px apart.
const MAX_LAG: i32 = 16;
/// Autocorrelation a lattice peak needs.
const PEAK_MIN: f64 = 0.3;
/// Ring of known pixels measured around each masked area: at least 32 px.
const RING_DIVISOR: u32 = 40;
const RING_MIN: u32 = 32;
/// The known ring must be grey (mean channel spread at most this).
const GREY_MAX_SPREAD: f64 = 12.0;
/// Paper and ink must differ by at least this for dots to be drawn.
const MIN_CONTRAST: f64 = 60.0;
/// Correlation between the pattern drawn over the ring and the ring itself
/// that a component needs, and that each 16 px tile needs to be rescreened.
const MATCH_MIN: f64 = 0.5;
const TILE_MATCH_MIN: f64 = 0.4;
const TILE: u32 = 16;
/// Share of the contrast within which a model pixel counts as plain paper or
/// solid ink.
const SOLID_MARGIN: f64 = 0.15;
/// Spread of a tile's dots, as a share of the paper-ink contrast, below which
/// it holds no screen.
const DOT_MIN_DETAIL: f64 = 0.1;
/// Subsamples per pixel side: a binary screen seen at the page's resolution.
const SUPERSAMPLE: u32 = 4;

/// Redraws the screentone over the pixels of `mask` in `output`, measured
/// from `source` around each masked area. Returns the pixels redrawn.
pub(super) fn rescreen(source: &RgbImage, output: &mut RgbImage, mask: &GrayImage) -> usize {
    let (width, height) = source.dimensions();
    let ring = (width.max(height) / RING_DIVISOR).max(RING_MIN);
    let known = |x: u32, y: u32| mask.get_pixel(x, y)[0] < 127;
    let (labels, components) = mask_components(mask);
    let mut redrawn = 0;
    let output_luma = luma(output);
    for component in components {
        let [left, top, right, bottom] = component.bounds;
        let crop = [
            left.saturating_sub(ring),
            top.saturating_sub(ring),
            (right + ring).min(width),
            (bottom + ring).min(height),
        ];
        let Some(screen) = Screen::measure(source, &known, crop) else {
            continue;
        };
        let tone = Tone::new(&output_luma, width, screen.cell_radius());
        let Some(tiles) = screen.tile_matches(source, &known, crop, &tone) else {
            continue;
        };
        let mut draws = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if labels.get_pixel(x, y)[0] != component.label {
                    continue;
                }
                if !tiles.accepts(x, y) {
                    continue;
                }
                // Where the model left plain paper or solid ink it is right,
                // also at the edge of a screened area; only its grey mush is
                // redrawn.
                let margin = SOLID_MARGIN * screen.contrast();
                let around = (y.saturating_sub(1)..(y + 2).min(height))
                    .flat_map(|sy| {
                        (x.saturating_sub(1)..(x + 2).min(width)).map(move |sx| (sx, sy))
                    })
                    .map(|(sx, sy)| f64::from(output_luma[(sy * width + sx) as usize]))
                    .collect::<Vec<_>>();
                if around.iter().all(|value| *value >= screen.paper - margin)
                    || around.iter().all(|value| *value <= screen.ink + margin)
                {
                    continue;
                }
                let model = f64::from(output_luma[(y * width + x) as usize]);
                let drawn = screen.render(x, y, tone.darkness(x, y, &screen));
                // A dark stroke the model continued through the area is a line,
                // not mush: keep it over the dots.
                let value = if model < screen.ink + 0.3 * screen.contrast() {
                    drawn.min(model)
                } else {
                    drawn
                };
                draws.push((x, y, value.round().clamp(0.0, 255.0) as u8));
            }
        }
        redrawn += draws.len();
        for (x, y, value) in draws {
            output.put_pixel(x, y, Rgb([value; 3]));
        }
    }
    redrawn
}

fn luma(image: &RgbImage) -> Vec<f32> {
    image
        .pixels()
        .map(|pixel| {
            0.299 * f32::from(pixel[0]) + 0.587 * f32::from(pixel[1]) + 0.114 * f32::from(pixel[2])
        })
        .collect()
}

/// A regular screen: dots on the lattice `origin + i * a + j * b`.
#[derive(Clone, Debug)]
struct Screen {
    a: (f64, f64),
    b: (f64, f64),
    /// Lattice coordinates of a pixel centre: `inverse * (x, y) - phase`.
    inverse: [[f64; 2]; 2],
    phase: (f64, f64),
    paper: f64,
    ink: f64,
    /// Spot-function thresholds for coverage 0..=1, sorted descending.
    thresholds: Vec<f64>,
}

impl Screen {
    fn measure(
        source: &RgbImage,
        known: &dyn Fn(u32, u32) -> bool,
        crop: [u32; 4],
    ) -> Option<Self> {
        let [left, top, right, bottom] = crop;
        let (crop_width, crop_height) = (right - left, bottom - top);
        let mut values = Vec::new();
        let mut spread = 0.0;
        let mut plane = vec![f64::NAN; (crop_width * crop_height) as usize];
        for y in top..bottom {
            for x in left..right {
                if !known(x, y) {
                    continue;
                }
                let pixel = source.get_pixel(x, y);
                let value = 0.299 * f64::from(pixel[0])
                    + 0.587 * f64::from(pixel[1])
                    + 0.114 * f64::from(pixel[2]);
                spread += f64::from(pixel.0.iter().max()? - pixel.0.iter().min()?);
                plane[((y - top) * crop_width + (x - left)) as usize] = value;
                values.push(value);
            }
        }
        if values.len() < 400 || spread / values.len() as f64 > GREY_MAX_SPREAD {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let paper = values[values.len() * 95 / 100];
        let ink = values[values.len() * 5 / 100];
        if paper - ink < MIN_CONTRAST {
            return None;
        }
        let detail = high_pass(&plane, crop_width, crop_height, 8);
        let correlation = autocorrelation(&detail, crop_width, crop_height);
        let at = |dx: i32, dy: i32| correlation(dx, dy);
        let peaks = peaks(&at);
        let a = refine(&at, *peaks.first()?)?;
        let b = peaks
            .iter()
            .filter_map(|&peak| {
                let vector = refine(&at, peak)?;
                let cosine =
                    (a.0 * vector.0 + a.1 * vector.1) / (a.0.hypot(a.1) * vector.0.hypot(vector.1));
                (cosine.abs() < 0.866).then_some(vector)
            })
            .next()?;
        // Periodic, not just textured: the diagonal of the cell repeats too.
        if at((a.0 + b.0).round() as i32, (a.1 + b.1).round() as i32) < 0.2 {
            return None;
        }
        let determinant = a.0 * b.1 - a.1 * b.0;
        if determinant.abs() < 2.0 {
            return None;
        }
        let inverse = [
            [b.1 / determinant, -b.0 / determinant],
            [-a.1 / determinant, a.0 / determinant],
        ];
        // Where the dots sit: the phase of darkness along both lattice axes.
        let (mut first, mut second) = ((0.0, 0.0), (0.0, 0.0));
        for y in 0..crop_height {
            for x in 0..crop_width {
                let darkness = -detail[(y * crop_width + x) as usize];
                if darkness.is_nan() {
                    continue;
                }
                let (px, py) = (f64::from(left + x) + 0.5, f64::from(top + y) + 0.5);
                let u = inverse[0][0] * px + inverse[0][1] * py;
                let v = inverse[1][0] * px + inverse[1][1] * py;
                let (su, cu) = (std::f64::consts::TAU * u).sin_cos();
                let (sv, cv) = (std::f64::consts::TAU * v).sin_cos();
                first = (first.0 + darkness * cu, first.1 + darkness * su);
                second = (second.0 + darkness * cv, second.1 + darkness * sv);
            }
        }
        let phase = (
            first.1.atan2(first.0) / std::f64::consts::TAU,
            second.1.atan2(second.0) / std::f64::consts::TAU,
        );
        let mut thresholds = Vec::with_capacity(64 * 64);
        for j in 0..64 {
            for i in 0..64 {
                thresholds.push(spot(
                    (f64::from(i) + 0.5) / 64.0,
                    (f64::from(j) + 0.5) / 64.0,
                ));
            }
        }
        thresholds.sort_by(|left, right| right.total_cmp(left));
        Some(Self {
            a,
            b,
            inverse,
            phase,
            paper,
            ink,
            thresholds,
        })
    }

    fn contrast(&self) -> f64 {
        self.paper - self.ink
    }

    /// Half the longer lattice vector, rounded up: the radius over which a
    /// mean sees whole cells.
    fn cell_radius(&self) -> u32 {
        (self.a.0.hypot(self.a.1).max(self.b.0.hypot(self.b.1))).ceil() as u32
    }

    /// The spot threshold that darkens `coverage` of a cell.
    fn threshold(&self, coverage: f64) -> f64 {
        if coverage <= 0.0 {
            return f64::INFINITY;
        }
        let index = (coverage * self.thresholds.len() as f64).ceil() as usize;
        self.thresholds[index.clamp(1, self.thresholds.len()) - 1]
    }

    /// The pixel at (x, y) of a screen darkened to `coverage`, averaged over
    /// subsamples.
    fn render(&self, x: u32, y: u32, coverage: f64) -> f64 {
        if coverage >= 1.0 {
            return self.ink;
        }
        let threshold = self.threshold(coverage);
        let mut dark = 0;
        for sy in 0..SUPERSAMPLE {
            for sx in 0..SUPERSAMPLE {
                let px = f64::from(x) + (f64::from(sx) + 0.5) / f64::from(SUPERSAMPLE);
                let py = f64::from(y) + (f64::from(sy) + 0.5) / f64::from(SUPERSAMPLE);
                let u = self.inverse[0][0] * px + self.inverse[0][1] * py - self.phase.0;
                let v = self.inverse[1][0] * px + self.inverse[1][1] * py - self.phase.1;
                dark += u32::from(spot(u, v) >= threshold);
            }
        }
        let share = f64::from(dark) / f64::from(SUPERSAMPLE * SUPERSAMPLE);
        self.paper - share * self.contrast()
    }

    /// How well the screen drawn over the known ring matches it, overall and
    /// per tile. `None` when the overall match is too weak.
    fn tile_matches(
        &self,
        source: &RgbImage,
        known: &dyn Fn(u32, u32) -> bool,
        crop: [u32; 4],
        tone: &Tone,
    ) -> Option<Tiles> {
        let [left, top, right, bottom] = crop;
        let columns = (right - left).div_ceil(TILE);
        let rows = (bottom - top).div_ceil(TILE);
        let mut sums = vec![[0.0f64; 5]; (columns * rows) as usize];
        let mut total = [0.0f64; 5];
        for y in top..bottom {
            for x in left..right {
                if !known(x, y) {
                    continue;
                }
                let pixel = source.get_pixel(x, y);
                let actual = 0.299 * f64::from(pixel[0])
                    + 0.587 * f64::from(pixel[1])
                    + 0.114 * f64::from(pixel[2]);
                // Dots only: both sides minus the local tone, so a match on
                // where black and white areas lie does not count.
                let darkness = tone.darkness(x, y, self);
                let level = self.paper - darkness * self.contrast();
                let (actual, drawn) = (actual - level, self.render(x, y, darkness) - level);
                let tile = ((y - top) / TILE * columns + (x - left) / TILE) as usize;
                for sums in [&mut sums[tile], &mut total] {
                    sums[0] += actual;
                    sums[1] += drawn;
                    sums[2] += actual * actual;
                    sums[3] += drawn * drawn;
                    sums[4] += actual * drawn;
                }
            }
        }
        let correlation = |sums: &[f64; 5], count: f64| {
            let covariance = sums[4] / count - sums[0] * sums[1] / (count * count);
            let actual = sums[2] / count - (sums[0] / count).powi(2);
            let drawn = sums[3] / count - (sums[1] / count).powi(2);
            covariance / (actual * drawn).sqrt().max(1e-9)
        };
        // A tile counts only where it has dots to match: flat black or white
        // areas match anything.
        let dotted = |sums: &[f64; 5], count: f64| {
            (sums[2] / count - (sums[0] / count).powi(2)).sqrt() >= DOT_MIN_DETAIL * self.contrast()
        };
        let mut counts = vec![0.0f64; sums.len()];
        let mut total_count = 0.0;
        for y in top..bottom {
            for x in left..right {
                if known(x, y) {
                    counts[((y - top) / TILE * columns + (x - left) / TILE) as usize] += 1.0;
                    total_count += 1.0;
                }
            }
        }
        if correlation(&total, total_count) < MATCH_MIN {
            return None;
        }
        let centres = (0..rows)
            .flat_map(|row| (0..columns).map(move |column| (column, row)))
            .filter_map(|(column, row)| {
                let tile = (row * columns + column) as usize;
                (counts[tile] >= 40.0).then(|| {
                    (
                        f64::from(left + column * TILE + TILE / 2),
                        f64::from(top + row * TILE + TILE / 2),
                        dotted(&sums[tile], counts[tile])
                            && correlation(&sums[tile], counts[tile]) >= TILE_MATCH_MIN,
                    )
                })
            })
            .collect();
        Some(Tiles { centres })
    }
}

/// Tiles of the known ring with whether the screen matched there; a masked
/// pixel follows the nearest one.
struct Tiles {
    centres: Vec<(f64, f64, bool)>,
}

impl Tiles {
    fn accepts(&self, x: u32, y: u32) -> bool {
        let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
        self.centres
            .iter()
            .min_by(|left, right| {
                (left.0 - px)
                    .hypot(left.1 - py)
                    .total_cmp(&(right.0 - px).hypot(right.1 - py))
            })
            .is_some_and(|centre| centre.2)
    }
}

/// Local mean brightness over whole screen cells, from an integral image.
struct Tone {
    integral: Vec<f64>,
    width: u32,
    height: u32,
    radius: u32,
}

impl Tone {
    fn new(luma: &[f32], width: u32, radius: u32) -> Self {
        let height = luma.len() as u32 / width;
        let mut integral = vec![0.0; ((width + 1) * (height + 1)) as usize];
        for y in 0..height {
            let mut row = 0.0;
            for x in 0..width {
                row += f64::from(luma[(y * width + x) as usize]);
                integral[((y + 1) * (width + 1) + x + 1) as usize] =
                    integral[(y * (width + 1) + x + 1) as usize] + row;
            }
        }
        Self {
            integral,
            width,
            height,
            radius: radius.max(1),
        }
    }

    /// The dot coverage at (x, y): how far its local mean sits from paper
    /// towards ink.
    fn darkness(&self, x: u32, y: u32, screen: &Screen) -> f64 {
        let (left, top) = (x.saturating_sub(self.radius), y.saturating_sub(self.radius));
        let (right, bottom) = (
            (x + self.radius + 1).min(self.width),
            (y + self.radius + 1).min(self.height),
        );
        let at = |x: u32, y: u32| self.integral[(y * (self.width + 1) + x) as usize];
        let sum = at(right, bottom) - at(left, bottom) - at(right, top) + at(left, top);
        let mean = sum / f64::from((right - left) * (bottom - top));
        ((screen.paper - mean) / screen.contrast()).clamp(0.0, 1.0)
    }
}

/// A round-dot spot function: 1 at a dot centre, -1 between dots.
fn spot(u: f64, v: f64) -> f64 {
    ((std::f64::consts::TAU * u).cos() + (std::f64::consts::TAU * v).cos()) / 2.0
}

/// Each known value minus the mean of the known values within `radius`;
/// NaN stays NaN.
fn high_pass(plane: &[f64], width: u32, height: u32, radius: u32) -> Vec<f64> {
    let stride = (width + 1) as usize;
    let mut sums = vec![0.0; stride * (height + 1) as usize];
    let mut counts = vec![0.0; stride * (height + 1) as usize];
    for y in 0..height as usize {
        let (mut row, mut row_count) = (0.0, 0.0);
        for x in 0..width as usize {
            let value = plane[y * width as usize + x];
            if !value.is_nan() {
                row += value;
                row_count += 1.0;
            }
            sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + row;
            counts[(y + 1) * stride + x + 1] = counts[y * stride + x + 1] + row_count;
        }
    }
    let box_sum = |table: &[f64], left: usize, top: usize, right: usize, bottom: usize| {
        table[bottom * stride + right] - table[top * stride + right] - table[bottom * stride + left]
            + table[top * stride + left]
    };
    let radius = radius as usize;
    let mut detail = vec![f64::NAN; plane.len()];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let value = plane[y * width as usize + x];
            if value.is_nan() {
                continue;
            }
            let (left, top) = (x.saturating_sub(radius), y.saturating_sub(radius));
            let (right, bottom) = (
                (x + radius + 1).min(width as usize),
                (y + radius + 1).min(height as usize),
            );
            let count = box_sum(&counts, left, top, right, bottom);
            detail[y * width as usize + x] =
                value - box_sum(&sums, left, top, right, bottom) / count;
        }
    }
    detail
}

/// Normalised autocorrelation of the known detail at lags up to `MAX_LAG`.
fn autocorrelation(detail: &[f64], width: u32, height: u32) -> impl Fn(i32, i32) -> f64 {
    let side = (2 * MAX_LAG + 1) as usize;
    let mut table = vec![0.0; side * side];
    let (width, height) = (width as i32, height as i32);
    let value = |x: i32, y: i32| detail[(y * width + x) as usize];
    for dy in 0..=MAX_LAG {
        for dx in -MAX_LAG..=MAX_LAG {
            if dy == 0 && dx < 0 {
                continue;
            }
            let (mut product, mut first, mut second) = (0.0, 0.0, 0.0);
            for y in 0..height - dy {
                for x in (-dx).max(0)..(width - dx.max(0)) {
                    let (a, b) = (value(x, y), value(x + dx, y + dy));
                    if a.is_nan() || b.is_nan() {
                        continue;
                    }
                    product += a * b;
                    first += a * a;
                    second += b * b;
                }
            }
            let correlation = product / (first * second).sqrt().max(1e-9);
            table[((dy + MAX_LAG) as usize) * side + (dx + MAX_LAG) as usize] = correlation;
            table[((MAX_LAG - dy) as usize) * side + (MAX_LAG - dx) as usize] = correlation;
        }
    }
    move |dx: i32, dy: i32| {
        if dx.abs() > MAX_LAG || dy.abs() > MAX_LAG {
            return 0.0;
        }
        table[((dy + MAX_LAG) as usize) * side + (dx + MAX_LAG) as usize]
    }
}

/// Local maxima of the autocorrelation above `PEAK_MIN`, nearest first.
fn peaks(at: &dyn Fn(i32, i32) -> f64) -> Vec<(i32, i32)> {
    let mut peaks = Vec::new();
    for dy in 0..MAX_LAG {
        for dx in -MAX_LAG + 1..MAX_LAG {
            if (dx, dy) == (0, 0) || dy == 0 && dx < 0 {
                continue;
            }
            let value = at(dx, dy);
            if value < PEAK_MIN {
                continue;
            }
            let highest = (-1..=1)
                .all(|oy| (-1..=1).all(|ox| (ox, oy) == (0, 0) || at(dx + ox, dy + oy) <= value));
            if highest {
                peaks.push((dx, dy));
            }
        }
    }
    peaks.sort_by_key(|&(dx, dy)| dx * dx + dy * dy);
    peaks
}

/// The lattice vector behind a peak, to a fraction of a pixel: parabolic
/// peaks at its multiples up to `MAX_LAG`, fitted by least squares.
fn refine(at: &dyn Fn(i32, i32) -> f64, peak: (i32, i32)) -> Option<(f64, f64)> {
    let sub = |(x, y): (i32, i32)| {
        let offset = |minus: f64, centre: f64, plus: f64| {
            let curvature = minus - 2.0 * centre + plus;
            if curvature >= 0.0 {
                0.0
            } else {
                (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5)
            }
        };
        (
            f64::from(x) + offset(at(x - 1, y), at(x, y), at(x + 1, y)),
            f64::from(y) + offset(at(x, y - 1), at(x, y), at(x, y + 1)),
        )
    };
    let mut vector = sub(peak);
    let (mut weighted, mut weights) = ((vector.0, vector.1), 1.0);
    for multiple in 2..=MAX_LAG {
        let guess = (
            vector.0 * f64::from(multiple),
            vector.1 * f64::from(multiple),
        );
        if guess.0.abs() > f64::from(MAX_LAG - 1) || guess.1.abs() > f64::from(MAX_LAG - 1) {
            break;
        }
        let rounded = (guess.0.round() as i32, guess.1.round() as i32);
        let best = (-1..=1)
            .flat_map(|oy| (-1..=1).map(move |ox| (rounded.0 + ox, rounded.1 + oy)))
            .max_by(|left, right| at(left.0, left.1).total_cmp(&at(right.0, right.1)))?;
        if at(best.0, best.1) < PEAK_MIN {
            break;
        }
        let found = sub(best);
        let weight = f64::from(multiple * multiple);
        weighted.0 += found.0 * f64::from(multiple);
        weighted.1 += found.1 * f64::from(multiple);
        weights += weight;
        vector = (weighted.0 / weights, weighted.1 / weights);
    }
    Some(vector)
}

#[cfg(test)]
mod tests {
    use image::Luma;

    use super::*;

    /// A 45-degree dot screen 3.5 px apart with a tone gradient, as a binary
    /// screen averaged down by 4.
    fn screened(width: u32, height: u32) -> RgbImage {
        let screen = Screen {
            a: (2.475, 2.475),
            b: (-2.475, 2.475),
            inverse: {
                let determinant = 2.475 * 2.475 * 2.0;
                [
                    [2.475 / determinant, 2.475 / determinant],
                    [-2.475 / determinant, 2.475 / determinant],
                ]
            },
            phase: (0.3, 0.1),
            paper: 250.0,
            ink: 20.0,
            thresholds: {
                let mut thresholds = (0..64 * 64)
                    .map(|index| {
                        spot(
                            (f64::from(index % 64) + 0.5) / 64.0,
                            (f64::from(index / 64) + 0.5) / 64.0,
                        )
                    })
                    .collect::<Vec<_>>();
                thresholds.sort_by(|left, right| right.total_cmp(left));
                thresholds
            },
        };
        RgbImage::from_fn(width, height, |x, y| {
            let coverage = 0.2 + 0.4 * f64::from(x) / f64::from(width);
            Rgb([screen.render(x, y, coverage).round() as u8; 3])
        })
    }

    #[test]
    fn a_hole_in_a_screen_gets_the_surrounding_dots_back() {
        let original = screened(160, 160);
        let mut mask = GrayImage::new(160, 160);
        // The model's guess: the right mean tone, no dots.
        let mut output = original.clone();
        for y in 60..100 {
            for x in 60..100 {
                mask.put_pixel(x, y, Luma([255]));
                let mean = (0..5)
                    .flat_map(|oy| (0..5).map(move |ox| (x - 2 + ox, y - 2 + oy)))
                    .map(|(sx, sy)| f64::from(original.get_pixel(sx, sy)[0]))
                    .sum::<f64>()
                    / 25.0;
                output.put_pixel(x, y, Rgb([mean.round() as u8; 3]));
            }
        }
        let error = |image: &RgbImage| {
            (60..100)
                .flat_map(|y| (60..100).map(move |x| (x, y)))
                .map(|(x, y)| {
                    f64::from(image.get_pixel(x, y)[0].abs_diff(original.get_pixel(x, y)[0]))
                })
                .sum::<f64>()
                / 1600.0
        };
        let before = error(&output);
        assert_eq!(rescreen(&original, &mut output, &mask), 1600);
        let after = error(&output);
        assert!(after < before * 0.5, "mean error {before:.1} -> {after:.1}");
    }

    #[test]
    fn flat_or_textured_surroundings_are_left_to_the_model() {
        let flat = RgbImage::from_pixel(120, 120, Rgb([240, 240, 240]));
        let mut mask = GrayImage::new(120, 120);
        for y in 50..70 {
            for x in 50..70 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let mut output = flat.clone();
        assert_eq!(rescreen(&flat, &mut output, &mask), 0);
        let mut seed = 7u32;
        let noise = RgbImage::from_fn(120, 120, |_, _| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            Rgb([if seed >> 16 & 1 == 0 { 20 } else { 240 }; 3])
        });
        let mut output = noise.clone();
        assert_eq!(rescreen(&noise, &mut output, &mask), 0);
    }
}
