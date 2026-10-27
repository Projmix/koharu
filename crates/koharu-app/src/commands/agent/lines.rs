//! The slant of printed lettering, measured on the page's pixels.
//!
//! Lettering printed on a newspaper, a screen or a board follows that
//! surface, and a translation laid over it must take the same slant. The
//! detector's boxes are upright, so the slant is read from the ink itself:
//! lines of horizontal text and columns of vertical text both make the ink's
//! projection across them sharpest when the projection is taken along them.

use image::RgbaImage;

/// Samples taken across the longer side of the measured area.
const SAMPLES: f64 = 400.0;
/// The steepest slant searched either way, in degrees; a steeper surface is
/// read as the other axis.
const REACH: f64 = 45.0;
/// The most ink positions the slant search projects; more are thinned.
const MAXIMUM_INK: usize = 6_000;
/// The share of ink below which an area holds no lettering worth measuring.
const MINIMUM_INK: f64 = 0.005;
/// How much sharper the best projection must be than the average one for
/// the slant to be lettering rather than tone or noise.
pub(super) const MINIMUM_STRENGTH: f64 = 1.15;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Along {
    /// The lettering is set in rows along the slant.
    Rows,
    /// The lettering is set in columns across the slant.
    Columns,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PrintedLines {
    /// The slant of the lettering's horizontal axis, degrees clockwise on the
    /// page, in (-45, 45].
    pub(super) angle: f64,
    pub(super) along: Along,
    /// The best projection's sharpness over the average one.
    pub(super) strength: f64,
    /// The ink's page positions, kept to fit a box around it.
    ink: Vec<[f64; 2]>,
}

impl PrintedLines {
    /// The box around the middle 96% of the ink along `angle`, unturned:
    /// [x, y, width, height] about the ink's centre.
    pub(super) fn ink_box(&self, angle: f64) -> [f64; 4] {
        let (sin, cos) = angle.to_radians().sin_cos();
        let mut along = self
            .ink
            .iter()
            .map(|[x, y]| x * cos + y * sin)
            .collect::<Vec<_>>();
        let mut across = self
            .ink
            .iter()
            .map(|[x, y]| -x * sin + y * cos)
            .collect::<Vec<_>>();
        along.sort_by(f64::total_cmp);
        across.sort_by(f64::total_cmp);
        let span = |values: &[f64]| {
            let low = values[values.len() / 50];
            let high = values[values.len() - 1 - values.len() / 50];
            (low, high)
        };
        let ((left, right), (top, bottom)) = (span(&along), span(&across));
        let (u, v) = ((left + right) * 0.5, (top + bottom) * 0.5);
        let (cx, cy) = (u * cos - v * sin, u * sin + v * cos);
        let (width, height) = (right - left + 4.0, bottom - top + 4.0);
        [cx - width * 0.5, cy - height * 0.5, width, height]
    }
}

/// The slant of the lettering printed inside `polygon` (page pixels), or none
/// when the area holds too little ink or no line structure.
pub(super) fn printed_lines(image: &RgbaImage, polygon: &[[f64; 2]]) -> Option<PrintedLines> {
    if polygon.len() < 3 {
        return None;
    }
    let (mut left, mut top) = (f64::INFINITY, f64::INFINITY);
    let (mut right, mut bottom) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for [x, y] in polygon {
        left = left.min(*x);
        top = top.min(*y);
        right = right.max(*x);
        bottom = bottom.max(*y);
    }
    let left = left.max(0.0);
    let top = top.max(0.0);
    let right = right.min(f64::from(image.width()));
    let bottom = bottom.min(f64::from(image.height()));
    if right - left < 8.0 || bottom - top < 8.0 {
        return None;
    }
    let step = ((right - left).max(bottom - top) / SAMPLES).max(1.0);
    let mut samples = Vec::new();
    let mut y = top + step * 0.5;
    while y < bottom {
        let mut x = left + step * 0.5;
        while x < right {
            if inside(polygon, x, y) {
                let pixel = image.get_pixel(x as u32, y as u32).0;
                let luma = 0.299 * f64::from(pixel[0])
                    + 0.587 * f64::from(pixel[1])
                    + 0.114 * f64::from(pixel[2]);
                samples.push((x, y, luma as u8));
            }
            x += step;
        }
        y += step;
    }
    let threshold = otsu(samples.iter().map(|(_, _, luma)| *luma))?;
    // Lettering is the minority tone: dark print on paper, light print on a
    // dark band or screen.
    let dark = samples
        .iter()
        .filter(|(_, _, luma)| *luma <= threshold)
        .count();
    let ink_is_dark = dark * 2 <= samples.len();
    let ink = samples
        .iter()
        .filter(|(_, _, luma)| (*luma <= threshold) == ink_is_dark)
        .map(|(x, y, _)| [*x, *y])
        .collect::<Vec<_>>();
    if (ink.len() as f64) < samples.len() as f64 * MINIMUM_INK || ink.len() < 40 {
        return None;
    }
    let (cx, cy) = (
        ink.iter().map(|[x, _]| x).sum::<f64>() / ink.len() as f64,
        ink.iter().map(|[_, y]| y).sum::<f64>() / ink.len() as f64,
    );
    let stride = ink.len().div_ceil(MAXIMUM_INK);
    let centred = ink
        .iter()
        .step_by(stride)
        .map(|[x, y]| [(x - cx) / step, (y - cy) / step])
        .collect::<Vec<_>>();
    let reach = centred
        .iter()
        .map(|[x, y]| x.hypot(*y))
        .fold(0.0, f64::max)
        .ceil() as usize
        + 1;
    let mut bins = vec![0.0f64; reach * 2 + 1];
    let mut sharpness = |angle: f64, along: Along| {
        let (sin, cos) = angle.to_radians().sin_cos();
        bins.fill(0.0);
        for [x, y] in &centred {
            let across = match along {
                Along::Rows => -x * sin + y * cos,
                Along::Columns => x * cos + y * sin,
            };
            bins[(across.floor() as isize + reach as isize) as usize] += 1.0;
        }
        bins.iter().map(|count| count * count).sum::<f64>()
    };
    let mut scores = Vec::new();
    for degree in -44..=45 {
        let angle = f64::from(degree);
        for along in [Along::Rows, Along::Columns] {
            scores.push((angle, along, sharpness(angle, along)));
        }
    }
    let mean = scores.iter().map(|(_, _, score)| score).sum::<f64>() / scores.len() as f64;
    let (coarse, along, _) = scores
        .iter()
        .copied()
        .max_by(|left, right| left.2.total_cmp(&right.2))?;
    let (angle, best) = (-10..=10)
        .map(|tenth| coarse + f64::from(tenth) * 0.1)
        .map(|angle| (angle, sharpness(angle, along)))
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let angle = if angle > REACH { angle - 90.0 } else { angle };
    Some(PrintedLines {
        angle: (angle * 10.0).round() / 10.0,
        along,
        strength: best / mean,
        ink: ink.clone(),
    })
}

fn inside(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &point in polygon {
        if (point[1] > y) != (previous[1] > y)
            && x < (previous[0] - point[0]) * (y - point[1]) / (previous[1] - point[1]) + point[0]
        {
            inside = !inside;
        }
        previous = point;
    }
    inside
}

/// Otsu's threshold of `values`: the luma that best splits them in two.
fn otsu(values: impl Iterator<Item = u8>) -> Option<u8> {
    let mut histogram = [0.0f64; 256];
    let mut total = 0.0;
    for value in values {
        histogram[usize::from(value)] += 1.0;
        total += 1.0;
    }
    if total == 0.0 {
        return None;
    }
    let sum = histogram
        .iter()
        .enumerate()
        .map(|(value, count)| value as f64 * count)
        .sum::<f64>();
    let (mut below, mut below_sum) = (0.0, 0.0);
    let mut best = (0.0, None);
    for (value, count) in histogram.iter().enumerate() {
        below += count;
        below_sum += value as f64 * count;
        let above = total - below;
        if below == 0.0 || above == 0.0 {
            continue;
        }
        let difference = below_sum / below - (sum - below_sum) / above;
        let between = below * above * difference * difference;
        if between > best.0 {
            best = (between, Some(value as u8));
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use image::Rgba;

    use super::*;

    /// Dark bars of `count` lines, `length` long, slanted by `angle`.
    fn lettering(angle: f64, count: usize, length: f64) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(400, 400, Rgba([255, 255, 255, 255]));
        let (sin, cos) = angle.to_radians().sin_cos();
        for line in 0..count {
            let offset = (line as f64 - count as f64 / 2.0) * 22.0;
            let mut along = -length / 2.0;
            while along < length / 2.0 {
                // Letters with gaps, so rows are not solid stripes.
                if (along / 9.0).floor() as i64 % 3 != 2 {
                    for thickness in -5..=5 {
                        let across = offset + f64::from(thickness);
                        let x = 200.0 + along * cos - across * sin;
                        let y = 200.0 + along * sin + across * cos;
                        image.put_pixel(x as u32, y as u32, Rgba([0, 0, 0, 255]));
                    }
                }
                along += 0.5;
            }
        }
        image
    }

    const PAGE: [[f64; 2]; 4] = [[20.0, 20.0], [380.0, 20.0], [380.0, 380.0], [20.0, 380.0]];

    #[test]
    fn slanted_rows_give_their_slant() {
        for angle in [-14.0, -3.0, 0.0, 6.5, 19.0] {
            let lines = printed_lines(&lettering(angle, 6, 240.0), &PAGE).unwrap();
            assert!((lines.angle - angle).abs() <= 0.6, "{angle}: {lines:?}");
            assert!(lines.strength > MINIMUM_STRENGTH);
        }
    }

    #[test]
    fn columns_give_the_axis_across_them() {
        // Vertical columns turned by 12 degrees: the text's horizontal axis
        // is turned by the same 12 degrees.
        let lines = printed_lines(&lettering(12.0 + 90.0, 6, 240.0), &PAGE).unwrap();
        assert_eq!(lines.along, Along::Columns);
        assert!((lines.angle - 12.0).abs() <= 0.6, "{lines:?}");
    }

    #[test]
    fn light_print_on_a_dark_band_is_read_too() {
        let mut image = lettering(-8.0, 5, 240.0);
        for pixel in image.pixels_mut() {
            pixel.0 = [255 - pixel.0[0], 255 - pixel.0[1], 255 - pixel.0[2], 255];
        }
        let lines = printed_lines(&image, &PAGE).unwrap();
        assert!((lines.angle + 8.0).abs() <= 0.6, "{lines:?}");
    }

    #[test]
    fn a_blank_area_has_no_slant() {
        let image = RgbaImage::from_pixel(400, 400, Rgba([255, 255, 255, 255]));
        assert!(printed_lines(&image, &PAGE).is_none());
    }
}
