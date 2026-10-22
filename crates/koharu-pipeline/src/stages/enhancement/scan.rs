//! Deterministic scan clean-up: page analysis, Photoshop's Surface Blur, the
//! tone curve that the Color Range + Levels recipe produces, and the checks a
//! de-artifacted luminance must pass before it may replace the source.
//!
//! Everything works on luminance. Colour pages keep their chroma: the change
//! in luminance is added to every channel, so a sepia or duotone page keeps
//! its ink colour. The tone curve runs only on pages whose ink is neutral, so
//! tinted ink and coloured paper that are part of the design stay as drawn.
//! Thresholds come from the measurements in docs/fork/scan-restoration.md.
use image::{GrayImage, Luma, RgbImage};
use rayon::prelude::*;

/// Luminance of a page as floats in `0..=255`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plane {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data: Vec<f32>,
}

impl Plane {
    pub(crate) fn from_gray(image: &GrayImage) -> Self {
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            data: image
                .as_raw()
                .iter()
                .map(|value| f32::from(*value))
                .collect(),
        }
    }

    /// ITU-R BT.601 luma, the weighting JPEG and libwebp use.
    pub(crate) fn luma(image: &RgbImage) -> Self {
        Self {
            width: image.width() as usize,
            height: image.height() as usize,
            data: image
                .pixels()
                .map(|pixel| {
                    let [r, g, b] = pixel.0.map(f32::from);
                    0.299 * r + 0.587 * g + 0.114 * b
                })
                .collect(),
        }
    }

    pub(crate) fn to_gray(&self) -> GrayImage {
        GrayImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            Luma([to_u8(self.at(x as usize, y as usize))])
        })
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    fn clamped(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.at(x, y)
    }

    fn map(&self, f: impl Fn(usize, usize) -> f32 + Sync) -> Self {
        let mut data = vec![0.0; self.data.len()];
        data.par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, value) in row.iter_mut().enumerate() {
                    *value = f(x, y);
                }
            });
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    fn mean3(&self) -> Self {
        self.map(|x, y| {
            let mut sum = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    sum += self.clamped(x as isize + dx, y as isize + dy);
                }
            }
            sum / 9.0
        })
    }
}

fn to_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn spread(pixel: [u8; 3]) -> u8 {
    let [r, g, b] = pixel;
    r.max(g).max(b) - r.min(g).min(b)
}

/// What the page is and what is wrong with it. Nothing here changes pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Analysis {
    /// Grey ink on grey paper, possibly stored with colour channels.
    pub(crate) grayscale: bool,
    /// Neutral ink and no colour beyond the paper's tint: a greyscale page
    /// printed or scanned on tinted paper. The tone curve runs only on these.
    pub(crate) monochrome: bool,
    /// libjpeg quality of the file's luminance table, for JPEG sources.
    pub(crate) jpeg_quality: Option<u8>,
    /// A lossless container (PNG, BMP, TIFF) that may hold a re-saved JPEG.
    pub(crate) lossless: bool,
    /// Step ratio across 8-pixel block borders (1.0 = no blocks).
    pub(crate) blockiness: f32,
    /// Mean deviation from the 3x3 mean on bright, locally flat pixels: the
    /// paper's own noise. Clean digital art measures about 0.03.
    pub(crate) paper_grain: f32,
}

impl Analysis {
    pub(crate) fn new(image: &RgbImage, luma: &Plane, encoded: &[u8]) -> Self {
        let (mut counted, mut total_spread, mut strong) = (0u64, 0u64, 0u64);
        let (mut ink, mut ink_spread) = (0u64, 0u64);
        let mut paper = Vec::new();
        for (pixel, value) in image.pixels().zip(&luma.data) {
            let value = *value;
            let spread = spread(pixel.0);
            if value < 96.0 {
                ink += 1;
                ink_spread += u64::from(spread);
            }
            if value >= 200.0 {
                paper.push(spread);
            }
            // Compression puts chroma noise on the extremes.
            let [r, g, b] = pixel.0;
            if r.max(g).max(b) <= 8 || r.min(g).min(b) >= 247 {
                continue;
            }
            counted += 1;
            total_spread += u64::from(spread);
            strong += u64::from(spread > 24);
        }
        let grayscale =
            counted == 0 || total_spread as f64 / counted as f64 <= 3.0 && strong * 1000 <= counted;
        let neutral_ink = ink == 0 || ink_spread as f64 / ink as f64 <= 8.0;
        let paper_tint = if paper.is_empty() {
            0
        } else {
            let middle = paper.len() / 2;
            *paper.select_nth_unstable(middle).1
        };
        let monochrome = grayscale
            || neutral_ink && {
                let limit = (u16::from(paper_tint) + 12).max(24);
                let (mut mid, mut colourful) = (0u64, 0u64);
                for (pixel, value) in image.pixels().zip(&luma.data) {
                    if (40.0..200.0).contains(value) {
                        mid += 1;
                        colourful += u64::from(u16::from(spread(pixel.0)) > limit);
                    }
                }
                colourful * 50 <= mid
            };
        Self {
            grayscale,
            monochrome,
            jpeg_quality: jpeg_quality(encoded),
            lossless: encoded.starts_with(b"\x89PNG")
                || encoded.starts_with(b"BM")
                || encoded.starts_with(b"II*\0")
                || encoded.starts_with(b"MM\0*"),
            blockiness: blockiness(luma),
            paper_grain: paper_grain(luma),
        }
    }

    /// The JPEG quality to restore for, or `None` when the page shows no
    /// compression damage worth touching.
    ///
    /// Only monochrome pages qualify: on a colour cover the filter erased the
    /// kraft-paper texture the artist drew, and MangaJPEG is trained on
    /// greyscale manga. JPEG at quality 80 and above is left alone: on the
    /// owner's q85 releases (up to 4100x6000) no artifact shows at 1:1, and the
    /// model would take hours per page on a CPU.
    ///
    /// A lossless file counts as a re-saved JPEG only with both 8x8 block
    /// structure and noise on the paper: a clean digital page with a 4-pixel
    /// checker screentone measured blockiness 1.064, the same as its own JPEG
    /// q50 copy, but paper grain 0.03 against 0.41-0.47 for its q30-q60
    /// copies. WebP is skipped: VP8 blocks are not on JPEG's grid, and on the
    /// owner's WebP pages that measure only picked up paper texture.
    pub(crate) fn damage(self) -> Option<u8> {
        if !self.monochrome {
            return None;
        }
        match self.jpeg_quality {
            Some(quality) => (quality < 80).then_some(quality),
            None if !self.lossless || self.paper_grain < 0.2 => None,
            None if self.blockiness >= 1.12 => Some(30),
            None if self.blockiness >= 1.06 => Some(50),
            None => None,
        }
    }
}

/// libjpeg's luminance base table (ITU-T T.81 Annex K) in the zig-zag order
/// DQT stores it in.
const ANNEX_K_LUMA_ZIGZAG: [u16; 64] = [
    16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37,
    29, 40, 58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81,
    87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99,
];

/// The libjpeg quality whose scaled Annex K table is closest (least squares)
/// to the file's luminance table. `None` for bytes that are not a JPEG.
pub(crate) fn jpeg_quality(bytes: &[u8]) -> Option<u8> {
    let table = luma_table(bytes)?;
    (1..=100u32)
        .map(|quality| {
            let scale = if quality < 50 {
                5000 / quality
            } else {
                200 - 2 * quality
            };
            let error: u64 = ANNEX_K_LUMA_ZIGZAG
                .iter()
                .zip(&table)
                .map(|(base, actual)| {
                    let expected = ((u32::from(*base) * scale + 50) / 100).clamp(1, 255);
                    (i64::from(expected) - i64::from(*actual))
                        .unsigned_abs()
                        .pow(2)
                })
                .sum();
            (error, quality)
        })
        .min()
        .map(|(_, quality)| quality as u8)
}

fn luma_table(bytes: &[u8]) -> Option<[u16; 64]> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return None;
    }
    let mut offset = 2;
    while offset + 4 <= bytes.len() {
        if bytes[offset] != 0xff {
            return None;
        }
        let marker = bytes[offset + 1];
        if marker == 0xff {
            offset += 1;
            continue;
        }
        if marker == 0x01 || (0xd0..=0xd8).contains(&marker) {
            offset += 2;
            continue;
        }
        if marker == 0xda || marker == 0xd9 {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]));
        let segment = bytes.get(offset + 4..offset + 2 + length)?;
        if marker == 0xdb {
            let mut cursor = 0;
            while cursor < segment.len() {
                let precision = segment[cursor] >> 4;
                let id = segment[cursor] & 0x0f;
                cursor += 1;
                let size = if precision == 0 { 64 } else { 128 };
                let values = segment.get(cursor..cursor + size)?;
                cursor += size;
                if id == 0 {
                    return Some(std::array::from_fn(|index| {
                        if precision == 0 {
                            u16::from(values[index])
                        } else {
                            u16::from_be_bytes([values[2 * index], values[2 * index + 1]])
                        }
                    }));
                }
            }
        }
        offset += 2 + length;
    }
    None
}

/// Mean step across 8-pixel block borders over the mean step elsewhere.
pub(crate) fn blockiness(plane: &Plane) -> f32 {
    const PERIOD: usize = 8;
    let (mut border, mut border_count, mut inner, mut inner_count) = (0.0f64, 0u64, 0.0f64, 0u64);
    let mut add = |step: f32, at_border: bool| {
        if at_border {
            border += f64::from(step);
            border_count += 1;
        } else {
            inner += f64::from(step);
            inner_count += 1;
        }
    };
    for y in 0..plane.height {
        for x in 0..plane.width.saturating_sub(1) {
            add(
                (plane.at(x + 1, y) - plane.at(x, y)).abs(),
                x % PERIOD == PERIOD - 1,
            );
        }
    }
    for y in 0..plane.height.saturating_sub(1) {
        for x in 0..plane.width {
            add(
                (plane.at(x, y + 1) - plane.at(x, y)).abs(),
                y % PERIOD == PERIOD - 1,
            );
        }
    }
    let border = border / border_count.max(1) as f64;
    let inner = inner / inner_count.max(1) as f64;
    if inner <= f64::EPSILON {
        1.0
    } else {
        (border / inner) as f32
    }
}

/// Mean |pixel - 3x3 mean| over bright (>= 200), locally flat (3x3 range
/// <= 40) pixels, sampled on every other row and column.
pub(crate) fn paper_grain(plane: &Plane) -> f32 {
    let (mut grain, mut count) = (0.0f64, 0u64);
    for y in (1..plane.height.saturating_sub(1)).step_by(2) {
        for x in (1..plane.width.saturating_sub(1)).step_by(2) {
            let center = plane.at(x, y);
            if center < 200.0 {
                continue;
            }
            let (mut sum, mut low, mut high) = (0.0, f32::MAX, f32::MIN);
            for dy in 0..3 {
                for dx in 0..3 {
                    let value = plane.at(x + dx - 1, y + dy - 1);
                    sum += value;
                    low = low.min(value);
                    high = high.max(value);
                }
            }
            if high - low <= 40.0 {
                grain += f64::from((center - sum / 9.0).abs());
                count += 1;
            }
        }
    }
    (grain / count.max(1) as f64) as f32
}

/// Photoshop's Surface Blur at radius 1, as open reimplementations define it:
/// a flat 3x3 window whose pixels count with `max(0, 1 - |d| / (2.5 t))`, so
/// values across a line do not mix. 2.5 is the commonly cited reverse
/// engineering of Photoshop's threshold; the recipe's threshold is 8-12 and
/// 9 kept screentone best in our measurements.
pub(crate) fn surface_blur(plane: &Plane) -> Plane {
    const REACH: f32 = 2.5 * 9.0;
    plane.map(|x, y| {
        let center = plane.at(x, y);
        let (mut sum, mut weight) = (0.0, 0.0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let value = plane.clamped(x as isize + dx, y as isize + dy);
                let w = (1.0 - (value - center).abs() / REACH).max(0.0);
                sum += w * value;
                weight += w;
            }
        }
        sum / weight
    })
}

/// Where paper and ink tones end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Levels {
    /// At or below: 0. A linear ramp reaches the identity at `2 * black`.
    pub(crate) black: f32,
    /// At or above: 255. The mirror ramp leaves the identity at
    /// `255 - 2 * (255 - white)`.
    pub(crate) white: f32,
}

impl Levels {
    fn identity_band(self) -> (f32, f32) {
        (2.0 * self.black, 255.0 - 2.0 * (255.0 - self.white))
    }

    /// The luminance-masked Levels that Color Range + Levels produces: only
    /// values near ink and paper move; mid-tones are untouched.
    pub(crate) fn apply(self, value: f32) -> f32 {
        let (low, high) = self.identity_band();
        if value <= self.black {
            0.0
        } else if value < low {
            (value - self.black) * low / (low - self.black)
        } else if value >= self.white {
            255.0
        } else if value > high {
            high + (value - high) * (255.0 - high) / (self.white - high)
        } else {
            value
        }
    }

    fn is_identity(self) -> bool {
        self.black <= 4.0 && self.white >= 251.0
    }
}

fn histogram(values: impl Iterator<Item = f32>) -> ([u64; 256], u64) {
    let mut histogram = [0u64; 256];
    let mut total = 0;
    for value in values {
        histogram[usize::from(to_u8(value))] += 1;
        total += 1;
    }
    (histogram, total)
}

/// The paper's dark edge: the tallest bin in 160..=255 holding at least 5% of
/// the page, extended down while bins stay above a tenth of it (at most 40
/// levels). `None` when no bright tone dominates.
fn paper_edge(histogram: &[u64; 256], total: u64) -> Option<f32> {
    let peak = (160..=255).max_by_key(|&value| (histogram[value], value))?;
    if histogram[peak] * 20 < total {
        return None;
    }
    let floor = histogram[peak] / 10;
    let mut edge = peak;
    while edge > peak.saturating_sub(40) && edge > 160 && histogram[edge - 1] >= floor {
        edge -= 1;
    }
    Some(edge as f32)
}

/// The ink's light edge: the darkest 5-level window in 0..=112 that holds at
/// least 1% of the page and stops rising, extended up while it stays above
/// half its height and keeps falling, at most 16 levels: just past the peak,
/// as in the Levels step of a scanlation clean. Windows rather than bins, and a range
/// past 96, because a washed-out print scan spreads its black over 70-110
/// instead of stacking it in one level. The darkest peak rather than the
/// tallest, because a page with real black can hold more mid-grey screentone
/// than ink, and that tone must not become black.
fn ink_edge(histogram: &[u64; 256], total: u64) -> Option<f32> {
    let window = |value: usize| {
        histogram[value.saturating_sub(2)..=(value + 2).min(255)]
            .iter()
            .sum::<u64>()
    };
    let peak = (0..=112)
        .find(|&value| window(value) * 100 >= total && window(value) >= window(value + 1))?;
    // Half height, not a tenth: the anti-aliased tail of clean black ink
    // stays above a tenth of a window for 30-60 levels.
    let floor = window(peak) / 2;
    let mut edge = peak;
    while edge < peak + 16
        && edge < 112
        && window(edge + 1) >= floor
        && window(edge + 1) <= window(edge)
    {
        edge += 1;
    }
    Some(edge as f32)
}

fn levels_of(histogram: &[u64; 256], total: u64) -> Levels {
    let mut levels = Levels {
        black: ink_edge(histogram, total).unwrap_or(0.0),
        white: paper_edge(histogram, total).unwrap_or(255.0),
    };
    // Keep a real mid-tone band between the two ramps: the paper ramp
    // shortens first, then the ink ramp gives up only what the band needs.
    let (low, high) = levels.identity_band();
    if high - low < 64.0 {
        levels.white = levels.white.max(223.0);
        let (_, high) = levels.identity_band();
        levels.black = levels.black.min(((high - 64.0) / 2.0).max(32.0));
    }
    levels
}

/// The tone curve of a monochrome page after its luminance changed from
/// `before` to `after`: ink from luminance, paper per channel so tinted paper
/// becomes white. `None` when it would change nothing.
pub(crate) fn tone(image: &RgbImage, before: &Plane, after: &Plane) -> Option<[Levels; 3]> {
    let (histogram_luma, total) = histogram(after.data.iter().copied());
    let black = levels_of(&histogram_luma, total).black;
    let levels: [Levels; 3] = std::array::from_fn(|channel| {
        let (histogram, total) = histogram(image.pixels().enumerate().map(|(index, pixel)| {
            f32::from(pixel[channel]) + after.data[index] - before.data[index]
        }));
        Levels {
            black,
            white: levels_of(&histogram, total).white,
        }
    });
    (!levels.iter().all(|levels| levels.is_identity())).then_some(levels)
}

/// Thresholds a de-artifacted luminance must meet against the source.
/// Calibrated on screentone crops recompressed at JPEG 85/60/40/20: every
/// model choice that looked right stayed within them, and MangaJPEG LQ (halos,
/// shifted mid-tones) did not.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Checks {
    /// Mean and 99th percentile of |after - before| (3x3 means) in mid-tones.
    pub(crate) midtone_mean: f32,
    pub(crate) midtone_p99: f32,
    /// 10th percentile, over textured mid-grey tiles, of retained detail.
    pub(crate) texture_p10: f32,
    /// F1 of strong edges after against before.
    pub(crate) edge_f1: f32,
    /// Share of mid-tone pixels pushed outside their 3x3 neighbourhood.
    pub(crate) halo: f32,
    /// Standard deviation of paper pixels, and blockiness, before and after.
    pub(crate) paper_noise: (f32, f32),
    pub(crate) blockiness: (f32, f32),
}

impl Checks {
    pub(crate) fn new(before: &Plane, after: &Plane) -> Self {
        let before_mean = before.mean3();
        let after_mean = after.mean3();
        let mut midtone: Vec<f32> = before_mean
            .data
            .iter()
            .zip(&after_mean.data)
            .filter(|(before, _)| (38.0..=217.0).contains(*before))
            .map(|(before, after)| (after - before).abs())
            .collect();
        let midtone_mean = midtone.iter().sum::<f32>() / midtone.len().max(1) as f32;
        let midtone_p99 = percentile(&mut midtone, 0.99);

        let detail = |plane: &Plane, mean: &Plane| -> Vec<f32> {
            plane
                .data
                .iter()
                .zip(&mean.data)
                .map(|(value, mean)| (value - mean).abs())
                .collect()
        };
        let before_detail = detail(before, &before_mean);
        let after_detail = detail(after, &after_mean);
        const TILE: usize = 32;
        let mut retained = Vec::new();
        for ty in 0..before.height / TILE {
            for tx in 0..before.width / TILE {
                let (mut mean, mut energy_before, mut energy_after) = (0.0, 0.0, 0.0);
                for y in ty * TILE..(ty + 1) * TILE {
                    for x in tx * TILE..(tx + 1) * TILE {
                        let index = y * before.width + x;
                        mean += before.data[index];
                        energy_before += before_detail[index];
                        energy_after += after_detail[index];
                    }
                }
                let area = (TILE * TILE) as f32;
                if (60.0..=200.0).contains(&(mean / area)) && energy_before / area > 12.0 {
                    retained.push(energy_after / energy_before);
                }
            }
        }
        let texture_p10 = if retained.len() < 8 {
            1.0
        } else {
            percentile(&mut retained, 0.1)
        };

        let edges_before = sobel(&before_mean);
        let edges_after = sobel(&after_mean);
        let (mut both, mut only_before, mut only_after) = (0u64, 0u64, 0u64);
        for (before, after) in edges_before.data.iter().zip(&edges_after.data) {
            match (*before > 160.0, *after > 160.0) {
                (true, true) => both += 1,
                (true, false) => only_before += 1,
                (false, true) => only_after += 1,
                (false, false) => {}
            }
        }
        let edge_f1 = if both + only_before + only_after == 0 {
            1.0
        } else {
            2.0 * both as f32 / (2 * both + only_before + only_after) as f32
        };

        let (mut halo, mut band) = (0u64, 0u64);
        for y in 0..before.height {
            for x in 0..before.width {
                let value = before.at(x, y);
                if !(38.0..=217.0).contains(&value) {
                    continue;
                }
                band += 1;
                let (mut low, mut high) = (f32::MAX, f32::MIN);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let neighbour = before.clamped(x as isize + dx, y as isize + dy);
                        low = low.min(neighbour);
                        high = high.max(neighbour);
                    }
                }
                let out = after.at(x, y);
                halo += u64::from(out > high + 8.0 || out < low - 8.0);
            }
        }

        let paper = |plane: &Plane| {
            let values: Vec<f32> = before_mean
                .data
                .iter()
                .zip(&plane.data)
                .filter(|(mean, _)| **mean >= 235.0)
                .map(|(_, value)| *value)
                .collect();
            let count = values.len().max(1) as f32;
            let mean = values.iter().sum::<f32>() / count;
            (values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / count)
                .sqrt()
        };

        Self {
            midtone_mean,
            midtone_p99,
            texture_p10,
            edge_f1,
            halo: halo as f32 / band.max(1) as f32,
            paper_noise: (paper(before), paper(after)),
            blockiness: (blockiness(before), blockiness(after)),
        }
    }

    /// Why the result may not replace the source, if it may not.
    pub(crate) fn rejection(&self) -> Option<String> {
        if self.midtone_mean > 3.0 || self.midtone_p99 > 10.0 {
            return Some(format!(
                "mid-tones shifted (mean {:.1}, p99 {:.1})",
                self.midtone_mean, self.midtone_p99
            ));
        }
        if self.texture_p10 < 0.85 {
            return Some(format!(
                "screentone lost detail (kept {:.0}%)",
                self.texture_p10 * 100.0
            ));
        }
        if self.edge_f1 < 0.97 {
            return Some(format!("line edges moved (F1 {:.3})", self.edge_f1));
        }
        if self.halo > 0.05 {
            return Some(format!("halos around lines ({:.1}%)", self.halo * 100.0));
        }
        None
    }

    /// Whether the result is measurably cleaner than the source.
    pub(crate) fn improves(&self) -> bool {
        self.paper_noise.1 <= 0.9 * self.paper_noise.0
            || self.blockiness.1 < self.blockiness.0 - 0.005
    }
}

fn sobel(plane: &Plane) -> Plane {
    plane.map(|x, y| {
        let p = |dx: isize, dy: isize| plane.clamped(x as isize + dx, y as isize + dy);
        let gx = p(1, -1) + 2.0 * p(1, 0) + p(1, 1) - p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1);
        let gy = p(-1, 1) + 2.0 * p(0, 1) + p(1, 1) - p(-1, -1) - 2.0 * p(0, -1) - p(1, -1);
        (gx * gx + gy * gy).sqrt()
    })
}

fn percentile(values: &mut [f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f32 * fraction).round() as usize;
    *values.select_nth_unstable_by(index, f32::total_cmp).1
}

/// Applies a new luminance and the tone curve to the page. Grey pages come
/// back grey; colour pages keep each pixel's chroma.
pub(crate) fn compose(
    image: &RgbImage,
    analysis: Analysis,
    before: &Plane,
    after: &Plane,
    tone: Option<[Levels; 3]>,
) -> image::DynamicImage {
    if analysis.grayscale {
        let levels = tone.map(|levels| levels[1]);
        return image::DynamicImage::ImageLuma8(GrayImage::from_fn(
            image.width(),
            image.height(),
            |x, y| {
                let value = after.at(x as usize, y as usize);
                Luma([to_u8(levels.map_or(value, |levels| levels.apply(value)))])
            },
        ));
    }
    let mut out = image.clone();
    for (index, pixel) in out.pixels_mut().enumerate() {
        let delta = after.data[index] - before.data[index];
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) + delta;
            pixel[channel] = to_u8(tone.map_or(value, |levels| levels[channel].apply(value)));
        }
    }
    image::DynamicImage::ImageRgb8(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_quality_reads_the_encoder_setting() {
        let image = GrayImage::from_fn(64, 64, |x, y| Luma([((x * 7 + y * 3) % 256) as u8]));
        for quality in [15, 40, 60, 85, 95] {
            let mut bytes = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality)
                .encode_image(&image)
                .unwrap();
            assert_eq!(jpeg_quality(&bytes), Some(quality));
        }
        assert_eq!(jpeg_quality(b"\x89PNG"), None);
    }

    #[test]
    fn levels_move_only_ink_and_paper() {
        let levels = Levels {
            black: 30.0,
            white: 230.0,
        };
        assert_eq!(levels.apply(20.0), 0.0);
        assert_eq!(levels.apply(240.0), 255.0);
        for value in [60.0, 128.0, 180.0] {
            assert_eq!(levels.apply(value), value);
        }
        let mut previous = -1.0;
        for value in 0..=255 {
            let mapped = levels.apply(value as f32);
            assert!(mapped >= previous, "the curve is monotonic");
            previous = mapped;
        }
    }

    #[test]
    fn washed_out_ink_is_found_before_dark_screentone() {
        // A print scan's black spread over 66-86 with dark screentone at
        // 90-100 above it, on mostly white paper.
        let mut histogram = [0u64; 256];
        for value in 66..=86 {
            histogram[value] = 50 - (value as u64).abs_diff(74) * 2;
        }
        for value in 87..=100 {
            histogram[value] = 30 + (value as u64).abs_diff(87);
        }
        histogram[255] = 6000;
        let total = histogram.iter().sum();
        let levels = levels_of(&histogram, total);
        assert!(
            (80.0..=90.0).contains(&levels.black),
            "the ink ends past its peak and before the screentone: {}",
            levels.black
        );
        assert_eq!(levels.white, 255.0);
    }

    #[test]
    fn mid_grey_screentone_is_not_taken_for_ink() {
        // Real black at 0 and more grey screentone around 38 than ink.
        let mut histogram = [0u64; 256];
        histogram[0] = 120;
        for value in 30..=46 {
            histogram[value] = 40;
        }
        histogram[255] = 6000;
        let total = histogram.iter().sum();
        assert!(levels_of(&histogram, total).black < 8.0);
    }
}
