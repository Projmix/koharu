# Scan enhancement stage

Status, 2026-10-21: implemented on `main`'s working tree, not yet built end
to end (the machine lacked LLVM for `koharu-torch-sys`). The owner approved
the stage, the layer kind and the MangaJPEG integration, and accepted the
CC BY-NC-SA 4.0 license of its weights for this non-commercial fork.

Goal: low-quality scans and web raws (JPEG noise, grey paper, washed-out
ink) become clean before detection, cleaning and typesetting, without new
artifacts. When a result cannot pass the checks below, the page keeps its
original and the run says why.

## What the Photoshop recipes reduce to

| Recipe step | What it computes | Decision |
| --- | --- | --- |
| Grayscale, Gray Gamma vs Dot Gain | Colour management: the same numbers under two profiles. | Decode once; never assign a profile. Work on BT.601 luma and keep each pixel's chroma, so sepia and duotone pages keep their ink colour. |
| Surface Blur r=4 t=8, or r=1 t=8-12 | Edge-preserving mean: 3x3 (r=1) window, weight `max(0, 1 - abs(d) / (2.5 t))`. | Kept at r=1, t=9. r=4 smears screentone (the article admits it). |
| Levels on the whole page | Moves every tone; clips light tone and dark gradients. | Rejected. |
| Color Range (fuzziness 20-25) + Levels | On grey: snap near-ink to 0 and near-paper to 255 with a soft ramp, mid-tones unchanged. | Kept as the tone curve, only on pages with neutral ink. |
| 1x MangaJPEG by JPEG quality | ESRGAN trained on manga to remove JPEG blocks while keeping screentone. | Kept, one step lighter than the published ranges (below). |
| Waifu2x, Topaz | Trained on anime/photos; smear screentone. | Rejected. |

## Behaviour

`Stage::Enhancement` runs first, per page, only when the user selects it
(not part of `Operation::Full` or the default selection). Settings:
`pipeline.enhancement.mangajpeg` and `pipeline.enhancement.tone`, both on.

1. **Analyse** the imported source: greyscale, monochrome (neutral ink and no
   colour beyond the paper's tint), JPEG quality from the DQT table
   (least-squares match to libjpeg's scaled Annex K table; exact on 15-95),
   8x8 blockiness.
2. **Damage**: only monochrome pages. JPEG below quality 80; a lossless file
   (PNG, BMP, TIFF) that holds a re-saved JPEG, which needs both 8x8
   blockiness (>= 1.06 as quality 50, >= 1.12 as quality 30) and paper grain
   >= 0.2. WebP is skipped.
3. **De-artifact**: MangaJPEG for the quality, then Surface Blur as the
   fallback. A candidate is used only if it passes the checks and measurably
   lowers paper noise or blockiness.
4. **Tone**: on monochrome pages, the masked Levels curve. Ink edge from the
   luminance histogram, paper edge per channel (tinted paper becomes white).
   Ink is the darkest 5-level window peak in 0..=112 holding 1% of the page
   (not the tallest: Osananajimi page 20 holds more grey tone at 38 than ink
   at 0). Its edge goes up while the histogram keeps falling and stays above
   half the peak, at most 16 levels (a tenth let the anti-aliased tail of
   clean ink run 30-60 levels into dark tone): a print scan
   (Shanpankouru 2) spreads its black over 70-110 with dark screentone just
   above, so no single bin reached the old 2% and the ink stayed grey. When
   the ramps would squeeze the mid-tones below 64 levels, the paper ramp
   shortens first and black is lowered only as far as the band needs.
5. **Result**: one opaque raster layer of kind `Enhancement` at the bottom of
   the page, owned by `dev.koharu.pipeline.enhancement`. The source asset is
   never replaced. Detection, OCR, markup, translation crops and inpainting
   read the layer while it is visible. Hiding it is the before/after
   comparison; PSD exports both. A rerun with the same settings skips the
   page; a user-owned layer is never replaced.
6. **Nothing to fix** gives an empty patch (the stage is skipped). **Rejected**
   candidates with nothing else to apply become the page's warning ("kept the
   original page: ..."), and the run continues.

Checks on luminance against the source (`scan::Checks`): mid-tone change
(mean <= 3, p99 <= 10 levels), screentone detail kept (10th percentile of
textured tiles >= 85%), strong-edge F1 >= 0.97, halos <= 5% of mid-tone
pixels.

## Measurements

Screentone crop 192x192 from the owner's chapter 3 page 5 (luma of a
JPEG q100 page), recompressed with the `image` crate's encoder. PSNR in dB
against the clean crop; reference forward in plain Rust from the same
weights.

| JPEG q | Compressed | Surface Blur | HQPlus | HQ | MQ | LQ |
| --- | --- | --- | --- | --- | --- | --- |
| 85 | 37.14 | 39.11 | 38.49 | 35.57 | 31.70 | 25.10 |
| 60 | 29.98 | 30.85 | 31.30 | 31.14 | 30.17 | 25.04 |
| 40 | 26.96 | 27.42 | 27.89 | 28.11 | 28.31 | 24.73 |
| 20 | 23.64 | 23.81 | 24.14 | 24.42 | 25.22 | 24.18 |

LQ added halos at every quality (11-24% of mid-tone pixels, mid-tones
shifted by 4 levels) and fails the checks. Hence the mapping in
`MangaJpegVariant::for_quality`: >= 55 HQPlus, 45-54 HQ, 12-44 MQ, below
LQ. On whole pages recompressed at 15-90, Surface Blur alone gained 0.1-1.5
dB and kept 96-99% of screentone detail; at q90 it lost detail, hence the
cut-off at 90.

On the owner's 106 pages (28 WebP, 26 PNG, 33 JPEG q100, all clean digital
releases): JPEG and WebP pages are untouched, sepia/colour pages get no tone
curve, and 9 monochrome PNG pages get only the tone curve (ink edge 0-32,
paper 244-250). An earlier rule that ran Surface Blur on WebP by blockiness
erased the kraft-paper texture of a colour cover; that is why de-artifacting
is monochrome-only and skips WebP.

## The owner's library (2026-10-21)

20 chapters, 1052 pages in `C:\Code\MangaProject`, seven tagged
`[Digital]`. A tag in the file name is not evidence; the analysis is. Per
page: format, size, JPEG quality, monochrome, 8x8 blockiness, paper grain
(mean deviation from the 3x3 mean on bright flat pixels), and the tone
curve the stage would apply.

- Clean digital releases measure paper grain 0.02-0.13 (JPEG q100 or flat
  PNG). Physical scans measure 1.1-1.8 and show grey paper (paper edge 226-
  230) or grey ink (ink edge 19-32).
- Two rules were changed after looking at 1:1 crops:
  - A `[Digital]` PNG chapter with a 4-pixel checker screentone measured
    blockiness 1.06, the same as its own JPEG q50 copy, and the old rule ran
    Surface Blur on 23 clean pages. Its paper grain is 0.02; its q30-q60
    copies measure 0.41-0.47. A lossless page now also needs grain >= 0.2.
  - Six chapters are JPEG q85 at up to 4100x6000. No artifact shows at 1:1,
    and MangaJPEG on 25 megapixels would take hours on the owner's CPU
    (i7-1355U). JPEG now needs quality below 80.
- After the change: 8 pages get de-artifacting (7 JPEG q75 pages in one
  chapter, 1 page in a physical scan); the tone curve runs on the physical
  scans (37 pages with grey paper in one chapter, 16 in another) and on
  single pages elsewhere with grey ink. On a scan with ink at 29 it turns
  the ink black and keeps screentone and mid-tones.
- Colour detection is per page and mostly right: covers (one or two
  pages per chapter) come out as colour. One limit: a black-and-white
  chapter with an orange watermark over every page counts as colour on 40
  of 43 pages, so the stage leaves those pages alone. The failure is safe
  (no change), and that chapter is a clean digital release anyway.
- The 84 colour pages need no white balance: the measured tints are the
  artists' design (red, cream or dark backgrounds, pages without any
  white). Noise there is print grain on scanned covers; two pages of one
  chapter are JPEG q75 with light artifacts.
- One chapter is 800x1228: its problem is resolution, which this stage
  does not address (upscaling is a separate decision).

## Open

- Build and run the stage end to end; compare the libtorch MangaJPEG port
  with the reference forward on identical crops (`koharu-ml` bin
  `manga_jpeg`), and time it on the owner's CPU.
- Calibrate the checks on a real low-quality scan, not only on recompressed
  clean pages.
- Upscaling (MangaJaNai) is a separate decision: it changes page size.
- Colour pages: IllustrationJaNai DeJPEG FDAT is the analogue, not ported.
