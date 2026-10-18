# Pipeline improvement plan

A review of each stage (detection, OCR, cleaning, translation, typesetting
and a final check) on `main` at `ce0ce15` (2026-10-18, after the merge of
upstream `main` at `85d6de3`), with what is weak, what to change, the
expected gain and a rough effort (S: a day or less, M: a few days, L: a week
or more). Nothing here is implemented yet; the owner picks items, and each
lands as its own PR. Upstream removed `koharu_metrics` telemetry, so the
measurements below must be offline, not metrics events.

Read together with [README.md](README.md) (fork state and the JEV OCR flow)
and the root `AGENTS.md`. Line references are to `main` at the commit above.

## Priorities

| # | Item | Stage | Gain | Effort |
| --- | --- | --- | --- | --- |
| 1 | Concurrent API OCR requests within a page | OCR | API and hybrid OCR several times faster | S |
| 2 | Offline evaluation set and scorer | all | Lets every later choice be measured | M |
| 3 | Review flags in Scene, then JEV OCR selection | OCR | Better text and an explicit "check by hand" list | M+M |
| 3a | Keep profanity and register; cross-check suspicious lines with DeepL or Google | translation | Swearing and rude lines are not softened or dropped | M |
| 4 | Surface text overflow and add "shorten" | translation, typesetting | Fewer manual rewrites of long lines | M |
| 5 | Project glossary and character sheet | translation | Consistent names, gender and address in Russian | M |
| 6 | Masked compositing and a fallback chain for API cleanup | cleaning | Art outside text stays untouched; fewer failed pages | M |
| 7 | Page images in chapter translation | translation | Speaker and gender from the art | M |
| 8 | Per-region inpainting routing (fill, LaMa, cloud) | cleaning | Lower cloud cost, faster cleanup | M |
| 9 | Missed-text check per page | detection | Text the detector skipped gets flagged | M |
| 10 | Translation QA through JEV | translation | Suspicious translations flagged like OCR | S after 3 |
| 11 | Reading-order checks for Japanese pages | detection | Fewer swapped lines in chapter context | M |
| 12 | Tighter text masks from an unused mask model | detection, cleaning | Less over-erased art | M |
| 13 | Final check before export | final QA | One list of everything still wrong on a chapter | M after 3 |

## Cross-cutting

### Evaluation set (item 2)

There is no way to tell whether a model or threshold change helps. Build a
small labeled set from the owner's manga (20 to 30 pages: vertical and
horizontal text, furigana, tiny text, SFX, handwriting, textured
backgrounds) with expected OCR text per region and reading order. Extend the
existing in-process runner `crates/koharu-pipeline/src/bin/run.rs` to write
per-region JSON (OCR text per pass, translation, render diagnostics) and add
a scorer for character error rate, misses, hallucinated regions, overflow
count, cost and time. `docs/research/api-ocr-models-2026-08.md` already lists
what such a benchmark must cover. Keep the set out of the repository
(`AGENTS.md`: no datasets); commit only the runner and scorer.

### Review flags (part of item 3)

OCR, translation, overflow and inpainting checks all need the same output: a
region a human must look at, with a reason. Add one Scene-owned component on
the region, committed in the same patch as the result that raised it, so it
survives save, undo and versions. A user edit clears it. The editor lists
flagged regions and jumps to them. Build it once, with the JEV PR, and reuse
it for items 4, 6, 9 and 10. Do not add a second job kind for review.

## Detection

Current state: one model, `koharu-layout-rfdetr-seg-2xl`, with global text,
bubble and panel thresholds (`stages/detection.rs:63`). Reading order is a
geometric panel, bubble and text tree, column-major for Japanese and Chinese
(`stages/detection.rs:1837`, `:1879`). The inpainting text mask is the layout
masks dilated and closed with a radius of about 6 px per 1024 px of the long
side (`stages/detection.rs:1486`).

- **Missed text (item 9).** Nothing checks recall. A region the detector
  misses is never read, translated or cleaned, and nobody is told. Option: one
  vision-LLM call per page that lists visible text lines with rough boxes;
  lines with no overlapping region become flagged empty regions. Run it only
  when the user enables it, since it costs one call per page.
- **Reading order (item 11).** Errors in the geometric order flow into the
  chapter prompt as wrong context. Bring back the deterministic Japanese
  order checks from `d1ebfef` (`git show d1ebfef --
  crates/koharu-pipeline/src/stages/ocr.rs`) as a validator; when it finds an
  ambiguous page (overlapping columns, text outside panels), flag the page
  instead of guessing.
- **Mask quality (item 12).** `koharu-ml` contains models the pipeline never
  calls: `manga_text_mask`, `comic_text_detector`, the speech-bubble YOLOs,
  the onomatopoeia detector and `font_detector` (they appear only in
  `koharu-ml` and its benches). A pixel-level text mask intersected with the
  layout region would erase less art than a dilated region mask. This is a
  model integration: compare masks on identical pages first (`AGENTS.md`,
  Upstream Alignment) and open an issue before the PR.

## OCR

Current state: `local`, `api` or hybrid `local-api`. In hybrid mode the local
pass runs on the whole chapter, then the API pass rereads every region with
the local text as an untrusted draft, and the API result always replaces the
local one (`stages/ocr.rs:251`). API OCR is supported for OpenAI,
OpenAI-compatible, OpenRouter and Qwen-VL-OCR (`provider.rs:212`).

- **Sequential API calls (item 1).** `run_api` awaits one region at a time
  (`stages/ocr.rs:261`). A page with 20 bubbles makes 20 round trips in
  series. Send them with bounded concurrency (for example 4 to 8 in flight,
  configurable per provider) inside the same job, keeping result order and
  cooperative stop. This stays one job per stage. Quota errors already map to
  `QuotaExceeded` (`remote/mod.rs:220`); stop the remaining requests on the
  first one.
- **No confidence (feeds item 3).** `OcrAnalysis.confidence` is always
  `None` (`stages/ocr.rs:416`). The local models decode token by token, so they can
  return a mean or minimum token probability without an extra pass. Store it; it is
  the main numeric input for JEV and later lets the API pass skip regions the
  local model read confidently.
- **JEV selection (item 3).** Follow the flow in [README.md](README.md#jev).
  The cheapest first step needs no judge: when local and API text agree after
  normalization, accept it. Only disagreements go to JEV.
- **More OCR models.** Gemini and Claude are not OCR providers directly, but
  they are reachable through OpenRouter, which already supports OCR. Benchmark
  them on the evaluation set through OpenRouter before writing direct
  adapters.
- **Text direction.** Vertical versus horizontal is decided by aspect ratio
  alone (`stages/ocr.rs:411`). Narrow horizontal lines in tall bubbles are
  misclassified. Low priority; the renderer or JEV inputs can use column
  count instead once it exists.

## JEV

Checked on 2026-10-18 against public sources:

- JEV is TypeSafe AI's "System One" decision model (`jev-1.13.0`, alias
  `jev-latest`). It returns typed answers, not text: `noul` (probability of
  yes), `choice` (up to 255 options with per-option probabilities and
  confidence) and `score` (2 to 10 ordered levels).
- **Input is text only.** JEV cannot look at the crop. The image-grounded
  reading must come from the vision LLM; JEV arbitrates from text and numeric
  features (both candidates, local confidence, agreement, character classes,
  length against region area, column count, source language).
- The request is `POST .../v1/systemone` with a `state` and a `questions`
  map, bearer auth. LLM Gateway also serves it and announced on 2026-10-12 an
  OpenAI-compatible `/v1/decisions` endpoint and the planned deprecation of
  `/v1/systemone`. `d1ebfef` called `/v1/system-one/evaluate`, which does not
  match either.
- Price on LLM Gateway: $0.042 per million input tokens, output free, 600
  requests per minute. All questions in one request share one read of the
  state, so one batch per page is the right unit.

Implementation rules, in addition to the README:

- Off by default. A settings toggle enables it; the key is stored only in
  `koharu_secrets` (OS keychain). No environment variable or project file.
- Confirm the TypeSafe base URL and the endpoint that will remain
  (`/v1/systemone` or `/v1/decisions`) from TypeSafe's own documentation
  before writing the client. A community Rust crate (`typesafe-rs` 0.1.0)
  exists; prefer a small `reqwest` client like the other providers, since a
  new dependency needs an issue first.
- When JEV is unreachable, fall back to "flag every disagreement" and report
  that the judge was offline.

## Translation

Current state: page mode sends ordered segments with role and bubble
metadata, user-edited translations as references, and the page image when the
model has vision (`stages/translation.rs:251`). Chapter mode sends every page
in one request with adaptive units (`stages/translation.rs:86`), capped at an
estimated 100k input tokens (`koharu-translator/src/lib.rs:19`). Schemas go
through each provider's structured output.

- **Overflow is computed and dropped (item 4).** The renderer reports
  `TextOverflow` and `TextBelowReadableSize` per text node
  (`koharu-renderer/src/text_renderer.rs:276`), but nothing outside the
  renderer's own tests reads these diagnostics. The user finds overflowing
  bubbles by eye. Surface them as review flags, and add a "shorten" action
  that re-translates one block with a character budget derived from the fitted
  layout. This is the useful half of the `d1ebfef` copyfit variants. Do not
  bring back its deterministic Russian synonym compactor: it changes meaning
  without a model.
- **No glossary (item 5).** Nothing stores names, gender, forms of address or
  recurring terms; each chapter request rediscovers them. Russian needs gender
  agreement in verbs and adjectives, so a wrong guess spreads across a
  chapter. Add a project-owned glossary (term, translation, gender, note),
  sent as reference data in both page and chapter requests and in the manual
  export. Offer to fill it from the first translated chapter.
- **Chapter mode is blind (item 7).** `ChapterTranslation::from_input` never
  attaches images, while page mode does. For vision models, attach downscaled
  page images within the existing token budget, and fall back to text only
  when the estimate is over it.
- **Profanity and register (item 3a).** The system prompt asks to preserve
  "emotional tone" (`koharu-translator/src/prompt.rs:277`) but never says to
  keep swearing, insults and slang at the same strength, and hosted LLMs
  tend to soften or drop them. Nothing detects it: the Gemini adapter sends
  no safety settings and no adapter treats a content-filter finish as its own
  error. Plan, cheapest first:
  1. State it in the prompt: keep profanity, insults, slurs used by
     characters and crude slang at the source's strength; never censor,
     soften, omit or add euphemisms.
  2. Mark suspicious segments deterministically: a small per-language list
     of source markers (for Japanese, for example `くそ`, `ちくしょう`,
     `てめえ`, `死ね`, `殺す`, `ぶっ`), a translation much shorter than its
     source relative to the language pair, an empty or refused output, or a
     content-filter finish.
  3. For marked segments only, translate the same segment through DeepL or
     Google Cloud Translation, which are already providers here with keys in
     the keychain. Hosted machine translation fails immediately, per
     `AGENTS.md`; if it fails, flag the segment without the comparison.
  4. Decide with JEV (text only fits it well): a `choice` of "LLM keeps the
     intensity", "machine translation keeps it, LLM softened", "both
     softened", plus the text of both. Raise a review flag that shows the
     machine translation next to the LLM line. Never replace the LLM line
     automatically: machine translation is usually worse at tone and
     context. Without JEV, flag when the machine translation contains a
     profanity marker of the target language and the LLM line does not.
  Effort M; the compare step costs one DeepL or Google call per marked line.
- **Translation QA (item 10).** JEV fits well here because the check is
  text-only: `noul` "meaning preserved", "gender and address match the
  glossary", "no untranslated source text". Flag, do not rewrite.
- **One request per chapter.** A single segment-count error fails the whole
  chapter. If this happens in practice, split large chapters into page groups
  that share the glossary, through the same prompt path. Measure first.

## Cleaning (inpainting)

Current state: local LaMa, AOT, FLUX.2 Klein or RORem, tiled at 512 px with
128 px context, with flat bubbles filled without inference
(`stages/inpainting.rs:652`, `:829`). API cleanup goes through Fal,
OpenRouter or Replicate, optionally with a preview before sending.

- **Full-page replacement by default (item 6).** API cleanup defaults to
  `FullPage` (`config.rs:663`): the generated image replaces every pixel of
  the page, so linework, tones and resolution outside the text can drift.
  Default to compositing only inside a dilated text and SFX mask. Keep
  full-page as an explicit choice for pages where the mask misses stylized
  SFX.
- **No fallback (item 6).** A policy rejection or a result with leftover
  lettering simply fails or is committed. Bring back the `d1ebfef` chain
  (`run_cloud_inpaint_with_fallback`): a deterministic residual-ink check
  inside the mask, then a retry on tiles around what is left, then local
  LaMa. Do not resubmit a paid job automatically when its submission state is
  unknown (`stages/fal.rs` already refuses this).
- **Routing per region (item 8).** Every masked region of a page goes to the
  same backend. Flat regions are already filled; route plain backgrounds to
  LaMa and only textured or SFX-over-art regions to the cloud as crops.
- **Leftover compatibility shim.** `normalize_cleanup_prompt` rewrites three
  old default prompts on load (`stages/fal.rs:26`). `AGENTS.md` forbids
  backward compatibility; remove it in a small cleanup PR.

## Typesetting

The renderer already auto-fits font size, balances lines inside the balloon
contour and hyphenates as a last resort. The missing parts are item 4
(overflow is never surfaced) and font choice: typography is inferred from
colors and stroke only (`stages/detection.rs:620`), so SFX and shouted lines
get the same font as dialogue. The unused `font_detector` model could map a
region to a font class; treat it like item 12 (compare first, issue first).

## Final check

There is no final stage. Before export, nothing tells the user which pages
still have problems. Add one chapter-level check (item 13) that reads the
scene and the prepared frame and lists, per page, with a jump to each region:

- open review flags from OCR, translation, profanity, overflow and cleaning;
- empty translations and translations that still contain source-script
  characters (kana or kanji in Russian text);
- `TextOverflow` and `TextBelowReadableSize` from the render diagnostics;
- text regions with no cleanup under them and leftover ink inside cleaned
  masks (the same residual-ink check as item 6);
- regions the missed-text check (item 9) found.

Everything above is deterministic and free. As an option, a vision LLM can
look at each rendered page next to the source page and answer a fixed list
(untranslated visible text, text over art, broken cleanup) through structured
output; JEV cannot do this part because it cannot see images. Show the result
as a checklist, not as a blocking error: export stays possible. This reuses
the review flags and adds no second job kind.
