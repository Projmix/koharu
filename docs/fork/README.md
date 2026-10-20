# Projmix/koharu fork: agent guide

This file describes how the fork differs from upstream
[koharu-rs/koharu](https://github.com/koharu-rs/koharu), what state `main` is
in, and where JEV fits in the pipeline. Read it together with the root
`AGENTS.md`, whose rules still apply. Update this file when fork behavior or
the plan changes; it was last checked on 2026-10-20.

## Product direction

- The fork is API-first, not local-first. Hosted vision/LLM models are the
  preferred path for OCR, translation and inpainting because they are faster
  and more convenient. Local models stay as a fallback and as a cheap first
  pass, not as the priority.
- The owner wants a single long-lived branch, `main`. Feature work branches
  off `main` and merges back; no parallel long-running branches.
- The owner writes in Russian. User-facing summaries go in Russian; code,
  commits and this file stay in English.

## Branch and upstream state

On 2026-10-18 `main` was rebuilt (PR #1) from `codex/upstream-editor-ocr`
plus a real `git merge` of upstream tag `0.83.5`, then of upstream `main` at
`85d6de3` (unreleased work after 0.83.5). Git tracks the upstream base, so the
next sync is `git fetch upstream main` (or a tag) then `git merge`. Upstream
removed `koharu-metrics` telemetry; fork code must not emit `koharu_metrics`
events. The Wu Chinese locale reuses Simplified Chinese text for fork-only
strings until someone translates them.

The previous `main` commit `d1ebfef` (JEV, Japanese OCR validation, copyfit
variants, cloud inpainting fallback chain) never compiled and is not in the
tree. It stays in history; bring its ideas back in separate changes, JEV
first, following the plan below. `git show d1ebfef -- <path>` shows the old
code.

Since the rebuild, work lands on `claude/project-thread-lbw4bo`; on
2026-10-20 it is more than 85 commits ahead of `main` and not merged yet.
The fork features below describe that branch.

Known gaps after the rebuild:

- `packages/bridge/src/protocol.ts` was merged by hand because the cloud
  environment cannot download CEF to run the generator. Regenerate it with
  `cargo run -p koharu-app --bin generate` on a desktop machine and commit
  any difference.
- `koharu-app` (the Tauri crate) was not compiled in that environment; the
  rest of the workspace was checked, linted and tested.

## Fork features

- Chapter translation export/import (`export_chapter_translation`,
  `import_chapter_translation`): the exact chapter request as JSON for an
  external chat, and import of its answer as a normal pipeline patch.
- Named project versions (`save_project_version` and friends).
- OCR method `local`, `api` or hybrid `local-api`. The hybrid method uses an
  internal `OcrPass { Local, Final }`: the scheduler runs the local pass on
  every page of the chapter, then the API pass, and only then chapter
  translation. The local pass reads every region with the chosen model and a
  second local reader (Manga OCR, or Hayai when Manga OCR is chosen); only
  regions where they disagree go to the API. Text the user edited is never
  overwritten.
- Scene markup is its own pipeline stage, `markup`, between OCR and
  Translation (config `pipeline.markup`; default model
  `google/gemini-3.7-flash` through OpenRouter at temperature 0). It runs
  when the user runs it, alone or with other stages: each page in project
  order, after that page's OCR, is sent to a vision model with numbered
  boxes drawn on it. When a run includes both, page translation waits for
  that page's markup and chapter translation waits for markup on every
  page; Translation never runs markup itself and uses whatever markup
  exists. The model only labels each text's kind (speech, thought,
  narration, sound-effect, sign, aside), speaker and the earlier text it
  continues; it never transcribes or translates. Results go to the scene as
  `SpeechMarkup`, the `Continues` relation, the project `CharacterList` and a
  per-page `PageMarkup` with the texts it saw, so a rerun skips unchanged
  pages. A refusal, filtered, invalid or failed answer records
  `PageMarkup` as missing and the run continues. User-authored
  markup and a user-edited character list are never overwritten. Chapter
  translation sends `speech_kind`, `speaker` {name, gender}, `continues` per
  segment and a top-level `characters` list, so the translator can use the
  speaker's gender for Russian verb and adjective agreement.
- Qwen-VL-OCR provider (`remote/qwen_ocr.rs`, DashScope) with its own
  endpoint and credential.
- Image-edit inpainting through Fal, OpenRouter and Replicate, with a
  preview and approval dialog before the result is committed.
- Manual translation, chapter translation presets and unit policy, paste of
  text blocks, page duplication.
- Fork updater against `Projmix/koharu` releases (`docs/fork-releases.md`).
- Research notes in `docs/research/` and plans in `docs/superpowers/plans/`.
- Automatic typesetting diverges from upstream to match the owner's
  hand-set Russian pages (numbers in [results.ru.md](results.ru.md#тайпинг)):
  text inside a balloon that detection marks as dialogue takes the speech
  kind font before scene markup; a word is split only when that buys at
  least 1.15x the whole-word size; a line never ends on a one-letter
  Cyrillic word, and a chain of them binds through to the next real word;
  balloon auto-fit trades up to 20% of size for fewer lone one- or
  two-letter inner lines and penalises a hyphen break that leaves such a
  tail. Detection reads text paint against a screentone's 5x5 mean shade,
  and rejects outline readings whose band is surrounded by its fill or
  whose "outline" is a glyph enclosed by its own ground. An automatic
  outline is capped at 0.1x the font size, and text inside a balloon gets
  none; a user-set outline is kept. Spaced sound-effect letters stack one
  per line. The Border row has a swap button for text and outline colours.


## JEV

### What it was

`d1ebfef` added `koharu-pipeline/src/stages/jev.rs`: a `JevClient` posting
batches to a "TypeSafe System One" endpoint
(`https://api.typesafe.ai/v1/system-one/evaluate`) with three typed answer
primitives instead of free text:

- `Noul`: a probability for a yes/no question.
- `Choice<T>`: exactly one option from a closed list, with confidence.
- `Score`: a rubric score in `[0, 1]`.

It judged Japanese OCR reading order and retries (`ocr_validation_v1`) and
picked a copyfit translation variant (`copyfit_selection_batch_v1`), with a
deterministic offline fallback. The service's API contract was never
verified, and the client read keys from environment variables, which
`AGENTS.md` forbids for the desktop app. When reimplementing, keep the typed
primitives, take the key only from `koharu_secrets`, and verify the API
against its documentation.

### API facts

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

Implementation rules:

- Off by default. A settings toggle enables it; the key is stored only in
  `koharu_secrets` (OS keychain). No environment variable or project file.
- Confirm the TypeSafe base URL and the endpoint that will remain
  (`/v1/systemone` or `/v1/decisions`) from TypeSafe's own documentation
  before writing the client. A community Rust crate (`typesafe-rs` 0.1.0)
  exists; prefer a small `reqwest` client like the other providers, since a
  new dependency needs an issue first.
- When JEV is unreachable, fall back to "flag every disagreement" and report
  that the judge was offline.

### Planned OCR flow: local, then LLM, then JEV decides

Goal: better text than either OCR alone, and an explicit list of regions a
human must check (handwriting, stylized SFX, text neither model read).

1. **Local pass.** Local OCR on every region of every page in the chapter
   (fast, free). This is the existing `OcrPass::Local` and chapter
   barrier; add a deterministic quality score per region.
2. **LLM pass.** Send each crop to a vision LLM (OpenRouter, Qwen-VL-OCR,
   and so on). Send the image as the source of truth and the local draft as
   data, so the model corrects rather than invents. Start with every region
   so accuracy can be measured; later, send only regions whose local score
   is below a calibrated threshold to save cost.
3. **JEV decides.** One batch per page with both candidates, both inspector
   scores, geometry and column count. A new task (for example
   `ocr_candidate_selection_v1`) returns:
   - `Choice { local, api, needs_review }` for the text to keep;
   - `Noul` for "the kept text matches the image";
   - `Choice` of reason when review is needed: `handwritten`,
     `models_disagree`, `low_confidence`, `empty`.
   Japanese reading-order and retry checks from `d1ebfef` can return here
   as deterministic inputs to the judge.
4. **Flag for manual review.** When the verdict is `needs_review` or the
   probability is below the threshold, keep the best candidate as the text
   but record a review flag on the region in the scene (a component owned
   by Scene, committed in the same patch, so it survives save, undo and
   versions). The editor lists flagged regions and jumps to them; a user
   edit (`Origin::User`) clears the flag. Translation still runs, but
   flagged regions are visibly marked so they are checked before export.

Rules to follow while building it:

- If JEV is unreachable, do not silently treat everything as accepted. A
  fallback verdict should flag disagreements, and the run should report
  that the judge was offline.
- Keep a single OCR stage with internal passes; do not add a second job kind
  (`AGENTS.md`).
- Calibrate thresholds on a small labeled set of the owner's manga
  (character error rate, misses, furigana, SFX, cost, and time spent fixing
  by hand). A model's stated confidence is not a probability.

### Other places JEV could help later

- Translation QA: `Noul` "meaning, gender and tone preserved" for a block,
  so suspicious translations are flagged the same way as OCR.
- Inpainting QA: a judge on the image-edit preview to decide when a
  cloud result needs a retry or human review.

## Removed

- `packages/koharu/lib/web-tauri-shim.ts`: a 1.2k-line browser mock that
  faked `window.__TAURI_INTERNALS__` with sample data (a "Blue Archive"
  project, fake fonts and models) so the UI could open in a plain browser,
  probably for an online preview tool. It never touched real projects or
  models, and the desktop app does not need it. Its web-hosting config
  (Next.js `standalone` output, npm scripts, `metadata.json`, a committed
  wasm-bindgen file) was removed with it; the static export that Tauri loads
  is restored.
