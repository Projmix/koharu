# Projmix/koharu fork: agent guide

This file describes how the fork differs from upstream
[koharu-rs/koharu](https://github.com/koharu-rs/koharu), what state `main` is
in, and where JEV fits in the pipeline. Read it together with the root
`AGENTS.md`, whose rules still apply. Update this file when fork behavior or
the plan changes; it was last checked on 2026-10-18 against `main` at
`755ecb4`.

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

| | `main` | `codex/upstream-editor-ocr` |
| --- | --- | --- |
| Upstream base | Files copied from about 0.83.5, **not a git merge** (commit `755ecb4`) | Real merge of upstream 0.82.1 (`b07ef1e`) |
| Fork features | Chapter export/import, named versions, JEV, cloud inpainting rework | Hybrid local+API OCR, Qwen OCR, Replicate image edit, image-edit preview, manual translation, fork updater, editor improvements (`3f254b7`) |
| Shared base | `a7714ce` | `a7714ce` |

Consequences for anyone touching sync work:

- Git does not know that `main` contains upstream 0.83.x, because the sync
  was a file copy. A plain `git merge` of upstream or of the codex branch
  replays the same upstream changes from both sides (about 60 conflicts).
  Cherry-picking only the fork commit `3f254b7` onto `main` leaves 21
  conflicted files, concentrated in `stages/ocr.rs` and
  `stages/inpainting.rs`, where the two branches implement OCR and cloud
  inpainting differently.
- The copy is also partial. Root `Cargo.toml` pins Tauri to git rev
  `1991873f…` while `packages/bridge` uses `@tauri-apps/api` 3.0.0-alpha and
  `packages/koharu` still uses 2.x, so the TypeScript build reports
  `Channel` type mismatches between the two copies.
- To make future upstream syncs cheap, record the next sync as a real merge
  of an upstream tag (`git fetch upstream tag <x>` then `git merge`), so git
  tracks the base.

## Known problems on `main`

These were observed, not inferred:

- `cargo check -p koharu-pipeline` fails in `koharu-storage`
  (`session.rs` passes a `String` to `Error::Task`, which now wraps
  `anyhow::Error`). The fork's named-versions code was not adapted to the
  upstream error type.
- `stages/translation.rs` calls `JevClient::from_config()` and
  `JevClient::deterministic_copyfit_verdict`, reads `verdict.choice` and
  writes `judge_input.region_index`. None of these exist in `stages/jev.rs`
  (the verdict field is `selected_variant`; the copyfit fallback is a free
  function). The JEV commit has not compiled since it landed.
- `Cargo.lock` and `bun.lock` are stale: `cargo check` adds dependencies to
  the lock, and `bun install --frozen-lockfile` (what CI runs) fails.
- Three tests in `packages/koharu/tests/components/editor-components.test.tsx`
  fail on `main` (layers panel, expanded processing controls, settings
  shortcuts).
- `JevClient::api_key` falls back to the `TYPESAFE_API_KEY` / `JEV_API_KEY`
  environment variables. `AGENTS.md` forbids environment fallbacks for
  desktop API keys; keys must come only from the OS credential store
  (`koharu_secrets`).

## Fork features on `main`

### Chapter translation export and import

Commands `export_chapter_translation` / `import_chapter_translation`
(`koharu-app/src/commands/processing.rs`). Export writes the exact chapter
translation request as JSON so it can be run in any external chat or batch
tool. Import accepts either the fork's `{"translations": …}` result or a raw
OpenAI-style `{"choices": …}` response, tolerates the model echoing the
request, and commits the result as a normal pipeline patch. Both refuse to
run while processing is active.

### Named project versions

Commands `save_project_version`, `list_project_versions`,
`restore_project_version`, `delete_project_version`
(`commands/versions.rs`, storage in `koharu-storage/src/session.rs`, UI in
`ProjectVersionsDialog.tsx`). A version is a named snapshot of the scene
document.

### OCR: local or API, with deterministic validation

`OcrConfig { method: Local | Api, local_model, api }` in
`koharu-pipeline/src/config.rs`.

- `Local` runs the selected local model (manga-ocr, baberu, hayai,
  PaddleOCR-VL). `Api` sends each crop to a vision LLM through the
  translator provider layer.
- For Japanese, every result goes through `validate_and_recover_targets`
  (`stages/ocr.rs`):
  1. `inspect_japanese_ocr` scores the text deterministically (valid-script
     ratio, kinsoku line-start rules, cross-column grammar to detect
     inverted column order, furigana noise).
  2. JEV (`judge_ocr_batch`) picks a reading order (RTL / LTR / corrupted),
     gives a probability that the text is valid Japanese, and picks a retry
     strategy.
  3. If recovery is needed: retry on a padded crop with furigana
     suppressed, then column re-slicing and an optional orientation flip,
     then the alternate local OCR model (`alternate_local_ocr_model`). Each
     step replaces the text only if the inspector score improves.
  4. `reconcile_bubble_sibling_reading_order` fixes the order of sibling
     vertical regions inside one bubble.
- The final inspector score is stored as the OCR result `confidence`.
  Nothing marks a region as "needs human review" yet.

### Translation: copyfit variants

The translation prompt asks for `full`, `compact` and `ultra_short`
variants (`TranslationVariants` in `koharu-translator/src/backend.rs`). The
renderer measures each one in its bubble (`CopyfitDiagnostics`,
`koharu-renderer`; `bubble.rs` adds multi-lobe bubble splitting), and JEV
`judge_copyfitting_batch` picks the variant or asks for regeneration. This
path does not compile today (see above).

### Inpainting: cloud first, with a fallback chain

`InpaintingConfig { method: Local | Api, api: { provider: Fal | OpenRouter,
model, prompt, apply_mode } }`. `run_cloud_inpaint_with_fallback`
(`stages/inpainting.rs`) runs:

1. a full-page cloud edit; residual ink or border damage
   (`inspect_inpaint_diff`) or a content-policy/422 rejection sends it to
2. tiled crop edits on the remaining mask, then
3. the other cloud provider if it has a stored credential, then
4. local LaMa on what is still left, then
5. a deterministic harmonic fill.

Keys come from `koharu_secrets` (`fal`, `openrouter`).

## Unmerged work on `codex/upstream-editor-ocr`

Commit `3f254b7` carries fork work that `main` does not have:

- Hybrid OCR method `local-api`: an internal `OcrPass { Local, Final }`; the
  scheduler runs the local pass on every page of the chapter, then the API
  pass, and only then chapter translation. Text the user edited is never
  overwritten.
- A dedicated Qwen-VL-OCR provider (`remote/qwen_ocr.rs`, DashScope) with its
  own endpoint and credential.
- Replicate image-edit inpainting, an image-edit preview/approval dialog
  (`review.rs`, `ImageEditPreviewDialog.tsx`), manual translation, paste of
  text blocks, page duplication.
- Fork updater against `Projmix/koharu` releases, plus an informational
  notice about newer upstream releases (`docs/fork-releases.md` on that
  branch).
- Research notes in Russian: `docs/research/2026-09-12-fork-review.md`,
  `docs/research/2026-09-12-japanese-ocr.md`.

Its upstream part (0.82.1) is older than what `main` already copied, so the
only thing worth bringing over is `3f254b7`. Its OCR pass ordering and
`main`'s JEV validation are the two halves of the planned OCR flow below,
which is why they need a deliberate merge rather than a mechanical one.

## JEV

### What it is in code

`koharu-pipeline/src/stages/jev.rs`. `JevClient` posts batches to a
"TypeSafe System One" evaluation endpoint (default
`https://api.typesafe.ai/v1/system-one/evaluate`, override
`TYPESAFE_API_URL`, 6 s timeout, bearer key from secret `typesafe` or
`jev`). The answers use three typed primitives instead of free text:

- `Noul`: a probability for a yes/no question.
- `Choice<T>`: exactly one option from a closed list, with confidence.
- `Score`: a rubric score in `[0, 1]`.

Every call has a deterministic offline fallback
(`deterministic_ocr_verdict`, `deterministic_copyfit_verdict`). With no key,
a network error or a wrong-length answer, the fallback result is used
silently. The real service's API contract has not been verified against its
documentation in this repo; treat the request shape as the fork's
assumption.

### Where JEV is used now

| Stage | Task | Question JEV answers |
| --- | --- | --- |
| OCR (Japanese only) | `ocr_validation_v1` | Which reading order is right, is this valid Japanese, which retry to try |
| Translation | `copyfit_selection_batch_v1` | Which of full/compact/ultra_short/lobe-split fits the bubble, is meaning preserved, readability score |

### Planned OCR flow: local, then LLM, then JEV decides

Goal: better text than either OCR alone, and an explicit list of regions a
human must check (handwriting, stylized SFX, text neither model read).

1. **Local pass.** Local OCR on every region of every page in the chapter
   (fast, free). Keep the text and the inspector score. Reuse the codex
   branch's `OcrPass::Local` and chapter barrier.
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
   The existing reading-order and retry logic still runs on the kept
   candidate.
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
- Inpainting QA: a judge on top of `inspect_inpaint_diff` to decide when a
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
