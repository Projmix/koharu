# Hybrid OCR and fork updates

**Goal:** three OCR modes with independent local/API selection, all chapter local work before API work, dedicated Qwen OCR, fork releases plus upstream notices.

**Spec:** user instruction in the current task dated 2026-09-12. No candidate comparison or accept/reject UI. Existing OpenAI-compatible local servers remain supported. Qwen-VL-OCR 2025-11-20 uses the built-in multi_lan task for direct OCR and its documented image + custom user prompt (without ocr_options) for a local draft. No backward compatibility adapters; update all repository consumers. Preserve existing WIP; no release publication requested.

**Architecture:** OCR gets an internal local/final pass, scheduled separately under the existing public OCR stage. Local results are committed before the final API pass; the project scheduler enforces a chapter-wide barrier. Dedicated Qwen OCR owns its endpoint, request and model default. Fork updater uses its own manifest/signing configuration; upstream releases are informational and compared with a pinned integrated upstream baseline.

## Tasks

- [x] OCR orchestration: scheduler and StageInput pass, processor local/final lifecycle and cancellation; test chapter barrier, translation ordering, local failure/stop and unchanged ordinary API/local modes.
- [x] Qwen OCR: official DashScope adapter, configuration/catalog with a separate credential, actual OCR capability filtering for generic/local API models; tests for payload/response and UI dual selectors. Keep user selections unless choosing a new provider.
- [x] Fork updates: own endpoint and workable signing/release setup, upstream release notice with dismissal and canonical links, focused updater tests. Document any required unpublished key or release prerequisite accurately.
- [x] Regenerate protocol, focused debug Rust tests, TypeScript and affected UI tests. Review changes and update the earlier audit to remove the rejected comparison UI.

Rulings: continue in existing codex/upstream-editor-ocr branch because it owns this task and contains preserved user WIP. Keep changes uncommitted rather than mixing user work into feature commits. Do not publish a release or mutate remote secrets without identifying the actual existing setup first.

## Result

Completed locally on the existing branch, preserving the user's WIP. Qwen contract tests: 8 passed; final OCR tests: 25 passed; scheduler: 7 passed; native upstream version check: 1 passed; targeted OCR/translation UI: 7 passed; final updater UI: 7 passed. Debug protocol generation, TypeScript, affected UI lint/format checks and updater signature verification passed. Review found one mixed-page overwrite bug, fixed with a per-content completion predicate and a regression test; follow-up review confirmed resolution.

The matching private signing key is outside the repository. Publishing still requires its Actions secret and a first fork release with latest.json; no remote secrets or releases were changed. Live paid OCR accuracy and the final desktop window were not tested. See docs/research/2026-09-12-fork-review.md and docs/fork-releases.md.
