# Image edit preview implementation plan

User scope: inspect the exact image sent by page API cleanup and rectangle Restore; edit only the prompt in a modal before sending. No image editor or layer picker.

## Design

- Preserve existing composition: page API cleanup uses the original and visible Paint layers; rectangle Restore also includes visible Cleanup layers, in scene order, and crops to the selection. Translated text and editor decorations are excluded.
- Add `pipeline.inpainting.preview_before_send`, initially false. Desktop requests attach an asynchronous reviewer when enabled; headless requests stay unattended.
- Pause the actual API branch after composition, cropping and provider prompt preparation. Show those pixels and that prompt. Apply an edited prompt to this request only.
- Native state owns pending previews and one-shot answers. The existing startup subscription carries the pending list; a command resolves a preview. Cancellation stops the job, including chapter processing, without sending. Stop/project replacement removes pending previews. Dropped review futures remove their pending entry.
- The modal displays page, provider/model, dimensions, the outgoing image, an editable prompt, Send and Cancel. Queue entries have distinct IDs so drafts cannot leak between requests.

## Work and verification

- [x] Pipeline review boundary, persistent option, layer/prompt/cancellation tests. Owner: preview_pipeline. Eight focused tests passed; one alpha expectation was corrected and that test rerun separately.
- [x] Desktop pending-preview lifecycle, IPC subscription and resolver, focused lifecycle tests. Owner: root. Seven focused tests passed.
- [x] Settings switch, global modal, subscription/store and focused UI tests; nine locales. Owner: preview_ui. Eight modal tests, two runtime tests, one persistence test, TypeScript/lint/format checks passed. Existing unrelated locale schema gaps remain.
- [x] Generate protocol from Rust; run focused debug checks and build; inspect actual desktop preview without sending a paid request. Debug EXE built 2026-09-13 15:09:04; actual Restore preview showed a loaded 128×96 PNG, OpenRouter MAI-Image-2.6 and the configured prompt. Editing the draft and cancelling closed the modal without changing project revision/page or the stored prompt. Preview mode left enabled. Read-only app lifecycle review found no P1/P2 issues.

No commits or unrelated WIP cleanup; no full suites or end-to-end test runs.
