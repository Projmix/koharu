# Replicate image editing implementation plan

**Goal:** Connect Replicate FLUX.2 Dev to automatic cleanup and restoration, let the user enter their token, and build the debug app for their own trial.

**Design:** Extend the current remote image-edit boundary. Replicate owns its model ID, payload and asynchronous prediction lifecycle. Reuse the existing scene composition, prompt review and layer application. The user's existing request authorizes implementation in this checkout; preserve all other WIP and make no paid calls or Git commits.

**Interfaces:** `InpaintingProvider::Replicate` serializes as `replicate`; `ProviderPreferences.replicate` is `CredentialInput` stored under secret key `replicate`. The catalog returns `black-forest-labs/flux-2-dev`, named `FLUX.2 Dev Edit`. Existing prompt/apply-mode configuration stays unchanged. Provider payload uses `disable_safety_checker: true`, `go_fast: false`, PNG and `match_input_image`, as discussed with the user.

- [x] Native adapter and integration: token loading; PNG input; asynchronous prediction; cancellation; bounded retries of reads; no replay of paid creation; constrained/sanitized errors; output dimension preservation. Focused local HTTP tests and config roundtrip tests.
- [x] Frontend: Replicate token entry/save/clear; provider/model selection in cleanup/restoration and toolbar; accurate preview provider name; concise provider description; all consumers/fixtures updated. Focused component tests.
- [x] Regenerate the protocol from Rust, run relevant debug checks, independent read-only review of the new integration, and build/start via the existing script.

**Results:** 11 focused Rust tests and 4 focused frontend tests passed; TypeScript, targeted formatting/lint checks and protocol generation passed. Independent review found no confirmed P1/P2 issues. The debug build completed and the launched process was verified to use the newly built executable. Live API generation remains for the user's token and trial.

**Verification constraints:** No live paid generation, no E2E tests, no full test suite, no release build. UI and adapter work have disjoint file ownership; protocol generation and Rust builds are controlled by the root agent.
