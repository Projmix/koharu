# Fork releases and upstream notices

Installable updates come only from `Projmix/koharu` through
`https://github.com/Projmix/koharu/releases/latest/download/latest.json`.
The updater verifies their signatures against this fork's public key in
`crates/koharu/tauri.conf.json`.

The original `koharu-rs/koharu` channel is informational. Its latest stable release
is compared with integrated upstream version **0.83.5**, commit
`52bb4f55e2266f72c1523562a583c827ec191ceb`, independently of the fork's version.
Update the baseline in `crates/koharu-app/src/commands/updates.rs` only after
integrating a newer upstream release. The notice links to the original release,
can be dismissed for that version, and cannot install it. A fork update takes
priority; the original-release notice follows after the fork dialog is closed.
Both checks run asynchronously with an eight-second request timeout; network
errors do not block application startup or the other channel.

## Signing setup

The existing configured public key must be preserved for future releases. Keep
the matching private key outside the repository, back it up, and store its file
contents as the `TAURI_SIGNING_PRIVATE_KEY` Actions secret in **Projmix/koharu**.
Use `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` only when that key has a password.
Do not replace the public key for each release: installed applications trust the
key embedded in the version they currently run.

For an initial setup with an empty public key, Windows maintainers can run:

```powershell
pwsh -File scripts/setup-updater-signing.ps1
```

The script creates an unencrypted private key in a new directory under the user
profile, restricts that directory to the current user and SYSTEM, and writes only
the public key to the tracked Tauri configuration. It refuses to overwrite an
existing key directory or rotate an already configured key. The private key
must never be printed, committed, or included in a release artifact.

To check an existing local key without building or publishing a release:

```powershell
$env:TAURI_SIGNING_PRIVATE_KEY_PATH = 'C:\outside-the-repository\koharu.key'
bun scripts/check-updater-signing.ts
Remove-Item Env:TAURI_SIGNING_PRIVATE_KEY_PATH
```

This signs a temporary challenge with Tauri and verifies its signature against
the configured public key, then removes the temporary files. The check prints
no private key contents. In CI the same check uses the Actions signing secret.

## Publishing prerequisites

1. Add the matching signing secret to the fork's Actions settings.
2. Set the workspace package version to a new version greater than installed
   fork releases, and create the matching tag. The release workflow checks the
   repository, endpoint, tag/version, key configuration, and a real signature
   before building artifacts.
3. When publication is intended, push that tag to `Projmix/koharu`. The existing
   tag-triggered release workflow builds the platform installers and signed
   updater artifacts and publishes `latest.json` through `tauri-action`.
4. Verify that the published manifest references the fork's artifacts. Until
   the first release includes `latest.json`, the fork check has no update to
   offer. An application built before the fork endpoint/key change needs one
   manual installation of a fork build to join this channel.

Windows installers do not use upstream Azure credentials. macOS bundles use ad
hoc signing without upstream Apple certificates or notarization. Operating-system
publisher trust is separate from the updater signature and may require a
maintainer's own certificates for a publicly distributed release. The workflow
does not submit releases to the original `mayocream.koharu` winget package.
