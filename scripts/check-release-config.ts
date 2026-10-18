import { readFile } from 'node:fs/promises'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const config = JSON.parse(await readFile(path.join(root, 'crates/koharu/tauri.conf.json'), 'utf8'))
const repository = process.env.GITHUB_REPOSITORY
if (repository !== 'Projmix/koharu') {
  throw new Error('This release workflow belongs to Projmix/koharu.')
}
const expected = `https://github.com/${repository}/releases/latest/download/latest.json`
if (
  config.plugins.updater.endpoints.length !== 1 ||
  config.plugins.updater.endpoints[0] !== expected
) {
  throw new Error('Updater endpoint must point to this fork.')
}
if (!config.bundle.createUpdaterArtifacts || !config.plugins.updater.pubkey) {
  throw new Error(
    'Configure a fork signing public key and enable updater artifacts before releasing.',
  )
}
if (!process.env.TAURI_SIGNING_PRIVATE_KEY) {
  throw new Error('Add the fork Actions secret TAURI_SIGNING_PRIVATE_KEY before releasing.')
}
const cargo = Bun.TOML.parse(await readFile(path.join(root, 'Cargo.toml'), 'utf8'))
if (process.env.GITHUB_REF_NAME?.replace(/^v/, '') !== cargo.workspace.package.version) {
  throw new Error('Release tag must match the workspace package version.')
}
console.log('Fork endpoint, release version, and signing configuration are ready.')
