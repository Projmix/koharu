import { spawnSync } from 'node:child_process'
import { createHash, createPublicKey, verify } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const config = JSON.parse(await readFile(path.join(root, 'crates/koharu/tauri.conf.json'), 'utf8'))
const directory = await mkdtemp(path.join(tmpdir(), 'koharu-signing-check-'))
try {
  const challenge = Buffer.from('Projmix/koharu updater signing check\n')
  const challengePath = path.join(directory, 'challenge.txt')
  await writeFile(challengePath, challenge)
  // Capture signer output: neither CI logs nor errors should include secret environment values.
  const result = spawnSync('cargo', ['tauri', 'signer', 'sign', challengePath], {
    cwd: root,
    stdio: 'pipe',
    timeout: 30000,
    env: {
      ...process.env,
      TAURI_SIGNING_PRIVATE_KEY_PASSWORD: process.env.TAURI_SIGNING_PRIVATE_KEY_PASSWORD ?? '',
    },
  })
  if (result.status !== 0) {
    throw new Error('Tauri could not sign the check file. Verify the private key and password.')
  }

  const publicText = Buffer.from(config.plugins.updater.pubkey, 'base64').toString('utf8')
  const signatureText = Buffer.from(
    await readFile(challengePath + '.sig', 'utf8'),
    'base64',
  ).toString('utf8')
  const publicPacket = Buffer.from(publicText.split('\n')[1] ?? '', 'base64')
  const signaturePacket = Buffer.from(signatureText.split('\n')[1] ?? '', 'base64')
  // Tauri's pinned minisign format: algorithm (2), key ID (8), Ed25519 key/signature.
  if (
    publicPacket.length !== 42 ||
    signaturePacket.length !== 74 ||
    signaturePacket.subarray(0, 2).toString() !== 'ED' ||
    !publicPacket.subarray(2, 10).equals(signaturePacket.subarray(2, 10))
  ) {
    throw new Error('The updater public key does not match the signing key or signature format.')
  }
  const publicKey = createPublicKey({
    key: Buffer.concat([Buffer.from('302a300506032b6570032100', 'hex'), publicPacket.subarray(10)]),
    format: 'der',
    type: 'spki',
  })
  const digest = createHash('blake2b512').update(challenge).digest()
  if (!verify(null, digest, publicKey, signaturePacket.subarray(10))) {
    throw new Error('The configured updater public key does not verify the signing key.')
  }
  console.log('Tauri updater signature verified against the configured fork public key.')
} finally {
  if (
    path.dirname(path.resolve(directory)) !== path.resolve(tmpdir()) ||
    !path.basename(directory).startsWith('koharu-signing-check-')
  ) {
    throw new Error('Refusing to remove an unexpected signing check directory.')
  }
  await rm(directory, { recursive: true, force: true })
}
