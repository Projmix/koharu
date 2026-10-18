param(
    [string]$KeyDirectory = (Join-Path $env:USERPROFILE '.koharu/updater-signing')
)

$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$keyDirectoryPath = [IO.Path]::GetFullPath($KeyDirectory)
if ($keyDirectoryPath.Equals($repository, [StringComparison]::OrdinalIgnoreCase) -or
    $keyDirectoryPath.StartsWith($repository + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The signing key directory must be outside the repository.'
}
if (Test-Path -LiteralPath $keyDirectoryPath) {
    throw 'Choose a new dedicated key directory. Existing signing keys must not be overwritten.'
}

$configPath = Join-Path $repository 'crates/koharu/tauri.conf.json'
$configText = Get-Content -Raw -LiteralPath $configPath
$config = $configText | ConvertFrom-Json
if ($config.plugins.updater.pubkey) {
    throw 'An updater public key is already configured. Keep using its private key; this script does not rotate keys.'
}

New-Item -ItemType Directory -Path $keyDirectoryPath | Out-Null
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
& icacls.exe $keyDirectoryPath '/inheritance:r' '/grant:r' "*${sid}:(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Could not restrict access to the key directory.' }

$privateKeyPath = Join-Path $keyDirectoryPath 'koharu.key'
# --write-keys prints paths only. The unencrypted private key stays in the restricted directory.
& cargo tauri signer generate --ci --write-keys $privateKeyPath 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Tauri signing key generation failed.' }
$publicKey = (Get-Content -Raw -LiteralPath ($privateKeyPath + '.pub')).Trim()
$decoded = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($publicKey))
if ($decoded -notmatch '^untrusted comment: minisign public key:') {
    throw 'The generated file is not a Tauri updater public key.'
}
$updated = [regex]::Replace($configText, '("pubkey"\s*:\s*)""', ('$1"' + $publicKey + '"'))
if ($updated -eq $configText) { throw 'Could not update the empty public key in tauri.conf.json.' }
[IO.File]::WriteAllText($configPath, $updated, [Text.UTF8Encoding]::new($false))

Write-Output "Configured the fork public key. Back up the private key at: $privateKeyPath"
Write-Output 'Add the private key file contents as Projmix/koharu Actions secret TAURI_SIGNING_PRIVATE_KEY. Do not commit or print the file contents.'
Write-Output 'This key has no password; leave TAURI_SIGNING_PRIVATE_KEY_PASSWORD unset.'
