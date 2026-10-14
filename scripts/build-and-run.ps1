#requires -Version 7.0
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$targetDirectory = Join-Path $repoRoot 'target'
$executable = Join-Path $targetDirectory 'debug\koharu.exe'
$debugConfig = Join-Path $repoRoot 'crates\koharu\tauri.debug.conf.json'
$previousTargetDirectory = $env:CARGO_TARGET_DIR

function Invoke-BuildStep {
    param([string]$Title, [string]$Program, [string[]]$Arguments)
    Write-Host "`n$Title" -ForegroundColor Cyan
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed (exit $LASTEXITCODE). The application was not launched."
    }
}

Push-Location -LiteralPath $repoRoot
try {
    foreach ($program in @('cargo', 'bun')) {
        if (-not (Get-Command $program -ErrorAction SilentlyContinue)) {
            throw "$program is not available. See the prerequisites in README.md."
        }
    }
    Invoke-BuildStep 'Checking Tauri CLI (install with bun install if missing)' 'cargo' @('tauri', '--version')

    # Only this checkout's executable may be closed; other installed copies are independent.
    $running = @(Get-Process -Name koharu -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $executable })
    foreach ($process in $running | Where-Object { $_.MainWindowHandle -ne 0 }) {
        Write-Host "Closing the previous Koharu window (PID $($process.Id))..."
        if (-not $process.CloseMainWindow()) {
            throw 'Close the previous Koharu window and run this script again.'
        }
    }
    foreach ($process in $running) {
        if (-not $process.WaitForExit(15000)) {
            throw 'Koharu is still running. Finish closing it and run this script again.'
        }
    }

    # Keep the output location explicit so the launched file is always the one just built.
    $env:CARGO_TARGET_DIR = $targetDirectory
    Invoke-BuildStep 'Installing locked frontend dependencies' 'bun' @('install', '--frozen-lockfile', '--ignore-scripts')
    Invoke-BuildStep 'Generating the native/frontend protocol' 'cargo' @('run', '-p', 'koharu-app', '--bin', 'generate', '--locked')
    Invoke-BuildStep 'Building debug WASM, frontend and desktop application' 'cargo' @(
        'tauri', 'build', '--debug', '--no-bundle', '--config', $debugConfig, '--', '--locked'
    )

    $built = Get-Item -LiteralPath $executable
    Write-Host "`nBuilt: $($built.FullName)" -ForegroundColor Green
    Write-Host "Executable timestamp: $($built.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    $launched = Start-Process -FilePath $executable -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
    Write-Host "Launched Koharu (PID $($launched.Id))."
} catch {
    Write-Host "`n$($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    $env:CARGO_TARGET_DIR = $previousTargetDirectory
    Pop-Location
}
