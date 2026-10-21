#Requires -Version 7
<#
.SYNOPSIS
Moves Koharu projects, settings and API keys to another Windows computer.

.DESCRIPTION
Export packs into one .zip:
  - every project from Documents\Koharu;
  - the settings folder %USERPROFILE%\.koharu (config.toml and the rest);
  - every Koharu credential from the Windows Credential Manager, encrypted
    with a password you choose (AES-256-GCM, PBKDF2-SHA256);
  - with -WithModels, the downloaded model and runtime cache, so the other
    computer does not download it again.
Import unpacks such a .zip on the other computer. A project that already
exists there is skipped, never overwritten; an existing settings folder is
kept as .koharu.bak-<time>.

Close Koharu on both computers before running this.

.EXAMPLE
pwsh -File scripts\transfer-koharu.ps1 export -Path D:\koharu-transfer.zip

.EXAMPLE
pwsh -File scripts\transfer-koharu.ps1 import -Path D:\koharu-transfer.zip
#>
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('export', 'import')]
    [string] $Action,
    [Parameter(Mandatory)]
    [string] $Path,
    [switch] $WithModels
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Documents = [Environment]::GetFolderPath('MyDocuments')
$ProjectsDir = Join-Path $Documents 'Koharu'
$SettingsDir = Join-Path $env:USERPROFILE '.koharu'
$ModelsDir = Join-Path $env:LOCALAPPDATA 'koharu\packages'
$Path = [IO.Path]::GetFullPath($Path)

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class KoharuCredentials
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct CREDENTIAL
    {
        public int Flags;
        public int Type;
        public string TargetName;
        public string Comment;
        public long LastWritten;
        public int CredentialBlobSize;
        public IntPtr CredentialBlob;
        public int Persist;
        public int AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool CredEnumerateW(string filter, int flags, out int count, out IntPtr credentials);

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool CredWriteW(ref CREDENTIAL credential, int flags);

    [DllImport("advapi32.dll")]
    static extern void CredFree(IntPtr buffer);

    public class Entry
    {
        public int Type;
        public string TargetName;
        public string Comment;
        public string TargetAlias;
        public string UserName;
        public int Persist;
        public string Blob;
    }

    public static List<Entry> Read(string marker)
    {
        var result = new List<Entry>();
        int count;
        IntPtr list;
        if (!CredEnumerateW(null, 0, out count, out list))
        {
            int error = Marshal.GetLastWin32Error();
            if (error == 1168) return result; // ERROR_NOT_FOUND: no credentials at all
            throw new System.ComponentModel.Win32Exception(error);
        }
        try
        {
            for (int i = 0; i < count; i++)
            {
                IntPtr item = Marshal.ReadIntPtr(list, i * IntPtr.Size);
                var credential = Marshal.PtrToStructure<CREDENTIAL>(item);
                if (credential.TargetName == null
                    || credential.TargetName.IndexOf(marker, StringComparison.OrdinalIgnoreCase) < 0)
                    continue;
                var blob = new byte[credential.CredentialBlobSize];
                if (blob.Length > 0) Marshal.Copy(credential.CredentialBlob, blob, 0, blob.Length);
                result.Add(new Entry
                {
                    Type = credential.Type,
                    TargetName = credential.TargetName,
                    Comment = credential.Comment,
                    TargetAlias = credential.TargetAlias,
                    UserName = credential.UserName,
                    Persist = credential.Persist,
                    Blob = Convert.ToBase64String(blob),
                });
            }
        }
        finally
        {
            CredFree(list);
        }
        return result;
    }

    public static void Write(Entry entry)
    {
        var blob = Convert.FromBase64String(entry.Blob);
        var buffer = Marshal.AllocHGlobal(Math.Max(blob.Length, 1));
        try
        {
            Marshal.Copy(blob, 0, buffer, blob.Length);
            var credential = new CREDENTIAL
            {
                Type = entry.Type,
                TargetName = entry.TargetName,
                Comment = entry.Comment,
                TargetAlias = entry.TargetAlias,
                UserName = entry.UserName,
                Persist = entry.Persist,
                CredentialBlobSize = blob.Length,
                CredentialBlob = buffer,
            };
            if (!CredWriteW(ref credential, 0))
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }
}
'@

function Read-Password([string] $Prompt) {
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    [Net.NetworkCredential]::new('', $secure).Password
}

function Get-Key([string] $Password, [byte[]] $Salt) {
    [Security.Cryptography.Rfc2898DeriveBytes]::Pbkdf2(
        [Text.Encoding]::UTF8.GetBytes($Password), $Salt, 600000,
        [Security.Cryptography.HashAlgorithmName]::SHA256, 32)
}

function Protect-Text([string] $Text, [string] $Password) {
    $salt = [Security.Cryptography.RandomNumberGenerator]::GetBytes(16)
    $nonce = [Security.Cryptography.RandomNumberGenerator]::GetBytes(12)
    $plain = [Text.Encoding]::UTF8.GetBytes($Text)
    $cipher = [byte[]]::new($plain.Length)
    $tag = [byte[]]::new(16)
    $aes = [Security.Cryptography.AesGcm]::new((Get-Key $Password $salt), 16)
    try { $aes.Encrypt($nonce, $plain, $cipher, $tag) } finally { $aes.Dispose() }
    [ordered]@{
        salt = [Convert]::ToBase64String($salt)
        nonce = [Convert]::ToBase64String($nonce)
        tag = [Convert]::ToBase64String($tag)
        data = [Convert]::ToBase64String($cipher)
    } | ConvertTo-Json
}

function Unprotect-Text([string] $Json, [string] $Password) {
    $box = $Json | ConvertFrom-Json
    $cipher = [Convert]::FromBase64String($box.data)
    $plain = [byte[]]::new($cipher.Length)
    $aes = [Security.Cryptography.AesGcm]::new(
        (Get-Key $Password ([Convert]::FromBase64String($box.salt))), 16)
    try {
        $aes.Decrypt([Convert]::FromBase64String($box.nonce), $cipher,
            [Convert]::FromBase64String($box.tag), $plain)
    } catch {
        throw 'Wrong password, or the keys file is damaged.'
    } finally { $aes.Dispose() }
    [Text.Encoding]::UTF8.GetString($plain)
}

function Assert-KoharuClosed {
    if (Get-Process -Name 'koharu*' -ErrorAction SilentlyContinue) {
        throw 'Koharu is running. Close it and run the script again.'
    }
}

function Copy-Folder([string] $From, [string] $To) {
    # robocopy keeps long paths and large files; codes below 8 are success.
    robocopy $From $To /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $From failed (robocopy code $LASTEXITCODE)." }
}

Assert-KoharuClosed
$stage = Join-Path ([IO.Path]::GetTempPath()) "koharu-transfer-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    if ($Action -eq 'export') {
        if (Test-Path $Path) { throw "$Path already exists. Choose another file name." }
        # Fail before copying anything when the archive cannot be written there
        # (the root of C:\ needs administrator rights).
        try {
            New-Item -ItemType Directory -Path (Split-Path $Path) -Force | Out-Null
            [IO.File]::Create($Path).Dispose()
            Remove-Item $Path
        } catch {
            throw "Cannot write $Path. Choose a folder of your own, for example $([Environment]::GetFolderPath('Desktop'))\koharu-transfer.zip."
        }

        $projects = @(Get-ChildItem $ProjectsDir -Directory -Filter '*.khrproj' -ErrorAction SilentlyContinue)
        Write-Host "Projects: $($projects.Count) from $ProjectsDir"
        foreach ($project in $projects) {
            Copy-Folder $project.FullName (Join-Path $stage "projects\$($project.Name)")
        }

        if (Test-Path $SettingsDir) {
            Write-Host "Settings: $SettingsDir"
            Copy-Folder $SettingsDir (Join-Path $stage 'settings')
        } else {
            Write-Warning "No settings folder at $SettingsDir."
        }

        # Koharu stores each key as "<provider>.koharu"; other entries that
        # merely mention koharu (a git login for the repository) stay behind.
        $credentials = @([KoharuCredentials]::Read('koharu') |
                Where-Object { $_.TargetName -match '(^|[.:])koharu($|[.:])' })
        Write-Host "API keys: $($credentials.Count)"
        foreach ($credential in $credentials) { Write-Host "  $($credential.TargetName)" }
        if ($credentials.Count -gt 0) {
            $password = Read-Password 'Password to protect the API keys'
            if ($password.Length -lt 8) { throw 'Use a password of at least 8 characters.' }
            if ($password -ne (Read-Password 'Repeat the password')) { throw 'The passwords differ.' }
            $json = ConvertTo-Json -InputObject @($credentials) -Depth 3
            Protect-Text $json $password | Set-Content (Join-Path $stage 'keys.json') -Encoding utf8
        }

        if ($WithModels -and (Test-Path $ModelsDir)) {
            Write-Host "Models and runtimes: $ModelsDir"
            Copy-Folder $ModelsDir (Join-Path $stage 'packages')
        }

        Write-Host "Packing $Path ..."
        New-Item -ItemType Directory -Path (Split-Path $Path) -Force | Out-Null
        [IO.Compression.ZipFile]::CreateFromDirectory($stage, $Path,
            [IO.Compression.CompressionLevel]::Fastest, $false)
        $size = [math]::Round((Get-Item $Path).Length / 1MB)
        Write-Host "Done: $Path ($size MB). Copy it and the script to the other computer."
        return
    }

    if (-not (Test-Path $Path)) { throw "$Path does not exist." }
    Write-Host "Unpacking $Path ..."
    [IO.Compression.ZipFile]::ExtractToDirectory($Path, $stage)

    $keys = Join-Path $stage 'keys.json'
    if (Test-Path $keys) {
        $password = Read-Password 'Password of the API keys'
        $credentials = (Unprotect-Text (Get-Content $keys -Raw) $password) | ConvertFrom-Json
        foreach ($credential in @($credentials)) {
            $entry = [KoharuCredentials+Entry]::new()
            foreach ($name in 'Type', 'TargetName', 'Comment', 'TargetAlias', 'UserName', 'Persist', 'Blob') {
                $entry.$name = $credential.$name
            }
            [KoharuCredentials]::Write($entry)
            Write-Host "Key restored: $($entry.TargetName)"
        }
    }

    $settings = Join-Path $stage 'settings'
    if (Test-Path $settings) {
        if (Test-Path $SettingsDir) {
            $backup = "$SettingsDir.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
            Move-Item $SettingsDir $backup
            Write-Host "Previous settings kept in $backup"
        }
        Copy-Folder $settings $SettingsDir
        Write-Host "Settings restored to $SettingsDir"
    }

    $projects = Join-Path $stage 'projects'
    if (Test-Path $projects) {
        New-Item -ItemType Directory -Path $ProjectsDir -Force | Out-Null
        foreach ($project in Get-ChildItem $projects -Directory) {
            $target = Join-Path $ProjectsDir $project.Name
            if (Test-Path $target) {
                Write-Warning "Skipped $($project.Name): a project with this name already exists."
                continue
            }
            Copy-Folder $project.FullName $target
            Write-Host "Project restored: $($project.BaseName)"
        }
    }

    $packages = Join-Path $stage 'packages'
    if (Test-Path $packages) {
        Copy-Folder $packages $ModelsDir
        Write-Host "Models and runtimes restored to $ModelsDir"
    }
    Write-Host 'Done. Install the fonts you use in typesetting on this computer too.'
} finally {
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
}
