# Scope 3 Auditor installer for Windows PowerShell 5.1+.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$RepoUrl = ''
$Branch = 'main'
$Target = if ($env:SCOPE3_SOURCE) { $env:SCOPE3_SOURCE } elseif ($env:SCOPE3_DIR) { $env:SCOPE3_DIR } else { $PSScriptRoot }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'PUBLIC-DEMO-MODE')) { $env:SCOPE3_SOURCE = $PSScriptRoot; $Target = $PSScriptRoot }
$UvVersion = '0.12.13'
$Tier = if ($env:SCOPE3_TIER) { $env:SCOPE3_TIER } else { 'check' }

function Die { param($Message) Write-Error $Message; exit 1 }
function Assert-Source { param($Path) if (-not (Test-Path -LiteralPath (Join-Path $Path 'start.py')) -or -not (Test-Path -LiteralPath (Join-Path $Path 'requirements.txt'))) { Die "'$Path' is not a Scope 3 Auditor checkout." } }
function Assert-Origin {
    param($Path)
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "'$Path' already exists. Set SCOPE3_SOURCE only after reviewing that local source." }
    $origin = (& git -C $Path remote get-url origin 2>$null)
    if ($LASTEXITCODE -ne 0 -or ($origin -ne $RepoUrl -and $origin -ne "$RepoUrl.git")) {
        Die "'$Path' already exists but its Git origin is not $RepoUrl. Set SCOPE3_SOURCE only after reviewing that local source."
    }
}
if ($Tier -notin @('browse', 'check', 'full')) { Die 'SCOPE3_TIER must be browse, check or full.' }

if (-not $env:SCOPE3_SOURCE) {
    if (Test-Path -LiteralPath $Target) {
        Assert-Source $Target
        Assert-Origin $Target
    } elseif (Get-Command git -ErrorAction SilentlyContinue) {
        & git clone --depth 1 --branch $Branch $RepoUrl $Target
        if ($LASTEXITCODE -ne 0) { Die 'git clone failed.' }
    } else {
        Die 'Git is required for a fresh download. Install Git, or set SCOPE3_SOURCE to an existing checkout.'
    }
}
Assert-Source $Target
$Target = (Resolve-Path -LiteralPath $Target).Path
$Tools = Join-Path $Target '.tools'
New-Item -ItemType Directory -Force -Path $Tools | Out-Null

$Uv = if ($env:SCOPE3_UV) { $env:SCOPE3_UV } elseif (Get-Command uv -ErrorAction SilentlyContinue) { (Get-Command uv).Source } else { $null }
if (-not $Uv) {
    Die "uv $UvVersion is required. Install uv from https://docs.astral.sh/uv/getting-started/installation/, then run this installer again."
}
if (-not (Test-Path -LiteralPath $Uv) -and -not (Get-Command $Uv -ErrorAction SilentlyContinue)) { Die 'uv installation failed.' }
$UvFound = & $Uv --version
$UvFound = "$UvFound".Trim()
if ($LASTEXITCODE -ne 0 -or $UvFound -notmatch '^uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)' -or $Matches[1] -ne $UvVersion) { Die "uv $UvVersion is required; found '$UvFound'." }
$UvSource = (Get-Command $Uv).Source
$UvLocal = Join-Path $Tools 'uv.exe'
if ($UvSource -ne $UvLocal) { Copy-Item -LiteralPath $UvSource -Destination $UvLocal -Force }
$Uv = $UvLocal

$PythonDir = Join-Path $Tools 'python'
& $Uv python install 3.12 --install-dir $PythonDir --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { Die 'Python 3.12 installation failed.' }
$env:UV_PYTHON_INSTALL_DIR = $PythonDir
$Python = & $Uv python find 3.12 --managed-python
if ($LASTEXITCODE -ne 0) { Die 'Python 3.12 could not be found after installation.' }
$Python = "$Python".Trim()
if (-not $Python) { Die 'Python 3.12 could not be found after installation.' }

$env:SCOPE3_UV = $Uv
Set-Location -LiteralPath $Target
if ($Tier -ne 'browse') {
    & $Python start.py --install
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($env:SCOPE3_NO_START -eq '1') { exit 0 }
if ($Tier -eq 'browse') { & $Python start.py --browse; exit $LASTEXITCODE }
& $Python start.py
exit $LASTEXITCODE
