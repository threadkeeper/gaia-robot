<#
.SYNOPSIS
    Run the Gaia data-EXECUTION self-test (LLM Call 2 / push pass) against the
    remote Azure AI Foundry model.

.DESCRIPTION
    This is the on-demand wrapper around the `gaia-robot test-data-execution`
    subcommand and the push-pass counterpart to infra/TestDataRetrieval.ps1.

    For each turn captured under tests/LLM1/t{N}/ it reads that turn's
    responsedatacontext.md, asks LLM Call 2 to answer the user and plan the
    side-effecting actions.json, then audits that every required record was
    emitted this turn:

        response.json (may be multi-modal),
        WhatsApp send, Push send, Edwino actuate,
        and an upsert into each of GaiaConnections, GaiaKB, GaiaDataLake, GaiaDiary.

    Artifacts are written into tests/LLM2/t{N}/ and a TestSummary.md is written
    at tests/LLM2/. The full contract is in tests/LLM2/DataExecutionSpec.md.

    The test is READ-ONLY against production: it validates the actions.json that
    Call 2 plans; it never executes the sends/writes. It therefore only needs a
    model (no Cosmos token).

    Like infra/TestDataRetrieval.ps1, this script:
      1. Verifies prerequisites (Azure CLI login, cargo).
      2. Reads connection settings from infra/.env.
      3. Mints a Foundry AAD token when no FOUNDRY_API_KEY is configured.
      4. Turns on GAIA_MODE=dev and runs the Rust self-test, forwarding its exit
         code so this script also fails when the self-test fails.

.PARAMETER Turn
    Run only this turn number (1-5). Omit to run all five.

.PARAMETER SkipTokens
    Do not mint Azure tokens; use whatever is already present in infra/.env or
    the current environment.

.PARAMETER Release
    Build and run the optimized release binary instead of the debug build.

.EXAMPLE
    ./infra/DataExecution.ps1
    Mints a Foundry token (if needed) and runs all five push-pass turns.

.EXAMPLE
    ./infra/DataExecution.ps1 -Turn 3
    Runs just turn 3.

.NOTES
    Azure AD tokens are short-lived (~1 hour). Re-run this script to refresh.
#>
[CmdletBinding()]
param(
    ## Run only this turn number (1-5). Omit to run all five.
    [ValidateRange(1, 5)]
    [int]$Turn,
    [switch]$SkipTokens,
    [switch]$Release
)

$ErrorActionPreference = 'Stop'

# --- Paths ------------------------------------------------------------------
# $PSScriptRoot is infra/; the repo root is its parent.
$infraDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $infraDir
$rustDir  = Join-Path $repoRoot 'rust'
$envFile  = Join-Path $infraDir '.env'

Write-Host "Gaia data-execution self-test (LLM Call 2 / push pass)" -ForegroundColor Cyan
Write-Host "  repo root : $repoRoot"
Write-Host ""

# --- Helper: parse a minimal KEY=VALUE .env file ----------------------------
# Mirrors the tiny parser the Rust backend uses (rust/src/llm.rs).
function Read-DotEnv {
    param([string]$Path)

    $map = @{}
    if (-not (Test-Path $Path)) { return $map }

    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }

        $idx = $trimmed.IndexOf('=')
        if ($idx -lt 1) { continue }

        $key = $trimmed.Substring(0, $idx).Trim()
        $val = $trimmed.Substring($idx + 1).Trim()

        if ($val.Length -ge 2 -and
            (($val.StartsWith('"') -and $val.EndsWith('"')) -or
             ($val.StartsWith("'") -and $val.EndsWith("'")))) {
            $val = $val.Substring(1, $val.Length - 2)
        }

        $map[$key] = $val
    }
    return $map
}

# --- 1. Prerequisites -------------------------------------------------------
function Test-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Test-Command 'cargo'
if (-not $SkipTokens) { Test-Command 'az' }

if (-not (Test-Path $envFile)) {
    throw "infra/.env not found. Copy infra/.env.sample to infra/.env and fill it in."
}

$conf = Read-DotEnv -Path $envFile

# --- 2. Mint a Foundry token (only when no API key is configured) -----------
# Exported into THIS session so the Rust process inherits it; it takes
# precedence over infra/.env. Cosmos is NOT needed (no reads/writes this test).
if (-not $SkipTokens) {
    try {
        az account show --output none 2>$null
        if ($LASTEXITCODE -ne 0) { throw }
    } catch {
        throw "Not signed in to Azure CLI. Run 'az login' first (or pass -SkipTokens)."
    }

    $foundryKey = $conf['FOUNDRY_API_KEY']
    if ([string]::IsNullOrWhiteSpace($foundryKey)) {
        Write-Host "Minting Foundry AAD token (no FOUNDRY_API_KEY set) ..." -ForegroundColor DarkGray
        $foundryToken = az account get-access-token --resource 'https://cognitiveservices.azure.com' --query accessToken --output tsv
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($foundryToken)) {
            throw "Failed to obtain a Foundry AAD token via 'az account get-access-token'."
        }
        $env:FOUNDRY_AAD_TOKEN = $foundryToken
        Write-Host "  Foundry token acquired." -ForegroundColor Green
    } else {
        Write-Host "Using FOUNDRY_API_KEY from infra/.env for Foundry auth." -ForegroundColor DarkGray
    }
}

# --- 3. Runtime configuration ----------------------------------------------
# GAIA_MODE=dev turns on the live LLM code path the probe needs.
$env:GAIA_MODE = 'dev'

# --- 4. Run the self-test ---------------------------------------------------
Write-Host ""
Write-Host "Running data-execution self-test (user_id = threadkeeper) ..." -ForegroundColor Cyan
Write-Host ""

Push-Location $rustDir
try {
    $subArgs = @('test-data-execution')
    if ($Turn) { $subArgs += $Turn.ToString() }

    if ($Release) {
        cargo run --release --quiet -- @subArgs
    } else {
        cargo run --quiet -- @subArgs
    }
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Data-execution self-test PASSED." -ForegroundColor Green
} else {
    Write-Host "Data-execution self-test FAILED (exit $exitCode)." -ForegroundColor Red
}

# Forward the self-test's exit code so callers (and CI) can gate on it.
exit $exitCode
