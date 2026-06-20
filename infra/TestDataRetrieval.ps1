<#
.SYNOPSIS
    Run the Gaia data-retrieval self-test against the REMOTE Azure Cosmos DB,
    Azure AI Foundry, and Brave Search APIs.

.DESCRIPTION
    This is the on-demand wrapper around the `gaia-robot test-data-retrieval`
    subcommand. It validates the whole *retrieval* half of a turn before you
    ship: it asks the model five questions of varying length and subject as the
    `threadkeeper` user, parses the actions.json each one produces, executes the
    Cosmos queries and the Brave web search, and reports per question:

        success / failure,  rows retrieved,  data size (KB)

    The same subcommand is used as a CI gate before deploy: it exits non-zero on
    any failure, so a red self-test halts the pipeline.

    Like infra/run-local.ps1, this script:
      1. Verifies prerequisites (Azure CLI login, cargo).
      2. Reads connection settings from infra/.env.
      3. Mints a fresh Cosmos data-plane AAD token (COSMOS_AAD_TOKEN) and, when
         no FOUNDRY_API_KEY is configured, a Foundry AAD token.
      4. Turns on GAIA_MODE=dev and runs the Rust self-test, forwarding its exit
         code so this script also fails when the self-test fails.

    Cosmos, Foundry, and Brave run REMOTELY; only the probe runs locally.

.PARAMETER SkipTokens
    Do not mint Azure tokens; use whatever is already present in infra/.env or
    the current environment. Useful when tokens are still valid.

.PARAMETER Release
    Build and run the optimized release binary instead of the debug build.

.EXAMPLE
    ./infra/TestDataRetrieval.ps1
    Mints tokens and runs the five-question data-retrieval self-test.

.EXAMPLE
    ./infra/TestDataRetrieval.ps1 -SkipTokens
    Runs the self-test using the tokens/keys already in the environment.

.NOTES
    Azure AD tokens are short-lived (~1 hour). Re-run this script to refresh
    them if Cosmos calls start returning 401/403.
#>
[CmdletBinding()]
param(
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

Write-Host "Gaia data-retrieval self-test" -ForegroundColor Cyan
Write-Host "  repo root : $repoRoot"
Write-Host ""

# --- Helper: parse a minimal KEY=VALUE .env file ----------------------------
# Mirrors the tiny parser the Rust backend uses (rust/src/llm.rs): ignores blank
# lines and comments, strips an optional surrounding pair of quotes.
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

# --- 2. Mint Azure tokens (remote Cosmos + Foundry) -------------------------
# Exported into THIS session so the Rust process inherits them; they take
# precedence over infra/.env.
if (-not $SkipTokens) {
    try {
        az account show --output none 2>$null
        if ($LASTEXITCODE -ne 0) { throw }
    } catch {
        throw "Not signed in to Azure CLI. Run 'az login' first (or pass -SkipTokens)."
    }

    # -- Cosmos data-plane token --
    $cosmosEndpoint = $conf['COSMOS_ENDPOINT']
    if ([string]::IsNullOrWhiteSpace($cosmosEndpoint)) {
        throw "COSMOS_ENDPOINT is not set in infra/.env."
    }

    # The token resource is the account host with no port or path.
    $cosmosUri = [System.Uri]$cosmosEndpoint
    $cosmosResource = "$($cosmosUri.Scheme)://$($cosmosUri.Host)"

    Write-Host "Minting Cosmos AAD token for $cosmosResource ..." -ForegroundColor DarkGray
    $cosmosToken = az account get-access-token --resource $cosmosResource --query accessToken --output tsv
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($cosmosToken)) {
        throw "Failed to obtain a Cosmos AAD token via 'az account get-access-token'."
    }
    $env:COSMOS_AAD_TOKEN = $cosmosToken
    Write-Host "  Cosmos token acquired." -ForegroundColor Green

    # -- Foundry token (only if no API key is configured) --
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
# GAIA_MODE=dev turns on the live LLM + Cosmos + Brave code paths the probe needs.
$env:GAIA_MODE = 'dev'

# --- 4. Run the self-test ---------------------------------------------------
Write-Host ""
Write-Host "Running data-retrieval self-test (user_id = threadkeeper) ..." -ForegroundColor Cyan
Write-Host ""

Push-Location $rustDir
try {
    if ($Release) {
        cargo run --release --quiet -- test-data-retrieval
    } else {
        cargo run --quiet -- test-data-retrieval
    }
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "Data-retrieval self-test PASSED." -ForegroundColor Green
} else {
    Write-Host "Data-retrieval self-test FAILED (exit $exitCode)." -ForegroundColor Red
}

# Forward the self-test's exit code so callers (and CI) can gate on it.
exit $exitCode
