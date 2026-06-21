<#
.SYNOPSIS
    Start every component needed to run Gaia locally, wired to the REMOTE
    Azure Cosmos DB and Azure AI Foundry endpoints.

.DESCRIPTION
    This is the single "press play" script for local development. It:

      1. Verifies the prerequisites (Azure CLI login, cargo, npm).
      2. Reads connection settings from infra/.env (COSMOS_ENDPOINT,
         FOUNDRY_ENDPOINT, MODEL_ROUTER_DEPLOYMENT, ...).
      3. Mints a fresh Azure AD data-plane token for Cosmos
         (COSMOS_AAD_TOKEN) so the Rust backend can read/write the remote
         account keylessly. The Rust Cosmos client uses AAD bearer auth, so
         the static COSMOS_KEY in .env is NOT enough on its own.
      4. Mints a Foundry AAD token only if no FOUNDRY_API_KEY is configured.
      5. Launches the Rust backend (rust/) in HTTP-server mode on
         localhost:<BackendPort> with GAIA_MODE=dev.
      6. Launches the SvelteKit web dev server (web/) on localhost:<WebPort>,
         which proxies /v1, /healthz, /readyz to the backend.

    Cosmos and Foundry run REMOTELY in Azure; only the backend and the web
    front end run on your machine.

.PARAMETER BackendPort
    Port for the Rust backend HTTP/WebSocket server. Default 8080 (matches the
    web dev proxy default in web/vite.config.ts).

.PARAMETER WebPort
    Port for the web dev server. Default 5173.

.PARAMETER SkipTokens
    Do not mint Azure tokens; use whatever is already present in infra/.env or
    the current environment. Useful when offline or when tokens are still valid.

.EXAMPLE
    ./infra/run-local.ps1
    Mints tokens, starts the backend on :8080 and the web app on :5173. Sign-in
    is enabled for whichever providers are configured in infra/.env (Google via
    GOOGLE_CLIENT_ID, GitHub via GITHUB_CLIENT_ID + GITHUB_CLIENT_SECRET).

.NOTES
    Azure AD tokens are short-lived (~1 hour). Re-run this script to refresh
    them if Cosmos calls start returning 401/403 after a long session.
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8080,
    [int]$WebPort = 5173,
    [switch]$SkipTokens
)

$ErrorActionPreference = 'Stop'

# --- Paths ------------------------------------------------------------------
# $PSScriptRoot is infra/; the repo root is its parent. Everything below is
# resolved relative to these two so the script works from any cwd.
$infraDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $infraDir
$rustDir  = Join-Path $repoRoot 'rust'
$webDir   = Join-Path $repoRoot 'web'
$envFile  = Join-Path $infraDir '.env'

Write-Host "Gaia local launcher" -ForegroundColor Cyan
Write-Host "  repo root : $repoRoot"
Write-Host "  backend   : http://localhost:$BackendPort (rust)"
Write-Host "  web app   : http://localhost:$WebPort (web)"
Write-Host ""

# --- Helper: parse a minimal KEY=VALUE .env file ----------------------------
# Mirrors the tiny parser the Rust backend uses (rust/src/llm.rs): ignores
# blank lines and comments, strips an optional surrounding pair of quotes.
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

        # Strip a single pair of surrounding quotes, if present.
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
Test-Command 'npm'
if (-not $SkipTokens) { Test-Command 'az' }

if (-not (Test-Path $envFile)) {
    throw "infra/.env not found. Copy infra/.env.sample to infra/.env and fill it in."
}

$conf = Read-DotEnv -Path $envFile

# --- 2. Mint Azure tokens (remote Cosmos + Foundry) -------------------------
# These are exported into THIS PowerShell session's environment; the backend
# process launched below inherits them. They take precedence over infra/.env.
if (-not $SkipTokens) {
    # Confirm we have an Azure CLI session; fail early with a clear message.
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

    # The token resource is the account host with no port or path, e.g.
    # https://acct.documents.azure.com:443/ -> https://acct.documents.azure.com
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

# --- 3. Backend runtime configuration --------------------------------------
# GAIA_MODE=dev turns on the live Cosmos + Foundry code paths; GAIA_HTTP_PORT
# switches main.rs from the interactive console into HTTP-server mode.
$env:GAIA_MODE      = 'dev'
$env:GAIA_HTTP_PORT = "$BackendPort"

# Point the web dev proxy at the chosen backend port (default already matches).
$env:VITE_API_PROXY = "http://localhost:$BackendPort"

# --- Auth: Google and/or GitHub sign-in -------------------------------------
# Sign-in is mandatory (the old dev name-picker was removed). infra/.env is the
# single source of truth: the backend reads GOOGLE_CLIENT_ID, GITHUB_CLIENT_ID
# and GITHUB_CLIENT_SECRET directly; here we also bridge the *public* client ids
# to the web app's VITE_* vars so the front end shows the matching buttons. The
# GitHub client secret is backend-only and never reaches the browser.
$googleClientId     = $conf['GOOGLE_CLIENT_ID']
$githubClientId     = $conf['GITHUB_CLIENT_ID']
$githubClientSecret = $conf['GITHUB_CLIENT_SECRET']

# Google: backend ID-token verification + web button.
$env:GOOGLE_CLIENT_ID      = $googleClientId
$env:VITE_GOOGLE_CLIENT_ID = $googleClientId

# GitHub: backend code exchange (needs id + secret) + web button (id only).
$env:GITHUB_CLIENT_ID      = $githubClientId
$env:GITHUB_CLIENT_SECRET  = $githubClientSecret
$env:VITE_GITHUB_CLIENT_ID = $githubClientId

$providers = @()
if (-not [string]::IsNullOrWhiteSpace($googleClientId)) { $providers += 'Google' }
if ((-not [string]::IsNullOrWhiteSpace($githubClientId)) -and
    (-not [string]::IsNullOrWhiteSpace($githubClientSecret))) { $providers += 'GitHub' }

if ($providers.Count -gt 0) {
    Write-Host ("Auth: LIVE sign-in via {0}." -f ($providers -join ' + ')) -ForegroundColor Green
} else {
    # Sign-in is mandatory, so with no provider configured the app is unusable.
    Write-Host "Auth: NO provider configured - sign-in is mandatory, so the app will be unusable." -ForegroundColor Yellow
    Write-Host "      Set GOOGLE_CLIENT_ID and/or GITHUB_CLIENT_ID (+ GITHUB_CLIENT_SECRET) in infra/.env." -ForegroundColor Yellow
    Write-Host "      GitHub OAuth App callback URL must be http://localhost:$WebPort/ (note trailing slash)." -ForegroundColor Yellow
}

# --- 4. Launch the two long-running processes -------------------------------
# Each runs in its own window so its logs are easy to read and it can be closed
# independently. Both inherit the environment (incl. the tokens) set above.
Write-Host ""
Write-Host "Starting Rust backend (cargo run) ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath 'powershell' -PassThru -WorkingDirectory $rustDir -ArgumentList @(
    '-NoExit', '-Command',
    "Write-Host 'Gaia backend (port $BackendPort)' -ForegroundColor Cyan; cargo run"
)

Write-Host "Starting web dev server (npm run dev) ..." -ForegroundColor Cyan
$web = Start-Process -FilePath 'powershell' -PassThru -WorkingDirectory $webDir -ArgumentList @(
    '-NoExit', '-Command',
    "Write-Host 'Gaia web (port $WebPort)' -ForegroundColor Cyan; npm run dev -- --port $WebPort"
)

# --- 5. Summary -------------------------------------------------------------
Write-Host ""
Write-Host "Gaia is starting up:" -ForegroundColor Green
Write-Host "  Backend  : http://localhost:$BackendPort/healthz   (PID $($backend.Id))"
Write-Host "  Web app  : http://localhost:$WebPort                (PID $($web.Id))"
Write-Host ""
Write-Host "Cosmos + Foundry are remote (Azure). AAD tokens expire ~1h; re-run this"
Write-Host "script to refresh them. Close the two windows (or stop the PIDs) to shut down."
