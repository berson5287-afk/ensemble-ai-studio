<#
.SYNOPSIS
  Deploy Ensemble AI Studio (hosted edition) to Google Cloud Run.

.DESCRIPTION
  One command from a fresh project to a public HTTPS URL:

    1. enables the APIs Cloud Run, Cloud Build, Artifact Registry and
       Vertex AI need;
    2. creates a least-privilege service account that can call Vertex AI
       and nothing else;
    3. builds the container from ./Dockerfile with Cloud Build and deploys
       it, scaled to zero when idle.

  Re-running is safe: every step is idempotent.  Run from the repo root.

.PARAMETER Project      Google Cloud project id.  Defaults to gcloud's current project.
.PARAMETER Region       Cloud Run region.  us-central1 is the cheapest tier.
.PARAMETER Service      Cloud Run service name.
.PARAMETER Models       Comma-separated Gemini model ids to offer in the UI.
.PARAMETER Passphrase   If set, the demo asks for this before it will answer.
.PARAMETER Private      Require Google sign-in (IAM) instead of a public URL.

.EXAMPLE
  .\deploy\gcp\deploy.ps1 -Project gen-lang-client-0378491575
#>
[CmdletBinding()]
param(
  [string]$Project = "",
  [string]$Region = "us-central1",
  [string]$Service = "ensemble-ai-studio",
  [string]$Models = "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite",
  [string]$Passphrase = "",
  [switch]$Private
)

$ErrorActionPreference = "Stop"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
  throw "gcloud is not on PATH. Install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install"
}
if (-not (Test-Path "Dockerfile")) {
  throw "Run this from the repository root (where the Dockerfile is)."
}
if (-not $Project) {
  $Project = (gcloud config get-value project 2>$null).Trim()
  if (-not $Project) { throw "No project. Pass -Project or run: gcloud config set project <id>" }
}

$account = "ensemble-web"
$serviceAccount = "$account@$Project.iam.gserviceaccount.com"

Step "Project $Project, region $Region, service $Service"
gcloud config set project $Project | Out-Null

Step "Enabling APIs (Cloud Run, Cloud Build, Artifact Registry, Vertex AI)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
  artifactregistry.googleapis.com aiplatform.googleapis.com

Step "Service account $serviceAccount (Vertex AI user only)"
$existing = gcloud iam service-accounts list --filter="email:$serviceAccount" --format="value(email)"
if (-not $existing) {
  gcloud iam service-accounts create $account `
    --display-name "Ensemble AI Studio web (Cloud Run runtime)"
}
gcloud projects add-iam-policy-binding $Project `
  --member "serviceAccount:$serviceAccount" --role "roles/aiplatform.user" `
  --condition=None --quiet | Out-Null

$envVars = @(
  "GOOGLE_CLOUD_PROJECT=$Project",
  "GOOGLE_CLOUD_LOCATION=global",
  "GEMINI_MODELS=$Models"
)
if ($Passphrase) { $envVars += "DEMO_PASSPHRASE=$Passphrase" }
$auth = if ($Private) { "--no-allow-unauthenticated" } else { "--allow-unauthenticated" }

Step "Building and deploying (Cloud Build → Artifact Registry → Cloud Run)"
gcloud run deploy $Service `
  --source . `
  --region $Region `
  --service-account $serviceAccount `
  $auth `
  --port 8080 `
  --cpu 1 --memory 512Mi `
  --min-instances 0 --max-instances 3 --concurrency 40 `
  --timeout 900 `
  --set-env-vars ("^|^" + ($envVars -join "|")) `
  --labels app=ensemble-ai-studio `
  --quiet

$url = gcloud run services describe $Service --region $Region --format "value(status.url)"
Step "Live: $url"
Write-Host "Health: $url/healthz"
if ($Private) { Write-Host "Private: grant viewers roles/run.invoker, open with an authenticated proxy." }
