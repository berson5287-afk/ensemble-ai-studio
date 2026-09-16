#!/usr/bin/env bash
# Deploy Ensemble AI Studio (hosted edition) to Google Cloud Run.
#
# Same steps as deploy.ps1, for macOS/Linux and Cloud Shell:
#   1. enable the APIs;  2. least-privilege service account;
#   3. build with Cloud Build and deploy, scaled to zero when idle.
#
# Usage:  deploy/gcp/deploy.sh [project-id]
# Env:    REGION, SERVICE, GEMINI_MODELS, DEMO_PASSPHRASE, PRIVATE=1
set -euo pipefail

PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ensemble-ai-studio}"
MODELS="${GEMINI_MODELS:-gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite}"
ACCOUNT="ensemble-web"
SA="${ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"

[[ -f Dockerfile ]] || { echo "Run from the repository root."; exit 1; }
[[ -n "$PROJECT" ]] || { echo "No project: pass one or 'gcloud config set project'."; exit 1; }

step() { printf '\n==> %s\n' "$*"; }

step "Project $PROJECT, region $REGION, service $SERVICE"
gcloud config set project "$PROJECT" >/dev/null

step "Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com

step "Service account $SA (Vertex AI user only)"
if ! gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$ACCOUNT" \
    --display-name "Ensemble AI Studio web (Cloud Run runtime)"
fi
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/aiplatform.user \
  --condition=None --quiet >/dev/null

ENV="GOOGLE_CLOUD_PROJECT=${PROJECT}|GOOGLE_CLOUD_LOCATION=global|GEMINI_MODELS=${MODELS}"
[[ -n "${DEMO_PASSPHRASE:-}" ]] && ENV="${ENV}|DEMO_PASSPHRASE=${DEMO_PASSPHRASE}"
AUTH="--allow-unauthenticated"
[[ "${PRIVATE:-0}" == "1" ]] && AUTH="--no-allow-unauthenticated"

step "Building and deploying"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --service-account "$SA" \
  $AUTH \
  --port 8080 \
  --cpu 1 --memory 512Mi \
  --min-instances 0 --max-instances 3 --concurrency 40 \
  --timeout 900 \
  --set-env-vars "^|^${ENV}" \
  --labels app=ensemble-ai-studio \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
step "Live: $URL"
