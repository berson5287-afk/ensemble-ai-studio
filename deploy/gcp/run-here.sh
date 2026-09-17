#!/usr/bin/env bash
# Run the hosted edition on this machine — Cloud Shell, a laptop, a VM.
#
#   deploy/gcp/run-here.sh            # serves on http://localhost:8080
#   PORT=9000 deploy/gcp/run-here.sh
#
# In Cloud Shell, click "Web Preview" → "Preview on port 8080" to open it.
# Credentials come from the environment: Cloud Shell and Compute VMs already
# have them; on a laptop run `gcloud auth application-default login` first.
set -euo pipefail
cd "$(dirname "$0")/../.."

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[[ -n "$PROJECT" ]] || { echo "Set GOOGLE_CLOUD_PROJECT or 'gcloud config set project <id>'."; exit 1; }

if [[ ! -x .venv/bin/python ]]; then
  echo "==> Creating a virtualenv and installing the web dependencies"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements-web.txt
fi

export GOOGLE_CLOUD_PROJECT="$PROJECT"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
PORT="${PORT:-8080}"
echo "==> Ensemble AI Studio on http://localhost:${PORT}  (project ${PROJECT}, Ctrl+C to stop)"
exec .venv/bin/uvicorn aichatlab.web.server:app --host 0.0.0.0 --port "$PORT"
