# Hosted edition: Ensemble AI Studio on Google Cloud Run

The desktop app is a Tkinter window over a UI-free orchestration core, driving
Ollama on a GPU under the desk. The hosted edition keeps the core and swaps
the two things that do not travel: the window becomes a browser page, and the
GPU becomes Gemini on Vertex AI. Nothing in `orchestrator.py` or `session.py`
changed to make that happen.

```
 Browser                      Cloud Run (scales to zero)                 Vertex AI
┌──────────────┐  POST /api/run  ┌───────────────────────────────┐        ┌──────────────┐
│ index.html   │ ──────────────► │ FastAPI  aichatlab/web/server │        │ gemini-3.8-  │
│ vanilla JS   │ ◄── NDJSON ──── │   └─ Orchestrator (unchanged) │ ─────► │   flash      │
│ streams cards│   turn_start    │        └─ GeminiClient        │ stream │ gemini-3.7-  │
│ as they come │   token, note,  │           (aichatlab/gemini)  │        │   flash      │
└──────────────┘   turn_end, …   └───────────────────────────────┘        │ gemini-3.5-  │
                                    runs as ensemble-web@…               │   flash-lite │
                                    (roles/aiplatform.user only)         └──────────────┘
```

## What is where

| Piece | File | Role |
| --- | --- | --- |
| Model adapter | `aichatlab/gemini.py` | Same method signatures as `OllamaClient`; translates messages, options and streamed parts. `num_predict` → `max_output_tokens`, `MAX_TOKENS` → `done_reason: length`, thoughts → `on_thought`. |
| HTTP layer | `aichatlab/web/server.py` | One streaming endpoint per run, the orchestrator on a worker thread, its `emit` events serialised as newline-delimited JSON. |
| Page | `aichatlab/web/static/index.html` | Model picker, seven-mode selector, streaming cards with thinking panels, join-in box for conversation mode. No build step. |
| Container | `Dockerfile` | `python:3.12-slim`, non-root, one uvicorn worker, health check. No Tk, no Ollama, no secrets. |
| Deploy | `deploy/gcp/deploy.ps1`, `deploy.sh` | Enable APIs → least-privilege service account → Cloud Build → Cloud Run. Idempotent. |

## Deploying

Prerequisites: the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install),
a project with billing enabled, and `gcloud auth login`.

```powershell
# Windows
.\deploy\gcp\deploy.ps1 -Project <project-id>
```

```bash
# macOS / Linux / Cloud Shell
deploy/gcp/deploy.sh <project-id>
```

The script prints the service URL at the end. First deploy takes three to five
minutes (Cloud Build has to build the image); later ones are faster.

Options:

| Flag | Effect |
| --- | --- |
| `-Models "a,b,c"` / `GEMINI_MODELS` | Which Gemini models the page offers. Any current Vertex chat model id works. |
| `-Passphrase xyz` / `DEMO_PASSPHRASE` | The page asks for a passphrase before it will answer. Cheap privacy for a demo. |
| `-Private` / `PRIVATE=1` | Require Google sign-in (Cloud Run IAM) instead of a public URL. |
| `-Region` / `REGION` | Cloud Run region. Vertex calls use the `global` endpoint regardless. |

## Running it in Cloud Shell (or anywhere with gcloud)

```bash
deploy/gcp/run-here.sh        # first run creates .venv and installs; then serves on :8080
```

In Cloud Shell, click **Web Preview → Preview on port 8080**. Credentials are
the ones the shell already has; no keys, no setup.

## Running it locally

```bash
pip install -r requirements-web.txt
gcloud auth application-default login          # once
set GOOGLE_CLOUD_PROJECT=<project-id>           # PowerShell: $env:GOOGLE_CLOUD_PROJECT="…"
uvicorn aichatlab.web.server:app --reload --port 8080
```

Or with the Gemini Developer API instead of Vertex: set `GEMINI_API_KEY` and
skip the project variable.

The container can be run the same way with Docker: `docker build -t ensemble .`
then `docker run -p 8080:8080 -e GEMINI_API_KEY=… ensemble`.

## Design decisions

**Why Cloud Run and not a GPU VM.** A VM running Ollama costs money every hour
it exists, whether or not anyone is looking. Cloud Run bills per request and
scales to zero, and Gemini bills per token. A portfolio demo that gets visited
twice a week costs cents. The trade is that the *models* are now Google's
rather than open weights; the orchestration — which is the point of the
project — is identical.

**Why a service account and no API key.** Cloud Run runs as
`ensemble-web@<project>.iam.gserviceaccount.com`, which holds exactly one role,
`roles/aiplatform.user`. The `google-genai` client picks those credentials up
from the metadata server. There is no key in the image, the repo, the env or
the build logs to rotate or leak. Locally the same code uses your own
`gcloud auth application-default login`.

**Why the event stream is the same as the desktop app's.** The orchestrator
emits `turn_start`, `token`, `thought`, `turn_end`, `note`, `plan`, `check`
and friends. The Tk app drains those from a queue on a timer; the web server
drains them from a queue into an HTTP response. The page is therefore a second
renderer of one protocol, and any new pattern added to the orchestrator shows
up in both front ends without further work.

**Why in-memory sessions.** Conversations are held per browser session for an
idle hour, in the instance's memory. Cloud Run can recycle that instance at any
time. A database would make the demo look more durable than a demo needs to
be, and would add a second bill and a second thing to secure. The page says so
in its sidebar.

**What guards the public URL.** A prompt length cap, at most four models per
run, at most three debate rounds and eight conversation turns, a per-caller
limit of six runs a minute (from `X-Forwarded-For`, which Cloud Run sets), a
per-run wall-clock timeout, `max-instances 3` on the service, and the option
of a passphrase or IAM. None of these is a substitute for a budget alert on the
project, which is a one-minute setup in the console and worth doing.

**Why one uvicorn worker.** Orchestration already fans out on threads inside
the process (a broadcast to four models is four concurrent streams). Cloud Run
adds capacity by adding instances, so a second worker per instance would only
compete for the same vCPU.

## Cost, roughly

| Component | Idle | In use |
| --- | --- | --- |
| Cloud Run (1 vCPU, 512 MiB, min 0) | $0 | fractions of a cent per request; a generous free tier |
| Cloud Build | $0 | first 120 build-minutes a day free |
| Artifact Registry | ~$0 for one small image | — |
| Vertex AI, Gemini Flash-class | $0 | on the order of a tenth of a cent per typical debate turn |

A Cloud Billing budget alert at $5 a month catches anything surprising.
