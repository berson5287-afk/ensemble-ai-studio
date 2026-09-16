# Ensemble AI Studio — hosted edition for Cloud Run.
#
# One small image: the orchestration package, the web front end, and the
# Gemini adapter.  No Tkinter, no Ollama — models come from Vertex AI through
# the service account Cloud Run runs as, so the image carries no secrets.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so a code change does not reinstall them.
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

COPY aichatlab ./aichatlab

# Run as a non-root user; Cloud Run does not require it, reviewers notice it.
RUN useradd --create-home --uid 10001 app
USER app

# Cloud Run sets PORT; 8080 is its default and works locally too.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')" || exit 1

# One worker: orchestration runs on threads inside the process, and Cloud Run
# scales by adding instances, not workers.
CMD ["sh", "-c", "exec uvicorn aichatlab.web.server:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75 --no-access-log"]
