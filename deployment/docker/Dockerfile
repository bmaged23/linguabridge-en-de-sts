# syntax=docker/dockerfile:1
FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="LinguaBridge EN-DE STS" \
      org.opencontainers.image.source="https://github.com/bmaged23/linguabridge-en-de-sts" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libsndfile1 libgomp1 nodejs \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/release/app.txt /tmp/app.txt
COPY requirements/release/tts.txt /tmp/tts.txt
COPY requirements/stt-cuda.txt /tmp/stt.txt
# Separate ONNX interpreters; ASR owns the one shared CUDA library collection.
RUN python -m venv .venv && python -m venv .venv-asr && python -m venv .venv-tts-gpu \
    && .venv/bin/python -m pip install --no-cache-dir --no-deps -r /tmp/app.txt \
    && .venv-asr/bin/python -m pip install --no-cache-dir --no-deps -r /tmp/stt.txt \
    && .venv-tts-gpu/bin/python -m pip install --no-cache-dir --no-deps -r /tmp/tts.txt
COPY config.py pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY web ./web
COPY tests ./tests
COPY evaluation ./evaluation
COPY docs/reference/model-manifest.json docs/reference/model-notices.md ./docs/reference/
COPY deployment/docker ./deployment/docker
RUN HF_HUB_OFFLINE=0 .venv/bin/python -m scripts.setup.download_speech_models \
    && rm -rf /app/.cache /app/models/.cache /root/.cache \
    && .venv/bin/python -m pytest -q
RUN useradd --uid 10001 --create-home bridge \
    && mkdir -p /app/data /app/logs /app/.cache \
    && chown -R bridge:bridge /app/data /app/logs /app/.cache
USER bridge
EXPOSE 8766
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/health',timeout=4)"]
ENTRYPOINT ["/app/.venv/bin/python", "-m", "deployment.docker.start"]
