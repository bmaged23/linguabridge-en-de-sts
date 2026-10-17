# Exact Python environment snapshots

The app uses separate interpreters because CPU/GPU ONNX Runtime and the Qwen PyTorch stack need independent environments. Run installation commands from the repository root; the root README contains the full setup and dependency-check caveats.

| Snapshot | Environment | Purpose |
| --- | --- | --- |
| [`../requirements.txt`](../requirements.txt) | `.venv` | Main application, CPU VAD, browser QA and tests |
| [`stt-cuda.txt`](stt-cuda.txt) | `.venv-asr` | Deployed Qwen3-ASR GPU recognizer |
| [`tts-cuda.txt`](tts-cuda.txt) | `.venv-tts-gpu` | Deployed GPU Kokoro English and Piper German |
| [`experimental-qwen-tts.txt`](experimental-qwen-tts.txt) | `.venv-qwen-tts` | Optional Qwen TTS experiment; not part of normal startup |
| [`archive/`](archive) | Historical evaluation environments | Previous snapshots, preserved for older experiment reproduction |

Every installed distribution is pinned with `==`. Keep snapshots separate and install with `--no-deps` as described in the [root README](../README.md). The app/TTS files include `-e .`, which refers to the repository root when you run the commands there. CUDA PyTorch snapshots include their wheel index.

```bash
.venv/bin/python -m scripts.dependencies.export_installed_versions
.venv/bin/python -m scripts.dependencies.export_installed_versions --check
```

These commands read installed package metadata without importing models or changing environments. The main `uv.lock` remains an older dependency resolution; it does not reproduce all three production environments.
