# Dependency environments

Run installation commands from the repository root. The application, recognizer and GPU voices use separate environments to keep CPU/GPU ONNX Runtime and PyTorch dependencies compatible. Follow the [complete installation steps](../README.md#installation--clone-to-first-conversation).

## Current installation files

| File | Environment | Purpose |
| --- | --- | --- |
| [`release/app.txt`](release/app.txt) | `.venv` | Application, CPU VAD, browser evaluation and offline tests |
| [`stt-cuda.txt`](stt-cuda.txt) | `.venv-asr` | Qwen3-ASR and shared CUDA libraries |
| [`release/tts.txt`](release/tts.txt) | `.venv-tts-gpu` | GPU Kokoro/Piper and required helper libraries |
| [`evaluation.txt`](evaluation.txt) | `.venv-evaluation` | Optional lightweight cloud scoring/judging tools; speech evaluation uses the main/GPU environments |

Install exact pins with `--no-deps` as shown in the root README. For `stt-cuda.txt`, also use uv's `--index-strategy unsafe-best-match`: its pins span PyPI and the official PyTorch CUDA index. This matches the original pip selection; uv's default first-index policy cannot find some pinned versions on the CUDA index. Do not apply that policy to unrelated installs.

The release app file omits unused main-interpreter CUDA packages. The release TTS file adds `coloredlogs`, `humanfriendly`, `sympy` and `mpmath` and shares the recognizer's CUDA libraries. Some voice packages declare the CPU `onnxruntime` distribution in their metadata, while the GPU distribution provides the required `onnxruntime` module. Do not co-install both distributions in one environment. See [container and dependency details](../deployment/README.md).

## Installed snapshots and historical experiments

| File | Purpose |
| --- | --- |
| [`../requirements.txt`](../requirements.txt) | Original exact main-environment snapshot |
| [`tts-cuda.txt`](tts-cuda.txt) | Original installed GPU voice snapshot; the release variant supplies additional declared helper dependencies |
| [`experimental-qwen-tts.txt`](experimental-qwen-tts.txt) | Optional Qwen TTS experiment; not used by application startup |
| [`evaluation.in`](evaluation.in) | Input declarations for the lightweight evaluation lock |
| [`archive/`](archive/README.md) | Earlier evaluation snapshots and the obsolete main-app uv lock |

Historical files preserve the environments used for earlier measurements. Their package metadata/dependency checks should not be interpreted as current clean-install instructions. The old root `uv.lock` is archived as [`archive/uv-main-2026-10-17.lock`](archive/uv-main-2026-10-17.lock); it does not reproduce the three production environments. New root lockfiles are ignored to avoid confusing them with the supported pinned-requirements route.

## Export installed versions

```bash
.venv/bin/python -m scripts.dependencies.export_installed_versions
.venv/bin/python -m scripts.dependencies.export_installed_versions --check
```

These commands read installed package metadata without importing models or changing environments. They export/check the original environment snapshots, not the curated release variants. Editable `-e .` entries refer to the repository root. An absent optional environment is skipped.

## Verification

The [verified GPU release](../evaluation/results/gpu-container-release-2026-10-24/README.md) records the exact source, public image digest, 94 offline tests and real-model/browser checks. The clean release application environment passed dependency checks. Earlier clean-install checkpoints with 89 or 91 tests describe the suite at those dates, not the current suite size. Fresh native installation on every possible device is not certified; the Docker report names the tested hardware and supported platform.
