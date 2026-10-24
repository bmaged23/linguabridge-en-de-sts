# Exact Python environment snapshots

The app uses separate interpreters because CPU/GPU ONNX Runtime and the Qwen PyTorch stack need independent environments. Run installation commands from the repository root; the root README contains the full setup and dependency-check caveats.

| Snapshot | Environment | Purpose |
| --- | --- | --- |
| [`../requirements.txt`](../requirements.txt) | `.venv` | Main application, CPU VAD, browser QA and tests |
| [`stt-cuda.txt`](stt-cuda.txt) | `.venv-asr` | Deployed Qwen3-ASR GPU recognizer |
| [`tts-cuda.txt`](tts-cuda.txt) | `.venv-tts-gpu` | Deployed GPU Kokoro English and Piper German |
| [`experimental-qwen-tts.txt`](experimental-qwen-tts.txt) | `.venv-qwen-tts` | Optional Qwen TTS experiment; not part of normal startup |
| [`evaluation.txt`](evaluation.txt) | `.venv-evaluation` | Pinned lightweight cloud scoring/judging tools; local speech evaluation uses the main/GPU environments |
| [`archive/`](archive) | Historical evaluation environments | Previous snapshots, preserved for older experiment reproduction |

Every installed distribution is pinned with `==`. Keep snapshots separate and install with `--no-deps` as described in the [root README](../README.md). The app/TTS files include `-e .`, which refers to the repository root when you run the commands there. CUDA PyTorch snapshots include their wheel index.

```bash
.venv/bin/python -m scripts.dependencies.export_installed_versions
.venv/bin/python -m scripts.dependencies.export_installed_versions --check
```

These commands read installed package metadata without importing models or changing environments. The main `uv.lock` remains an older dependency resolution; it does not reproduce all three production environments.

## Captured environment caveats

The main app and ASR environments pass `uv pip check`. GPU TTS currently reports six incompatibilities: four packages declare a dependency on the CPU `onnxruntime` distribution even though the GPU distribution provides the same Python module; `onnxruntime-gpu` also declares `coloredlogs` and `sympy`, which are absent in this captured environment. These files preserve the installed versions that ran the local app. They do not claim a clean dependency check or a validated clean-machine installation.

Do not fix the CPU/GPU package-name mismatch by co-installing both ONNX Runtime distributions. The project keeps CPU VAD and GPU voices in separate interpreters. The optional Qwen TTS environment is an experiment and is not required for normal startup.

```bash
uv pip check --python .venv/bin/python
uv pip check --python .venv-asr/bin/python
uv pip check --python .venv-tts-gpu/bin/python
```

## Release environments

`release/app.txt` omits unused main-interpreter CUDA packages and has been freshly installed: all 76 distributions pass dependency checks and 89 tests pass. `release/tts.txt` shares ASR CUDA libraries and adds declared missing helper dependencies; see the [container guide](../deployment/README.md). Original installed-environment snapshots above are retained unchanged.

The README now uses those release files for new installations. The clean main environment, including the editable project and its corrected core dependency metadata, passes dependency checks and all 91 current tests. The historical 89-test result above describes the earlier checkpoint.

For `stt-cuda.txt`, use uv's `--index-strategy unsafe-best-match`: this snapshot pins packages across PyPI and the official PyTorch CUDA index, matching pip's cross-index selection. uv's default first-index policy otherwise finds packages such as `certifi` on the CUDA index but cannot find their pinned version there. Versions remain exact. Do not add that index policy to unrelated installations.
