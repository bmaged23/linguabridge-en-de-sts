# Historical dependency snapshots

These files preserve previous ASR/TTS evaluation environments and the obsolete main-app uv resolution. They are reference artifacts for older reports, not the current installation entrypoints.

- `uv-main-2026-10-17.lock`: the original root `uv.lock`, retained unchanged for provenance. It predates the current multi-environment release installation.
- The `.txt` files: pinned dependencies for historical recognizer and voice experiments.

Install the current application from `requirements/release/app.txt`, the recognizer from `requirements/stt-cuda.txt` and the GPU voices from `requirements/release/tts.txt`, following the [root README](../../README.md). Keep the environments separate.
