# GPU speech generation

Kokoro English and Piper German now use CUDA ONNX Runtime in one persistent worker
process, sharing their loaded weights. Qwen remains in its own GPU process. The
browser still receives and plays audio in order as the LLM streams its reply.

The first phrase starts immediately. Subsequent ready phrases can be generated
concurrently, bounded by `tts.parallel_chunks`, per-language settings, and the
playback queue. Testing selected English concurrency 1 and German concurrency 2:
English parallel inference was slower than serial GPU inference on this card.
Parallelism does not wait for the whole LLM answer or duplicate model weights into
multiple GPU processes. Phonemization is locked separately for thread safety.

All settings are in `config.py`: `tts.device`, `runtime_python`,
`cuda_library_runtime`, `parallel_chunks`, `parallel_chunks_by_language`, CUDA
device/arena/workspace options, and `workers` queue/process counts. The CUDA arena
limit is 768 MiB per model, not a limit on total process VRAM. CUDA context,
libraries, and other allocations are additional. Runtime initialization fails
clearly if CUDA is unavailable instead of silently falling back to CPU inference.
ONNX may still assign supported helper/shape operations to CPU.

## Setup

First install the normal app and Qwen environments. Then:

```bash
uv venv .venv-tts-gpu --python .venv/bin/python
uv pip install --python .venv-tts-gpu/bin/python --no-deps -r requirements/tts-cuda.txt
.venv/bin/python -m scripts.application.start_voice_app
```

The pinned GPU runtime replaces `onnxruntime` with `onnxruntime-gpu==1.20.2`.
Do not install both distributions into that environment: they share the same
Python module. `--no-deps` uses the complete pinned list and prevents Piper/Kokoro
metadata from reinstalling the CPU distribution.

`scripts/application/start_voice_app.py` uses the CUDA 12.4 libraries from the Qwen `.venv-asr` runtime
when GPU TTS is selected. This matches the local driver and keeps the main
environment's CPU ONNX Runtime available for VAD. The TTS process is spawned using
the configured GPU interpreter. Use the launcher rather than bare uvicorn.

The CPU timings in [RUNTIME_PROFILE.md](../reports/runtime-profile.md) describe the previous
deployment. Current live results are in [GPU performance](../reports/gpu-performance.md).

Primary reference: [ONNX Runtime CUDA compatibility and options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Mixed-language replies

With `tts.language_segments.enabled=True`, the cloud LLM marks each actual English/German span. An incremental parser strips the markers, emits clean transcript tokens, and buffers speech phrases with their own language. A language switch flushes even a short quotation so it cannot be combined with the next language's voice. Same-language ready phrases can still run in parallel; a language change preserves order across batches. No classifier, second cloud request, or audio sample-rate assumption is added. Each browser chunk keeps its native sample rate.

Six synthetic real-cloud/GPU checks passed, including simple and complex EN → DE → EN and DE → EN → DE. See [playable output checks](../mixed-output-check/index.html). The test injects text instead of microphone STT to isolate output routing and uses a temporary memory database. It decodes and acknowledges audio without physical speaker listening. The deployed microphone pipeline is checked separately.

Punctuation-only phrases are never synthesized or used to switch voices. Punctuation remains in the visible transcript; short real words and numbers still produce speech. The LLM is instructed to keep punctuation with its preceding words.
