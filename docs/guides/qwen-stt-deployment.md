# Local Qwen recognizer

The app now uses Qwen3-ASR-0.6B on CUDA with FP32 and eager attention. A persistent
worker in `.venv-asr` keeps PyTorch separate from the app's existing `.venv`, Kokoro,
and Piper dependencies. All inference is local. The Ollama conversation model is
unchanged. The worker loads only the pinned local checkpoint, with downloads disabled
at runtime. It receives audio and STT settings, not cloud credentials.

On a fresh installation, after creating the normal app `.venv`:

```bash
uv venv .venv-asr --python .venv/bin/python
uv pip install --python .venv-asr/bin/python --no-deps -r requirements/stt-cuda.txt
.venv/bin/python -m scripts.setup.download_speech_models
.venv/bin/python -m scripts.application.start_voice_app
```

Open http://localhost:8766 and allow microphone access. Use headphones when testing
interruption. The UI shows replaceable partial transcripts every two seconds while
you speak; final transcription waits for the existing VAD endpoint. Ollama and TTS
continue streaming after the final transcript. This is progressive cumulative-audio
decoding with text-prefix reuse, not native vLLM streaming.

All model, interpreter, precision, attention, token, rollback, context, update interval,
audio-window, timeout, and worker settings are in `config.py`. Qwen runs as one GPU
process; the main process uses one serial transport thread. Kokoro/Piper retain their
separate GPU process. Health reports the active STT model and precision.

Each utterance has a session-specific stream ID. End-of-turn and new-stream requests
reset prefix state. Previews are cumulative (up to the configured maximum utterance),
and only final text is submitted as a conversation turn. Shutdown closes the worker.
Qwen emits a language label but no calibrated confidence; the existing `certain`
protocol field means a supported English/German label, not a confidence score.
Mixed speech remains mixed in the text; existing reply-language and TTS controls are
unchanged by this recognizer deployment.

See [local evaluation results](../reports/asr-evaluation.md). Synthetic tests found much
better mixed-language preservation, but occasional unwanted translation still occurs.
Human microphone accuracy has not been benchmarked. Earlier partial errors can be
retained by prefix reuse.

## Deployment validation

The running app reported Qwen3-ASR-0.6B, CUDA, FP32 after restart. All 22 regression tests passed. Three real-time synthetic WebSocket turns (English, German, and English → German → English) passed VAD, partial/final transcription, streamed Ollama tokens, and chunked TTS. Playback was decoded and acknowledged by the test, not physically heard through a speaker. Only the test’s own transcript rows were removed; the existing 18 user turns and settings were preserved. Synthetic validation details are in the ignored `logs/qwen-deployment-check.json`.
