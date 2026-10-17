# GPU pipeline performance

Measured 2026-10-16 on GTX 1660 Ti 6 GiB, i7-9750H (12 logical CPUs), WSL with 7.66 GiB RAM. Qwen3-ASR-0.6B FP32, Kokoro English and Piper German were resident together. Gemma4:31b runs remotely on Ollama Cloud.

| Live synthetic request | English | German |
| --- | --- | --- |
| Input words / audio | 23 / 8.38 s | 19 / 7.42 s |
| Final STT after VAD endpoint | 0.576 s | 0.992 s |
| LLM first token | 0.515 s | 0.445 s |
| LLM last token | 0.534 s | 0.591 s |
| First audio after final transcript | 0.965 s | 0.818 s |
| First audio after input WAV ended | 2.063 s | 2.360 s |
| Generated reply audio | 9.09 s | 9.88 s |
| Reply chunks | 3 | 2 |

Earlier CPU TTS live runs returned first audio in 3.64–3.79 seconds. These are small synthetic samples, with variable cloud replies; they are not a controlled speedup estimate. Input WAVs contain trailing silence. Playback packets were decoded and immediately acknowledged, so timings exclude speaker-device latency and real playback backpressure.

Peak sampled whole-device VRAM was 5,072 MiB (4.95 GiB), versus 4,812 MiB loaded idle. App RAM PSS peaked at 2,673 MiB (2.61 GiB); summed RSS peaked at 2,835 MiB. During LLM/TTS phases, app CPU averaged 4.8–4.9% of total CPU capacity with an 8.6% sampled peak; GPU utilization averaged 22–31.5%. Samples were roughly every 250–400 ms and may miss brief peaks. Remote LLM resources are not observable.

English, German and English → German → English synthetic turns passed progressive transcription, streamed cloud tokens, and decoded TTS output. All 23 regression tests passed, including bounded playback and parallel chunk ordering. Existing user history and settings were preserved; the checks removed their own transcript rows. Physical microphone and speaker listening remain a manual check. Mixed-input tests do not establish mixed-language voice switching within an output sentence.

One shared TTS worker avoids duplicating weights. It starts the first phrase immediately, then batches ready German phrases up to two at a time. English remains serial because its standalone GPU benchmark showed no consistent benefit from two concurrent chunks. The earlier standalone benchmark generated fixed 30–31-word outputs in 0.77–0.88 s for English serial and 0.45–0.48 s for German parallel; Qwen was not resident during that benchmark, so those figures are not simultaneous-pipeline guarantees.

[Raw live measurements](data/runtime-profile-gpu.json). Reproduce while the app is running:

```bash
.venv/bin/python -m scripts.validation.validate_deployed_qwen
.venv/bin/python -m scripts.evaluation.profile_voice_pipeline
```

Cloud calls consume account quota. Runtime and queue settings are documented in [GPU TTS](../guides/gpu-speech-synthesis.md) and [configuration](../reference/configuration-parameters.md).
