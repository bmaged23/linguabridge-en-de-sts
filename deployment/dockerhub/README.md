# LinguaBridge STS — English–German Speech-to-Speech Translation

[GitHub source, demo and full documentation](https://github.com/bmaged23/linguabridge-en-de-sts)

Talk in English, German or both. The application supports conversation, confirmed translation-mode changes, teaching, persistent notes and voice interruption. Qwen3-ASR-0.6B recognizes speech locally; Silero VAD handles turn timing; Ollama Cloud Gemma 4 31B generates streamed replies; Kokoro English and Piper Thorsten German produce ordered audio chunks locally.

## Pull and run

Requires Linux x86-64 containers and an NVIDIA GPU exposed to Docker. The tested native GPU has 6 GiB VRAM. [Verified GPU/container release, digest and measurements](https://github.com/bmaged23/linguabridge-en-de-sts/tree/main/evaluation/results/gpu-container-release-2026-10-24). This is not a CPU image.

Create a local `.env` containing your own credentials:

```dotenv
OLLAMA_API_KEY=your_own_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gemma4:31b
```

```bash
docker pull bmaged23/linguabridge-en-de-sts:gpu
docker run --rm --gpus all --env-file .env \
  -p 127.0.0.1:8766:8766 \
  -v linguabridge-memory:/app/data \
  bmaged23/linguabridge-en-de-sts:gpu
```

After warmup, open http://localhost:8766 and allow microphone access. No Git clone or first-run model download is needed: the image contains the production speech models. The cloud LLM requires your own account allowance. Your microphone stays in the browser, so no audio device passthrough is required. Notes/history persist in the named Docker volume. Stop with Ctrl+C.

## Application API

The FastAPI backend exposes HTTP health/state/configuration/control endpoints and `WS /ws` for streaming microphone audio, STT text, LLM replies and synthesized speech. Start the container first, then connect at `http://localhost:8766`. See the [API README](https://github.com/bmaged23/linguabridge-en-de-sts/blob/main/docs/api/README.md) for endpoint examples, audio formats, playback acknowledgements and the current same-origin, single-session scope.

## Verified container measurements

The qualified image passed **94 offline tests, 52/52 STT recordings and 17/17 browser turns**, with zero JavaScript errors. Median first audio after recording ended was **2.53 seconds**. Highest observed container memory was **2,601 MiB** and whole-device VRAM **4,954 MiB** on a GTX 1660 Ti with 6 GiB VRAM; the cloud LLM hardware is excluded. These are sampled observations on one setup, not device-wide guarantees. Whisper independently scored 37/37 output chunks; English agreement WER/CER 4.04%/2.84%, German 9.09%/7.08%, including number-format differences and one lexical disagreement. See the linked report for all comparisons and limitations.

Immutable image: `bmaged23/linguabridge-en-de-sts@sha256:04ec957cf2fb6df7e0d9b0ca378ac7f3f37e5fcc79ebcd04181ce36b2c08bcbd`.

## Evaluation and release process

The candidate image runs offline tests, warms real GPU models, compares 52 archived ASR waveforms against frozen references with per-recording WER/CER and meaning/routing gates, and completes 17 prerecorded browser conversation turns. Publication follows passing gates. The workflow pulls the published tag, verifies its identity and tests normal startup again. GitHub-hosted tests and the trusted GPU evaluation runner have separate roles.

Native measurements: caching completed ASR segments with GPU batches of two preserved all 52 final transcripts. A matched mixed story's first-audio delay fell from 7.77 to 2.69 seconds; the 17-turn native suite had 2.62-second median delay. These are native measurements, not a container performance guarantee. Independent recognition scored 37 generated chunks without execution failures; number/time formatting explains the reviewed text/audio differences. Noise and human code-switching remain limitations.

See the [evaluation reports](https://github.com/bmaged23/linguabridge-en-de-sts/tree/main/evaluation/results) and [model notices](https://github.com/bmaged23/linguabridge-en-de-sts/blob/main/docs/reference/model-notices.md). The image excludes personal memory, recordings, raw test inputs and API keys. Finalized text and relevant memory go to Ollama Cloud; recognition and speech synthesis run locally.
