# LinguaBridge STS — English–German Speech-to-Speech Translation

[GitHub source, demo and full documentation](https://github.com/bmaged23/linguabridge-en-de-sts)

Talk in English, German or both. The application supports conversation, confirmed translation-mode changes, teaching, persistent notes and voice interruption. Qwen3-ASR-0.6B recognizes speech locally; Silero VAD handles turn timing; Ollama Cloud Gemma 4 31B generates streamed replies; Kokoro English and Piper Thorsten German produce ordered audio chunks locally.

## Pull and run

Requires Linux x86-64 containers and an NVIDIA GPU exposed to Docker. The tested native GPU has 6 GiB VRAM. GPU/container verification details and exact release digest belong in the GitHub release record. This is not a CPU image.

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

## Evaluation and release process

The candidate image runs offline tests, warms real GPU models, compares 52 archived ASR waveforms against frozen references with per-recording WER/CER and meaning/routing gates, and completes 17 prerecorded browser conversation turns. Publication follows passing gates. The workflow pulls the published tag, verifies its identity and tests normal startup again. GitHub-hosted tests and the trusted GPU evaluation runner have separate roles.

Native measurements: caching completed ASR segments with GPU batches of two preserved all 52 final transcripts. A matched mixed story's first-audio delay fell from 7.77 to 2.69 seconds; the 17-turn native suite had 2.62-second median delay. These are native measurements, not a container performance guarantee. Independent recognition scored 37 generated chunks without execution failures; number/time formatting explains the reviewed text/audio differences. Noise and human code-switching remain limitations.

See the [evaluation reports](https://github.com/bmaged23/linguabridge-en-de-sts/tree/main/evaluation/results) and [model notices](https://github.com/bmaged23/linguabridge-en-de-sts/blob/main/docs/reference/model-notices.md). The image excludes personal memory, recordings, raw test inputs and API keys. Finalized text and relevant memory go to Ollama Cloud; recognition and speech synthesis run locally.
