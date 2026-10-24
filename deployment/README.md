# GPU container release

The container recipe is in `docker/Dockerfile`; it packages Python 3.12, three isolated interpreters, Qwen3-ASR-0.6B, Silero, Kokoro and Piper Thorsten. Model downloads use the pinned manifest. It does not contain an Ollama key, personal memory, raw evaluation inputs or the independent Whisper evaluator.

This route requires Linux x86-64 containers, an NVIDIA CUDA-compatible GPU and NVIDIA Container Toolkit (or Docker Desktop GPU support under WSL2). The native application was tested on a GTX 1660 Ti with 6 GiB VRAM. A CPU-only device is not supported by this configuration. Allow substantial image/build disk space in addition to the native environments.

## Build and test locally

From the repository root:

```bash
docker build -f deployment/docker/Dockerfile -t linguabridge-en-de-sts:release-candidate .
docker run --rm --gpus all --env-file .env \
  -p 127.0.0.1:8766:8766 \
  -v linguabridge-memory:/app/data \
  linguabridge-en-de-sts:release-candidate
```

Open http://localhost:8766 after model warmup. The browser captures and plays audio; the container does not need the host microphone device. Keep port binding on localhost. Your `.env` needs the three variables shown in `.env.example`. Persisted memory belongs to the named volume and survives container replacement.

## Publication status

The Docker Hub name reserved in the compose example is `bmaged23/linguabridge-en-de-sts:gpu`. **Publication and public-pull verification are pending.** The compose example becomes usable after that tag is published. Do not treat the repository URL or Docker recipe as evidence that a public image exists.

The release gate must test the candidate image with real GPU model inference and full browser flows before publishing it. GitHub-hosted offline tests cannot measure this laptop's GPU models or reach its local workers. A trusted GPU runner and archived waveform/reference fixtures are required for the real-model gate. Raw fixture paths, private credentials and user memory are excluded from Git and image context.

## Dependencies

`requirements/release/app.txt` preserves installed application pins while omitting unused CUDA packages in the main interpreter. `requirements/release/tts.txt` shares the recognizer's CUDA libraries and adds the GPU ONNX runtime's declared `coloredlogs` and `sympy` dependencies. It intentionally contains only GPU ONNX Runtime; the Piper/Kokoro metadata names the CPU distribution, although both distributions expose the same `onnxruntime` module. Do not co-install them. The original environment snapshots remain in `requirements.txt` and `requirements/*-cuda.txt`.
