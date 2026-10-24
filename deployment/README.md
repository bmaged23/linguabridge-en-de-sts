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

## Pull and run without cloning

The public `gpu` tag is verified. Run these commands from a Linux or WSL2 shell.

Create a small working folder and a local `.env` file containing:

```dotenv
OLLAMA_API_KEY=your_own_ollama_api_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gemma4:31b
```

Then run:

```bash
docker pull bmaged23/linguabridge-en-de-sts:gpu
docker run --rm --name linguabridge --gpus all --env-file .env \
  -p 127.0.0.1:8766:8766 \
  -v linguabridge-memory:/app/data \
  bmaged23/linguabridge-en-de-sts:gpu
```

Wait for `Application startup complete`, open http://localhost:8766, allow microphone access and click **Start conversation**. The image contains the speech assets, so no Git checkout, Python installation or model download is required for this route. The cloud model needs your own Ollama account. The named volume keeps notes/history between runs; removing the container does not remove that volume.

To check Docker GPU access separately from cloud credentials:

```bash
docker run --rm --gpus all --entrypoint nvidia-smi bmaged23/linguabridge-en-de-sts:gpu
```

## Publication status

The public image is [bmaged23/linguabridge-en-de-sts:gpu](https://hub.docker.com/r/bmaged23/linguabridge-en-de-sts). [Delivery run 40142850698](https://github.com/bmaged23/linguabridge-en-de-sts/actions/runs/40142850698) passed exact-manifest public verification, anonymous pull and normal GPU startup. Tested source: `21902e9e8380eb82464039637dc356715375d53b`. Immutable digest: `sha256:04ec957cf2fb6df7e0d9b0ca378ac7f3f37e5fcc79ebcd04181ce36b2c08bcbd`. See the [full measured release report](../evaluation/results/gpu-container-release-2026-10-24/README.md).

The release workflow builds candidates on an ephemeral GitHub-hosted CPU runner and stages them in GitHub Container Registry. The trusted laptop runner pulls the exact candidate digest, warms real GPU models, evaluates the frozen recordings and browser flows, and publishes the passing image to public Docker Hub. This keeps the large build workspace off the laptop while GPU inference stays local.

GitHub-hosted offline tests cannot exercise this laptop’s GPU. The real-model gate needs the registered GPU runner online, the archived waveform/reference fixture bundle and the installed host browser. Raw fixture paths, credentials and user memory are excluded from Git and the image context. GHCR candidates are public build artifacts identified by source revision/digest. Supported Docker Hub release tags are published only after qualification. Candidates contain distributable models and source, with secrets, user memory and raw test fixtures excluded. See [GitHub’s registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) and [current registry billing policy](https://docs.github.com/en/billing/concepts/product-billing/github-packages).

For a slow first upload or delivery-only retry, the owner can run **Publish already qualified GPU image** with the matching source SHA, GHCR digest and qualification run ID. It checks successful build/GPU-evaluation steps and source identity before streaming the exact manifest registry-to-registry, then verifies anonymous pull and normal GPU startup. It does not waive evaluation. This release used that route after cancelling the slow local upload after its successful gates. The standard push workflow remains automatic; the delivery-only route is manually dispatched.

Automatic image releases follow changes to application/image inputs and evaluation code or datasets. Documentation, historical dependency archives and workflow-only maintenance do not trigger a model-image rebuild. The independent offline test workflow still runs on every push; Pages and Docker Hub documentation use their own triggers. To validate release-workflow changes end to end, explicitly run **GPU image release** with its manual dispatch. Earlier cancelled runs remain in GitHub history; they are not evidence of a failed model test.

Local builds remain available with the commands above; allow at least 35 GiB of working space for a cold build. Pulling and running a released image needs substantially less space, but Docker stores compressed layers and unpacked files. Keep extra space for updates.

Commit labels are applied after the expensive build layers. Separately pinned download inputs in `docker/model_config.py` keep unrelated runtime tuning from invalidating the model layer; an offline contract test rejects model/config mismatches. Version-pinned uv installs the exact distributions without precompiling every library. Byte-identical libraries with matching permissions are hardlinked across the isolated environments; different CPU/GPU ONNX providers remain separate. Documentation-only changes skip image delivery, and maintained Docker Hub overview changes use a lightweight metadata workflow.

## Dependencies

`requirements/release/app.txt` preserves installed application pins while omitting unused CUDA packages in the main interpreter. `requirements/release/tts.txt` shares the recognizer's CUDA libraries and adds the GPU ONNX runtime's declared `coloredlogs` and `sympy` dependencies. It intentionally contains only GPU ONNX Runtime; the Piper/Kokoro metadata names the CPU distribution, although both distributions expose the same `onnxruntime` module. Do not co-install them. The original environment snapshots remain in `requirements.txt` and `requirements/*-cuda.txt`.
