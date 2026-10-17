# LinguaBridge STS — English–German Speech-to-Speech Translation & Conversation

**Speak English, German, or a mixture of both—and hear the AI respond.** LinguaBridge is an audio-to-audio app for bilingual conversation, interpreting between two people, and practising German or English. It remembers your confirmed mode, saved notes, and conversation context across sessions.

The pipeline is **microphone → speech-to-text (STT) → language model (LLM) → text-to-speech (TTS) → speakers**. Speech recognition and speech synthesis run on your local NVIDIA GPU. The language model runs on Ollama Cloud. This is a modular speech-to-speech system: the LLM receives transcribed text rather than raw audio.

Built by **Bola Maged Ramses Nessim**.

[Demo](#demo) · [Animated pipeline](#animated-pipeline) · [Models](#models) · [Installation](#installation--clone-to-first-conversation) · [Configuration](#important-configuration) · [Troubleshooting](#troubleshooting)

## Demo

[![Play the bilingual speech-to-speech demo](docs/demo/demo-cover.png)](https://bmaged23.github.io/linguabridge-en-de-sts/demo.html?v=2)

Click the play cover to watch a **three-minute, 13-turn story**: planning a Saturday in Berlin, interrupting an answer, switching between English and German, confirming translator mode, and recalling a saved detail. User voices are synthesized; assistant replies come from the real running pipeline. The original recording uses the earlier “Bridge” branding. [Recording details and transcript](docs/demo/README.md).

## Animated pipeline

[![Animated speech-to-speech pipeline: models, IN/OUT ports, moving arrows, memory feedback and interruptions](docs/assets/pipeline-animation.gif)](https://bmaged23.github.io/linguabridge-en-de-sts/architecture.html?v=2)

The animation plays directly in this README. Click it for the interactive version, where you can pause the animation and select individual entities. It illustrates the flow, not measured latency.

1. **Capture speech.** Your browser requests echo cancellation and noise suppression, then sends 16 kHz mono audio to the local server in 512-sample frames—32 ms each.
2. **Detect a turn.** Silero VAD distinguishes speech from silence. Each mode has its own pause allowance; translation waits longer so speakers can finish their sentences.
3. **Transcribe.** Qwen3-ASR produces replaceable previews while you speak, then a final transcript. It attempts to preserve English and German words in their original language. Only finalized text goes to the LLM.
4. **Build the request.** The app adds the confirmed mode, target language, and relevant local memory. Gemma streams its answer or translation through Ollama Cloud.
5. **Choose the voices.** The LLM marks English/German text spans. A local parser removes these markers from the transcript and sends short phrases to Kokoro or Piper. English → German → English output can switch voices within a reply while preserving order.
6. **Play and remember.** The browser plays each completed audio chunk. Only fully played assistant phrases count as heard history. New speech stops playback, clears pending audio, and cancels the previous response.

STT uses progressive cumulative-audio decoding with prefix reuse. TTS streams completed **phrases**, rather than generating individual audio tokens. One STT process and one shared TTS process keep their models loaded between turns.

## Models

| Stage | Model / voice | Location | Active settings |
| --- | --- | --- | --- |
| Speech detection | Silero VAD ONNX | Local CPU | 16 kHz audio, 32 ms frames |
| STT | Qwen3-ASR-0.6B | Local GPU | FP32, eager attention, progressive previews |
| LLM | `gemma4:31b` | Ollama Cloud | Streaming on, thinking off, temperature 0.3 |
| English TTS | Kokoro 82M ONNX, `af_heart` | Local GPU | Speed 1.05, 24 kHz output |
| German TTS | Piper `de_DE-thorsten-high` | Local GPU | Length scale 1.0, 22.05 kHz output |
| Memory | SQLite | Local disk | Mode, notes, recent turns and summaries |

No local LLM installation is required. An Ollama Cloud account with access to the configured model is required. Model weights download separately from the repository. See the [pinned model manifest](docs/reference/model-manifest.json) and [model notices](docs/reference/model-notices.md).

## Installation — clone to first conversation

### 1. Check your machine

| Requirement | Tested setup |
| --- | --- |
| Operating system | Linux x86_64 or WSL2 |
| GPU | NVIDIA GPU; tested on GTX 1660 Ti with 6 GB VRAM |
| Python | 3.12; package snapshots captured on 3.12.15 |
| RAM | Approximately 7.7 GiB available to the tested WSL instance |
| Storage | Allow at least 15–20 GB for the environments, models and caches |
| Browser | Chrome, Chromium or Edge, with microphone permission on localhost |
| Cloud access | Internet connection and an Ollama API key |

On Windows, run the following commands **inside WSL2**, and use your normal desktop browser for the microphone. Native Windows and macOS setups have not been validated.

Check GPU access before installing:

```bash
nvidia-smi
```

### 2. Install uv and Python

Install Git if it is not already available. Install [uv using its official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal so `uv` is on your PATH, then install Python 3.12:

```bash
uv --version
uv python install 3.12
```

### 3. Clone the project

```bash
git clone https://github.com/bmaged23/linguabridge-en-de-sts.git
cd linguabridge-en-de-sts
```

Run all remaining commands from this repository folder.

### 4. Create the environments and install the exact requirements

The app, recognizer and GPU voices have separate environments so their ONNX Runtime and PyTorch dependencies do not conflict.

```bash
# Main app, CPU VAD and testing tools
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --no-deps -r requirements.txt

# Qwen GPU speech recognition
uv venv .venv-asr --python .venv/bin/python
uv pip install --python .venv-asr/bin/python --no-deps -r requirements/stt-cuda.txt

# Kokoro / Piper GPU speech synthesis
uv venv .venv-tts-gpu --python .venv/bin/python
uv pip install --python .venv-tts-gpu/bin/python --no-deps -r requirements/tts-cuda.txt
```

These snapshots pin the installed versions, including transitive dependencies. `--no-deps` reproduces those versions without adding a separately resolved dependency set. Keep CPU `onnxruntime` and `onnxruntime-gpu` in their separate environments. The optional Qwen TTS experiment is not needed.

**Reproducibility note:** the app and ASR environments pass dependency checks. The captured GPU TTS environment has known metadata conflicts and two declared missing dependencies despite working in the tested app. A clean-machine reinstall has not yet been validated. See the [environment details and exact snapshot caveats](requirements/README.md). `uv sync` uses an older main-app lock and does not replace the three steps above.

### 5. Add your Ollama Cloud credentials

```bash
cp .env.example .env
chmod 600 .env
```

Open `.env` in your editor and fill in your key:

```dotenv
OLLAMA_API_KEY=your_own_ollama_api_key
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gemma4:31b
```

Keep an already configured `.env` when updating an existing checkout. It is ignored by Git. Your key stays server-side, and cloud calls consume your account allowance.

### 6. Download the speech models

```bash
.venv/bin/python -m scripts.setup.download_speech_models
```

This downloads Qwen3-ASR, Silero VAD, Kokoro and its voice embeddings, and the German Piper voice/configuration. It uses pinned revisions and verifies recorded hashes. No training dataset is needed to run the app.

### 7. Validate the settings and start the app

```bash
.venv/bin/python -m scripts.validation.validate_configuration
.venv/bin/python -m scripts.application.start_voice_app
```

Wait for **Application startup complete**, then open **http://localhost:8766**. The first launch loads and warms the GPU models. Keep the terminal running. Use the launcher above so speech workers receive the correct CUDA library paths.

### 8. Start your first conversation

Click **Start conversation**, allow microphone access, and speak. Headphones help prevent speaker echo.

- Try English: “Help me plan a weekend in Berlin.”
- Try German: “Was können wir am Samstag unternehmen?”
- Try mixed speech: “Explain what Ich freue mich darauf means.”
- Speak during a reply to interrupt it.
- Select **Translate**, then confirm the change to interpret English ⇄ German.

Stop capture with **End conversation**. Stop the server with **Ctrl+C**. Restart the server after editing `config.py`.

## Conversation, translation and teaching

| Mode | What it does | Pause allowance | Maximum utterance |
| --- | --- | ---: | ---: |
| Conversation | Answers naturally; matches your language or selected reply language | 700 ms | 30 s |
| Translation | Interprets EN → DE or DE → EN; translates questions instead of answering them | 1,200 ms | 45 s |
| Teaching | Gives explanations and language examples | 1,000 ms | 45 s |

Translation requires confirmation before it starts. Once confirmed, it stays active until you change modes—even after a refresh or restart. Direction follows the recognized language; speakers do not have to alternate. There is no idle timeout while waiting for the next speaker. Overlapping speakers are not separated or identified.

Modes can be selected in the UI or through explicit commands such as “LinguaBridge, translator mode” and “LinguaBridge, conversation mode.” `Bridge` and `Brücke` remain compatible prefixes.

## Important configuration

All tuning lives in [`config.py`](config.py). Credentials and cloud model selection live in `.env`. The [complete parameter reference](docs/reference/configuration-parameters.md) lists every setting.

| Area | Configuration | Default and effect |
| --- | --- | --- |
| Audio | `audio.input_sample_rate`, `frame_samples` | 16,000 Hz / 512 samples; matches the microphone worklet |
| Speech start | `vad.speech_threshold`, `minimum_rms`, `min_speech_ms` | 0.5 / 0.002 / 200 ms; prevents isolated noise from starting a turn |
| Pause allowance | `modes.<mode>.end_of_turn_silence_ms` | 700 / 1,200 / 1,000 ms; larger values wait longer before replying |
| Interruptions | `interruptions.enabled`, `min_speech_ms`, `speech_threshold`, `minimum_rms` | Enabled / 320 ms / 0.8 / 0.006; confirmed speech cuts off the current reply |
| STT compute | `stt.device`, `dtype`, `attention` | CUDA / FP32 / eager; the validated Qwen runtime |
| STT previews | `stt.partial_interval_ms`, `window_seconds` | ≥2,000 ms / 46 s cumulative window; previews stay local |
| STT decoding | `stt.max_new_tokens`, `unfixed_chunk_num`, `rollback_tokens` | 256 / 2 / 5; controls output length and progressive text-prefix reuse |
| LLM response | `llm.temperature`, `max_response_tokens`, `think` | 0.3 / 512 / false; bounded streamed replies |
| Cloud deadlines | `llm.first_token_timeout_seconds`, `stream_idle_timeout_seconds` | 15 s / 15 s; avoids waiting indefinitely for reply text |
| Voice selection | `tts.voices.en`, `tts.voices.de` | Kokoro `af_heart` at speed 1.05; Piper Thorsten High |
| TTS phrase chunks | `tts.chunking.min_characters`, `max_characters`, `flush_timeout_ms` | 40 / 180 / 350 ms; balances early audio and phrase size |
| TTS memory arena | `tts.cuda.gpu_mem_limit` | 768 MiB per model arena; CUDA context and other allocations are additional |
| TTS concurrency | `tts.parallel_chunks`, `parallel_chunks_by_language` | Upper bound 2; English 1, German 2; output order is preserved |
| Backpressure | `workers.llm_token_queue_size`, `tts_queue_max_phrases`, `playback_queue_max_chunks` | 32 token entries / 3 phrases / 3 audio chunks |
| Worker counts | `workers.stt_processes`, `tts_processes`, `max_sessions` | One recognizer, one TTS process, one browser session |
| Memory | `memory.recent_turns`, `summary_trigger_turns` | 12 recent turns; summarize older context after 20 turns |

Voice interruptions are on by default in all modes. The browser requests echo cancellation. Turning off **Interrupt replies by voice** pauses microphone input during reply playback and a 250 ms echo tail. A forced translation guard is also configurable. Timing is evaluated in 32 ms frames, so a 700 ms pause ends at approximately 704 ms.

## Memory and privacy

SQLite stores the confirmed mode and reply-language preference, explicit notes, separate histories for each mode, and summaries of older conversation/teaching context. Ask “LinguaBridge, remember that my German level is A2” to pin a note. Only completed assistant phrases are stored as heard. Summaries may lose detail; pin facts that must remain exact.

Use **Clear this history** to clear the current mode's history, or the memory panel to erase all notes and histories. [Memory and voice-command details](docs/guides/memory-and-voice-commands.md).

Audio is processed locally. Finalized text, selected history and relevant notes/summaries are sent to Ollama Cloud. Local transcript memory is enabled by default. `.env`, `data/`, `logs/`, `models/` and all virtual environments are excluded from Git. Public demo/report audio is synthetic.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| No microphone input | Allow browser microphone access, select the right device in Audio settings, and use localhost |
| CUDA or model loading error | Run `nvidia-smi`, download the models, and launch through `scripts.application.start_voice_app` |
| Cloud response fails | Check your key, selected model access and account allowance; run the cloud check below |
| Cannot interrupt / speaker echo | Enable voice interruptions; use headphones, or disable interruptions to pause input during playback |
| Translation did not start | Confirm the requested mode change first |
| Old information affects replies | Review saved notes and clear the relevant mode's history |
| Demo video will not start | Use **Open MP4** on the player page; the player also has a WebM fallback |

```bash
# Offline behavior tests
.venv/bin/python -m pytest -q

# Real cloud authentication check; consumes account allowance
.venv/bin/python -m scripts.validation.validate_cloud_connection
```

English/German recognition can still omit words or translate mixed speech, and a reply that omits language markers can use the wrong voice for a span. The demo is not a human microphone accuracy benchmark. See the [advanced story findings](docs/advanced-story/FINDINGS.md) and [pipeline validation report](docs/reports/full-pipeline-validation.md).

## Project structure and further documentation

```text
src/voice_bridge/     Voice pipeline, routing, memory and API
web/                 Browser interface and microphone worklet
config.py            Runtime model, VAD, queue and memory settings
requirements/        Exact GPU environment snapshots
scripts/             Setup, startup, validation, demos and evaluations
  documentation/     Animated diagram builder and GIF renderer
docs/                Guides, parameter reference, reports and demo media
tests/               Offline regression tests
```

[Documentation index](docs/README.md) · [Tool commands](scripts/README.md) · [Dependency environments](requirements/README.md) · [Model licenses and notices](docs/reference/model-notices.md)
