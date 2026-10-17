# LinguaBridge STS bilingual conversation demo

[Open the video player](../demo.html).

The story follows Bola and Lena planning a rainy Saturday at a Berlin food market. It differs from Voca's birthday-picnic demo: it uses English, German and mixed utterances, saves a note, interrupts an answer with a weather change, confirms interpreter mode, translates both sides of the conversation, returns to conversation mode and recalls the saved note.

The demo speaker is synthetic: Kokoro `am_michael` for English and Piper Thorsten High for German. The assistant uses its deployed Kokoro `af_heart` English voice and Piper Thorsten High German voice. The microphone input is synthesized, but **assistant responses are live, not mocked or prewritten**.

## What is recorded

- The actual local browser interface, AudioWorklet, Silero VAD, GPU Qwen3-ASR-0.6B, streamed Ollama Cloud `gemma4:31b` responses, GPU speech synthesis and browser playback acknowledgments.
- A real voice interruption while an assistant buffer is playing. Stopped audio is cut at its observed browser playback stop time.
- The original pauses and response times. There is no narrated walkthrough, voice-over, or sped-up conversation.
- A digital soundtrack reconstructed from the actual scheduled WebAudio buffers and playback start/stop times. Headless output is silent to avoid using physical speakers; microphone and speaker acoustics are not tested.
- Bilingual captions based on the source text used for user synthesis and the exact text used to synthesize completed assistant phrases. Interrupted assistant phrases are omitted from captions because not every displayed word was spoken. These are synthesis-text captions, not independent recognition of the soundtrack.

The demo uses a separate SQLite database under ignored `logs/demo-*/`, and real local speech-model weights. It does not use your private history or notes. Credentials are loaded server-side and are excluded from the recording artifacts. The regular app is restored after recording.

This is a functional demonstration, not a human speech or translation accuracy benchmark. Consult the [advanced story findings](../advanced-story/FINDINGS.md) for the broader tests and known limitations. The displayed transcript may contain recognition errors; the script does not replace it with the intended input.

## Files

| File | Contents |
| --- | --- |
| `bridge-demo.mp4` | H.264 Constrained Baseline / AAC-LC video (MP4 primary) |
| `bridge-demo.webm` | VP9/Vorbis alternative with an explicit codec type |
| `demo-cover.png` | README cover with the play button inside its left panel |
| `poster.png` | Actual app screenshot |
| `captions.vtt` | Bilingual synthesis-text captions |
| `transcript.txt`, `transcript.json` | Actual displayed transcript, including interruption labels |
| `recording.json` | Recording method, models, duration and validation metadata |
| `validation.json` | Media, sound and browser playback checks |

## Recording note

The saved demo contains all 13 turns, 26 transcript messages and one verified audio interruption cut. A post-capture cleanup timeout occurred after the audio buffers and video timeline had been saved; the video was encoded from those intact artifacts. The recorder now stops the session before exporting the large PCM data. The original page-error list was not preserved, so `recording.json` records it as unknown; the exported player is checked independently in `validation.json`.

## Reproduce

Complete the project setup in the root README, install FFmpeg/ffprobe, and install Playwright Chromium:

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python -m scripts.demo.record_bilingual_demo
.venv/bin/python -m scripts.demo.render_video_cover
.venv/bin/python -m scripts.demo.validate_demo_media
```

If Chromium is installed in a custom location, set `PLAYWRIGHT_BROWSERS_PATH` to that location before running the scripts. Recording temporarily stops this project's regular server to free GPU memory, uses isolated port 8767, then restores the regular app on 8766. It uses real cloud quota. End any microphone session before starting a recording.

Open `http://localhost:8766/docs/demo.html` with the regular app running, or serve the public `docs/` folder with any static HTTP server. A GitHub Pages deployment can serve the same files without running models. The published project uses GitHub Pages for the player. The recorder itself does not publish or change repository settings.

## Project name

The project is now **LinguaBridge STS — English–German Speech-to-Speech**. This recording predates the rename and preserves its original Bridge interface, spoken name and captions.

## Browser playback

The player prefers MP4 with explicit H.264/AAC codec information, falls back to the supported WebM format on a media error, and restores the play button when paused or finished. Open MP4 provides a direct alternative. The MP4 is exported with fast-start metadata for progressive loading.
