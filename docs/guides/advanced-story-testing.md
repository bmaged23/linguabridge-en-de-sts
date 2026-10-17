# Continuous bilingual story tests

Each input is a single continuous narrative about Anna travelling to Berlin, a cancelled train, a request not to buy tickets, and an interview at ten rather than nine. English and German versions tell the same events. Mixed versions change languages six times inside the story, including quoted speech, negation and numbers. They are not lists of unrelated prompts.

The frozen suite has 40 runs: 28 mixed-language and 12 monolingual. Conditions include clean speech, low volume, additive noise at several signal-to-noise ratios, pauses at language boundaries, and identical repeated inputs to inspect stability and growing conversation memory. Source voices change with language; that makes these synthetic diagnostics, not a human microphone benchmark.

```bash
.venv/bin/python -m scripts.story_tests.prepare_story_inputs logs/advanced-story/my-run
# Point this variable to your installed Playwright browser directory, if needed.
PLAYWRIGHT_BROWSERS_PATH=/mnt/d/voice-bridge-tools/playwright \
  .venv/bin/python -m scripts.story_tests.run_story_pipeline_tests --folder logs/advanced-story/my-run
.venv/bin/python -m scripts.reports.export_story_report logs/advanced-story/my-run
```

The runner starts an isolated QA server on port 8767 and preserves the normal application's memory. It stops the normal project server to free GPU memory and restarts it in a `finally` block. Use `--limit 1` for a smoke run. The QA-only audio audit endpoint is not installed in the normal app.

Every sample saves the continuous input WAV, exact reference, STT previews and final text, browser-captured STT input, raw LLM tokens including language markers, parsed language segments, native English/German TTS WAVs, playback acknowledgments, errors, timing and resource summaries. Generated phrases are also re-recognized after browser timing ends. That is a model agreement diagnostic, not a human pronunciation score.

`resources.jsonl` samples the local server process tree and the whole GPU approximately twice per second. It includes process RSS, CPU usage, system RAM, VRAM, GPU utilization, temperature and power when supported. RSS sums may count shared pages more than once; CPU percentages are provided both per core and normalized to the machine. Whole-GPU values include other GPU applications. Short stages may fall between samples, so sampled peaks are lower bounds. Cloud LLM RAM/VRAM cannot be measured from this computer.

Stage timing separates final-STT queue-plus-decode latency from decoder time, cloud time to first text, TTS inference and overall microphone-to-completed-playback time. Resource sampling and native trace timestamps use the same monotonic clock. The first request can include cold-kernel overhead. Reference words never enter the recognizer prompt.

Alternative recognizers use the **same browser-captured audio**, with automatic language detection and a transcription task. Final-only decoding time is not directly equivalent to progressive recognition time. Stop the normal GPU app before running candidates; restart it afterwards:

```bash
.venv/bin/python -m scripts.application.start_voice_app -m scripts.story_tests.compare_story_recognizers logs/advanced-story/my-run --model large-v3
.venv/bin/python -m scripts.application.start_voice_app -m scripts.story_tests.compare_story_recognizers logs/advanced-story/my-run --model turbo
```

Model sources: [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Whisper large-v3 conversion](https://huggingface.co/Systran/faster-whisper-large-v3). Candidates are measurements, not assumed upgrades.

Use `--resume` to retain already completed samples after an interrupted run. `prepare_advanced_qa.py --extended` generates ten more connected-story inputs with bilingual retelling requests and approximately 39-second stories to exercise the current 30-second turn limit. The saved-word review distinguishes structural failures from semantic inference heuristics.

Whisper candidates enable the installed runtime’s `multilingual=True` with standard 30-second windows by default; use `--whisper-chunk 5` for the short-window experiment, allowing built-in language detection to change between decoding windows. This remains automatic recognition; it does not force the entire recording into one language. Qwen fresh-final and cumulative-prefix replay are evaluated separately.
