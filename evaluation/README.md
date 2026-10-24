# Speech and conversation evaluation

Latest: [noisy mixed-speech fix](results/mixed-asr-fix-2026-10-24/README.md), with frozen audio comparisons, expanded human controls, explicit fact checks and the stronger browser suite. It improves the reported failure and validation scores; noise-related errors and long-input latency remain. The [teaching/voice fix](results/teaching-voice-fix-2026-10-24/README.md) and [earlier evaluation](results/review-2026-10-24/README.md) preserve prior measurements.

`datasets/` contains versioned references frozen before each experiment. Revised regressions are labelled as development cases, not held-out benchmarks. `results/` contains reviewed aggregate evidence. Inputs, generated speech, raw model outputs and resource traces stay in ignored `.evaluation-local/`.

Measured component results: [23 October report](results/component-v1/README.md). See the [first full browser evaluation and failures](results/browser-v1/README.md) for conversation, memory, voice-switching and interruption results. Component scores do not establish full-flow reliability.

The [fresh-final follow-up](results/browser-v2/README.md) records improved mixed-language transcription, its latency tradeoff and remaining answer/voice failures.

The independent TTS recognizer is the existing local CTranslate2 conversion of Whisper large-v3. Its transcript is a noisy measurement, not human ground truth. English/German human-speech calibration and TTS round-trip scores are reported separately. The application recognizer is Qwen3-ASR-0.6B.

WER/CER use NFC-normalized, lowercased word tokens; punctuation is removed, German umlauts and ß are retained, and numbers are not expanded. CER includes spaces between normalized words. Corpus metrics sum edit counts and reference lengths rather than averaging per-sample WER. Empty-reference hallucinations are counted separately; their WER/CER are undefined.

Human calibration data: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0, Conneau et al. (2022). The preparation script pins the dataset revision and streams only the first requested test clips from each archive; selection does not depend on model results. Raw audio is local. This small read-speech subset has no naturally mixed-language speech and does not establish generalization.

## Running the evaluation

First complete the application/model setup in the main README. Run component stages serially so the GPU models do not compete for memory. Use a new output directory for each experiment; retain failed runs.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.evaluation.prepare_human_speech --help
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.run_component_benchmarks --help
```

Cloud-only evaluation has a small separate environment:

```bash
uv venv .venv-evaluation --python .venv/bin/python
uv pip install --python .venv-evaluation/bin/python --no-deps -r requirements/evaluation.txt
PYTHONPATH=src .venv-evaluation/bin/python -m scripts.evaluation.evaluate_cloud_translation
```

Cloud evaluation consumes your Ollama allowance. Budgets, judge settings and browser timeouts are in `evaluation/config.py`; local component parameters are in `EVALUATION_CONFIG` in root `config.py`. Existing cloud-v1 results used the older 256-token protocol; new runs default to cloud-v2 with low reasoning and a 1024-token budget. The separate calibration script is an audit/repair of the recorded v1 run, not a general rerun command.

For isolated browser journeys, install the Chromium version required by the pinned Playwright package before running:

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.run_browser_flow_suite --stage prepare
.venv/bin/python -m scripts.evaluation.run_browser_flow_suite --output .evaluation-local/my-browser-run
```

These tests use prerecorded synthetic speech delivered in real time, a real browser, the actual GPU workers and cloud model. Browser output uses a silent real-time sink. They create their own SQLite memory and must never point at the normal user database. Native microphone quality, speaker echo and human listening quality require separate manual validation.

Once the browser run has stopped, audit its generated speech independently. Run the audio audit serially with other GPU stages:

```bash
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.audit_conversation_audio \
  --run .evaluation-local/my-browser-run --output .evaluation-local/my-audio-audit
PYTHONPATH=src:. .venv-evaluation/bin/python -m scripts.evaluation.judge_conversation_flows \
  --run .evaluation-local/my-browser-run --output .evaluation-local/my-semantic-review
```

The semantic judge consumes cloud allowance and uses `FLOW_JUDGE` in `evaluation/config.py`. Check known positive/negative cases and review its explanations before treating ratings as evidence. Null/incomplete/invalid outputs are not silently converted into passing scores. Original failed judge protocols remain preserved locally.

The ordered teaching regression suite uses actual cloud replies and the application prompt/parser:

```bash
PYTHONPATH=src:. .venv-evaluation/bin/python -m scripts.evaluation.check_teaching_regressions \
  --output .evaluation-local/my-teaching-check
```

Its 12-request budget and versioned dataset are configured in `TEACHING_REGRESSION`. It checks requested items, voice-language presence and known German phrases incorrectly labelled English. Those regression checks do not replace semantic review or audio scoring. Use a new output directory for each pass; preserve failures.

## Mixed-speech recognition comparison

The frozen story split, method settings and exact-waveform adoption gates are in the [mixed-ASR report](results/mixed-asr-fix-2026-10-24/README.md). Prepare the existing FLEURS calibration audio first; synthetic preparation uses the installed GPU voices:

```bash
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.benchmark_mixed_asr \
  --stage prepare --inputs .evaluation-local/my-mixed-inputs \
  --capture-run .evaluation-local/my-browser-run
.venv/bin/python -m scripts.evaluation.benchmark_mixed_asr \
  --stage evaluate --inputs .evaluation-local/my-mixed-inputs --split development \
  --method configured_greedy --method speech_gated_pause_beam3 \
  --output .evaluation-local/my-mixed-development
```

After choosing a method on development stories, use a separate output directory with `--split validation`. All variants of each source story stay in one split. Missing optional historical captures or human WAVs are listed in the prepared manifest; no missing data is silently counted as evaluated. Historical exact-waveform results require those original archives. A new capture is a new experiment.

`compare_mixed_asr` matches waveform hashes/references and rejects a candidate if a human clip worsens even when an aggregate score improves. The published gate expects the specific 52-waveform study, including its four archived capture variants. Do not label a smaller or regenerated dataset as the same evaluation. Frozen inputs and per-run source snapshots remain in `.evaluation-local/`.

## Streaming ASR speed checks

Run `scripts.evaluation.replay_asr_streaming` against an archived input manifest to compare whole final decoding, exact completed-segment caching, and caching with GPU batches of two. It replays cumulative prefixes without real-time waits. Final decode timing excludes endpointing, queueing, cloud calls and TTS; summed inference time is not conversation duration. Use a fresh `--output` folder. The 12-recording published replay uses the archived manifest configured in `ASR_LATENCY_EVALUATION`; historical captures are required to reproduce the same sample selection.

The [latency report](results/asr-latency-optimization-2026-10-24/README.md) keeps quality and timing evidence separate. Before adoption, compare all 52 final transcripts with the prior guarded method using `compare_mixed_asr --baseline-method speech_gated_pause_beam3 --strict-quality`. Then run the browser suite to measure real-time first audio with the GPU STT and TTS workers co-resident.
