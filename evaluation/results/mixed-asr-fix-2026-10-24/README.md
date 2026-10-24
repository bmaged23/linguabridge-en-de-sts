# Noisy mixed-language STT fix — 24 October 2026

The reported connected-story failure is fixed in the recorded full application run. German words are retained, the train is described as cancelled rather than full, and the ticket negation is preserved. This is an accuracy improvement with extra processing for long recordings, not a universal solution for noisy speech.

## Active recognition strategy

- Keep the existing Qwen3-ASR-0.6B CUDA FP32 model and greedy streaming previews.
- Requests shorter than 8seconds use the original fresh whole-utterance decode.
- For longer requests, the existing Silero model checks speech extent on CPU. Conservative margins retain context around it; internal low-energy pauses define ASR chunks. This does not split a user turn or change conversation VAD timing.
- Decode candidate chunks with 3 beams and 200 ms exterior digital padding. No words, language spans or references from the test dataset are given to the recognizer.
- Accept the joined result only when chunks identify both English and German with at least 4 words per language. Reject instruction echoes and implausible word rates; otherwise fall back to the original greedy whole decode.
- Monolingual/uncertain inputs retain the original decoder output. This protects accuracy but adds probing work for long monolingual recordings.

All runtime settings live under `stt` in root `config.py`. No new model weights, packages or cloud inference stages were added. A separate recurrent Silero state is reset for each ASR speech-boundary check. The ordinary microphone detector keeps its own state.

## Exact-waveform comparison

The study contains 8 new continuous synthetic stories: 4 development and 4 validation, each clean, 20 dB and 10 dB noise; 4 saved captures of one older story; and 24 existing FLEURS human clips. That is 52 waveforms, not 52independent people or stories. All compared methods receive matching WAV hashes and references.

The source-story split was frozen before recognition. Later guard revisions were rechecked on the same validation split after a human-control failure. These are development-validation results, not independently certified generalization estimates. Human calibration is read speech; no natural human code-switching benchmark is claimed.

| Cohort | Recordings | Baseline WER / CER | Enabled WER / CER | Fact checks before → after |
|---|---:|---:|---:|---:|
| Validation stories, all noise conditions | 12 | 14.32% / 11.21% | 9.40% / 6.72% | 35/45 → 39/45 |
| Validation, clean | 4 | 9.40% / 7.13% | 4.70% / 2.72% | 13/15 → 15/15 |
| Validation,20dB | 4 | 16.78% / 14.50% | 10.07% / 8.04% | 11/15 → 13/15 |
| Validation,10dB | 4 | 16.78% / 12.00% | 13.42% / 9.40% | 11/15 → 11/15 |
| Old story capture regressions | 4 | 13.95% / 6.40% | 4.65% / 1.24% | 8/12 → 12/12 |
| Human English controls | 12 | 3.77% / 2.05% | 3.77% / 2.05% | Not scored |
| Human German controls | 12 | 4.42% / 2.03% | 4.42% / 2.03% | Not scored |

Word/character error rates use the existing normalization and corpus edit-count aggregation. Numbers are not expanded; punctuation is removed. Fact checks are explicit case-specific pattern regressions, not a general semantic evaluator. All 24 human transcripts were unchanged and no individual human clip gained word errors.

FLEURS attribution: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), Conneau et al.(2022), CC-BY-4.0. The [pinned calibration manifest](../../datasets/fleurs-calibration-v1.json) documents the small sample and unknown training overlap. New synthetic texts and their development/validation assignments are in [the mixed-story dataset](../../datasets/mixed-asr-robustness-v1.json).

## What the earlier experiments found

Removing the context or changing padding alone did not give a consistent improvement. Plain pause chunking improved mixed recordings but damaged monolingual controls. An early adaptive version mistook an English instruction echo on a German recording’s noise tail for real code-switching; its human WER increased from 4.11% to 8.05%, so it was rejected. Speech extent, tail length, minimum language evidence and output plausibility guards address that failure. Original failed outputs are retained;`ablation-summaries.json` records the tested alternatives.

All 7 adoption gates passed: complete expected coverage, non-worsening validation WER/CER/fact scores, all captured facts retained, and no aggregate or individual human word-error regression. See`comparison.json` for every matched transcript, hash, score and processing time.

## Full application verification

The browser suite now defaults to bilingual-flows-v2. Its noisy story must explicitly retain train cancellation and ticket negation; mentioning Anna and Berlin is insufficient. The original v1 data/results remain unchanged.

- **17/17 turns passed all configured checks**, including the new meaning assertions. No JavaScript errors were recorded.
- The mixed connected story achieved **4.65% WER**, versus 25.58% in the earlier failing browser run.
- The independent Whisper audit completed **35/35 generated audio chunks**. Both the English and German spoken summaries passed the cancellation and ticket-negation checks.
- **83 application regression tests passed**, including padding invariance, silence handling, monolingual fallback, instruction-echo rejection and the original wrong-summary counterexample.

The 17 synthetic input turns measured corpus **2.31%WER / 0.51%CER**. These short scripted journeys are not a population accuracy estimate.

## Latency and resources

| Measurement | Value |
|---|---:|
| Median first audio after input recording ends | 2.75s |
| Long mixed-story first audio after recording ends | 8.46s |
| Spoken interruption, input start to interrupt packet | 0.904s |
| Peak browser/server/worker process-tree RSS | 3957MiB |
| Peak whole-device VRAM | 5008MiB |
| Mean CPU, one-core scale / machine-normalized | 65.9% / 5.5% |

The long-story latency is a real cost: it took 8.46 s versus 6.26 s in the prior failed-story run. Offline validation-story final processing rose from 2.92 s median to 5.35 s; long monolingual recordings also pay for probing before fallback. Short requests retain one original decode. Reusing finalized streaming chunks or batching is future performance work; no whole-application speedup is claimed.

Timing starts after the supplied recording, which includes 0.4 s trailing silence. Browser playback uses a silent real-time sink. RSS can double-count shared memory; GPU figures are device-wide. The main Gemma LLM runs on Ollama Cloud and its inference RAM/VRAM is excluded.

## Remaining failures

Six of 45 validation fact checks still fail under added noise. Examples include an omitted “not borrowed” clause and a distorted “after seven”. The fix resolves the reported train-story failure and improves the measured corpus; it does not eliminate all noisy transcription errors. Physical microphone/speaker echo, human listening quality and long-session endurance remain separate validation tasks.

## Reproduction and evidence

Use the commands in [the evaluation guide](../../README.md). Prepare human calibration audio and the speech models first. New synthetic audio can be regenerated; exact historical browser captures require the archived local WAVs or saved browser runs supplied with `--capture-run`. Fresh runs without those archives are not reproductions of the 52-waveform result. Raw audio, earlier failed attempts and per-run source snapshots stay in ignored `.evaluation-local/`.

Adjacent files preserve the exact-waveform comparison, ablation summaries, final configuration, native speech audit, raw ASR chunk results, final answers, semantic checks, playback/interruption and resource timing. The normal user memory database was never used. Changes are local and have not been pushed.
