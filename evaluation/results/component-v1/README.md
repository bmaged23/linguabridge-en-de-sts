# Component evaluation — 23 October 2026

These are local development measurements on a GTX 1660 Ti (6 GB), not independent benchmark certification. End-to-end browser testing is still in progress. Raw audio, outputs, traces and failed attempts are retained locally, outside Git.

## Speech recognition

Qwen3-ASR-0.6B on CUDA; progressive decoding includes cumulative previews every two seconds. Each row uses corpus edit counts, not a mean of per-recording percentages.

| Input | Recordings | WER | CER | Median decode | Median RTF |
|---|---:|---:|---:|---:|---:|
| Human English | 12 | 5.28% | 2.85% | 4.92 s | 0.519 |
| Human German | 12 | 4.76% | 2.28% | 6.00 s | 0.493 |
| Synthetic English | 8 | 1.31% | 0.23% | 2.63 s | 0.555 |
| Synthetic German | 8 | 7.69% | 5.32% | 3.38 s | 0.658 |
| Synthetic mixed | 8 | 2.99% | 0.91% | 3.70 s | 0.566 |
| Synthetic mixed + 20 dB noise | 8 | 10.18% | 7.39% | 3.73 s | 0.566 |

The 24 human clips are a small FLEURS read-speech subset (12 per language); training overlap is unknown. The synthetic recordings reuse this application’s voices and are not independent natural mixed-language speech. Fresh and progressive decodes reuse recordings, so 148 decodes are not 148 independent recordings. Model loading is excluded; the first decode includes kernel initialization. Silence/noise controls produced no Qwen words in either decoding mode.

## Speech synthesis and endpointing

- GPU Kokoro/Piper produced all **24 production-streamed outputs**. The unbounded-span comparison produced 23/24: one long English story exceeded the configured ONNX arena. This failure remains in the results.
- Independent Whisper large-v3 completed 191 decodes. Whole mixed-language outputs scored 44.31% WER because the recognizer can omit language spans. Known-language native chunks scored 6.15% English and 7.33% German WER across both synthesis variants; these are overlapping chunk measurements, not independent utterance counts or human quality ratings.
- Number formatting is literal: written digits and spoken number words can count as errors. No post-hoc number normalization was applied.
- Silero/endpoint tests: 200 cases across conversation, translation, teaching and interruption profiles; no missed speech, false detections or premature splits in this set. Boundaries are energy-derived. This does not measure full browser interruption latency.

## Translation

Twenty reference translations: EN→DE chrF++ **90.55**, BLEU **75.92**; DE→EN chrF++ **84.45**, BLEU **78.23**. All 20 passed the declared anchor and language-marker checks. GPT-OSS-120B initially scored 17/20 successfully; a separate higher-budget run repaired only the three unscored cases. Combined adequacy, faithfulness and target-language scores were 1.0; six simple positive/negative judge controls passed. This small synthetic set is not evidence of perfect translation or human-calibrated judging. Both protocols and the original incomplete coverage are retained in the JSON files.

## Resource scope

| Component run | Peak process-tree RSS | Peak whole-GPU VRAM | Mean CPU, one-core scale |
|---|---:|---:|---:|
| QWEN | 2287 MiB | 4233 MiB | 101.7% |
| TTS | 1275 MiB | 1579 MiB | 92.2% |
| VAD | 123 MiB | 0 MiB | 99.5% |

Runs are serial component measurements. GPU VRAM is device-wide and can include other consumers; process-tree RSS may double-count shared memory. These peaks cannot be added to estimate full-app memory. Cloud LLM inference resources are not measured locally.

See the adjacent JSON files for edit counts, decoding variants, resource statistics and limitations. Full conversation latency and actual playback interruption results will be reported separately after browser validation.
