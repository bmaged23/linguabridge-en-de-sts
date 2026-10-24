# Evaluation reports

## Current release and measurements

| Report | Contents |
| --- | --- |
| [Verified GPU container release](../../evaluation/results/gpu-container-release-2026-10-24/README.md) | Exact public image digest, CI/delivery evidence, 94 regressions, 52 STT recordings, 17 browser turns, independent TTS audit and resources |
| [Completed-segment caching and GPU batching](../../evaluation/results/asr-latency-optimization-2026-10-24/README.md) | Matched native latency measurements and accuracy gates |
| [Mixed-speech robustness](../../evaluation/results/mixed-asr-fix-2026-10-24/README.md) | Story/control comparisons and remaining noise-related failures |
| [All evaluation methods and results](../../evaluation/README.md) | Versioned datasets, evaluation commands and report index |

## Historical report exports

| Folder | Contents |
| --- | --- |
| [`../advanced-story/`](../advanced-story/README.md) | Continuous bilingual story dashboards, synthetic audio, model alternatives and resource logs |
| [`../full-qa/`](../full-qa/README.md) | Earlier full browser/pipeline QA exports |
| [`../mixed-output-check/`](../mixed-output-check/README.md) | Earlier bilingual output and voice-routing examples |
| [`../tts-comparison/`](../tts-comparison/README.md) | Earlier synthesis model/voice experiments |

These reviewed exports stay at their existing URLs because dashboards, report links and audio players depend on those paths. Their older configurations and scores are not the current deployed release. Keep new private recordings and raw runs in ignored local folders; publish selected evidence with an explicit date, configuration and scope.

Other Markdown files in this folder provide earlier research and validation narratives. The [documentation index](../README.md) describes each one.
