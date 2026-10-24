# Documentation and research reports

## Start here

- [Project overview and installation](../README.md)
- [Tools and commands](../scripts/README.md)
- [GPU container and release pipeline](../deployment/README.md)
- [Latest measured latency improvement](../evaluation/results/asr-latency-optimization-2026-10-24/README.md)
- [Exact dependency environments](../requirements/README.md)
- [Memory and voice commands](guides/memory-and-voice-commands.md)
- [Model notices and privacy](reference/model-notices.md)
- [All configuration parameters](reference/configuration-parameters.md)
- [Interactive animated voice pipeline](architecture.html)
- [Bilingual demo player](demo.html) and [recording details](demo/README.md)

## Runtime guides

| Guide | Contents |
| --- | --- |
| [Qwen deployment](guides/qwen-stt-deployment.md) | Isolated STT worker, precision and progressive decoding |
| [GPU TTS](guides/gpu-speech-synthesis.md) | Kokoro/Piper deployment, chunk concurrency and language routing |
| [GPU performance](reports/gpu-performance.md) | Measured latency and resource scope |
| [Runtime profile](reports/runtime-profile.md) | Earlier runtime measurements |

## Evaluation and validation

| Report | Contents |
| --- | --- |
| [Continuous bilingual story testing](guides/advanced-story-testing.md) | Reproduce the advanced pipeline tests |
| [Advanced story dashboard](advanced-story/index.html) | Per-sample stage outputs, timings, resources and audio |
| [Advanced story findings](advanced-story/FINDINGS.md) | Confirmed limitations and interpretation |
| [Full pipeline tests](reports/full-pipeline-validation.md) | Browser, cloud, speech and recovery checks |
| [ASR comparison](reports/asr-evaluation.md) | Recognition evaluation and synthetic-data limitations |
| [Mixed-language validation](reports/mixed-language-validation.md) | Code-switching test notes |
| [ASR alternatives](reports/asr-model-research.md) | Earlier recognizer research |
| [TTS comparison](reports/speech-synthesis-comparison.md) | Speech-generation quality, speed and resources |
| [General validation](reports/validation-history.md) | Earlier validation notes |

The demo and report directories retain their existing URLs so player links, synthetic audio and dashboards keep working. Raw private runtime material belongs in ignored `data/` and `logs/`; these documentation exports contain reviewed synthetic test material. Dates and model configurations in older reports describe those specific runs.

[Model revision/hash manifest](reference/model-manifest.json) · [Filename migration map](reference/filename-migration.json)
