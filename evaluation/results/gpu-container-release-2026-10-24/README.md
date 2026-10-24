# GPU container qualification

Source `21902e9e8380eb82464039637dc356715375d53b` passed the real GPU model and browser gates in [release run 40140657528](https://github.com/bmaged23/linguabridge-en-de-sts/actions/runs/40140657528). The exact image was published and passed anonymous pull, real GPU warmup, health and UI startup in [delivery run 40142850698](https://github.com/bmaged23/linguabridge-en-de-sts/actions/runs/40142850698), which is green. [Docker Hub](https://hub.docker.com/r/bmaged23/linguabridge-en-de-sts) serves `gpu` and the source-SHA tag. The original qualification run was cancelled during its slow laptop upload, after its model/browser gates had passed; hosted delivery resumed the same image without rebuilding.

Immutable image:

```text
bmaged23/linguabridge-en-de-sts@sha256:04ec957cf2fb6df7e0d9b0ca378ac7f3f37e5fcc79ebcd04181ce36b2c08bcbd
```

See [release identity and verification](release.json) and the [pull/run guide](../../../deployment/README.md). The compressed layers total about 5.4 GiB; Docker also needs space for unpacked content and updates. The anonymous pull reused local content-addressed layers. This verifies one Linux/amd64 NVIDIA setup, not every device.

## Model and workflow results

| Check | Result |
| --- | ---: |
| Offline regressions | 94 passed |
| Frozen STT recordings | 52/52, unchanged final transcripts |
| Full browser conversation turns | 17/17 passed |
| Browser JavaScript errors | 0 |
| First output audio after recording ends | 2.533 s median; 1.179–3.462 s range |
| Spoken interruption onset to interrupt packet | 1.104 s in the interruption case |
| English TTS synthesis batch | 0.331 s median, 22 batches |
| German TTS synthesis batch | 0.161 s median, 13 batches |

TTS durations measure model synthesis batches, sometimes containing several phrases; they exclude audio playback. The 17 cases cover translator confirmation/persistence, repeated same-language turns, saved notes, recall after reload, a noisy connected mixed story, bilingual teaching, silence and interruption/recovery. An interruption is detected after sufficient speech evidence; it is not instantaneous.

| STT cohort | Recordings | WER | CER |
| --- | ---: | ---: | ---: |
| Human English controls | 12 | 3.77% | 2.05% |
| Human German controls | 12 | 4.42% | 2.03% |
| Mixed validation stories, clean + noisy variants | 12 | 9.40% | 6.72% |
| Captures of the earlier regression story | 4 | 4.65% | 1.24% |

These scores match the frozen references; unchanged errors remain. The 52 recordings include development examples and repeated noisy/captured variants, so they are not 52 independent unseen stories. The full browser test uses prerecorded microphone input in real time and a silent audio output sink. Bola separately live-tested the native application. Neither test establishes quality for every accent, room or physical speaker/microphone combination.

## Independent output-audio audit

Whisper large-v3 independently decoded **37/37 generated audio chunks** with zero execution failures. English (22 chunks) had **4.04% WER / 2.84% CER**; German (15 chunks) had **9.09% WER / 7.08% CER** against text sent to TTS. Nine nonzero-WER chunks differed in number/time spelling, such as “ten” versus “10”. One German chunk differed lexically: reference “Ihr Zug” was recognized as “Der Zug”. This audit cannot attribute that difference to synthesis versus the independent recognizer, and it remains recorded. Literal scores retain all differences. These are text/audio agreement scores, not human pronunciation or naturalness ratings. Cancelled/unplayed chunks are included. See [audit summary](audio-audit-summary.json) and [every nonzero-WER comparison](audio-disagreements.json).

## Resource observations

Measured on GTX 1660 Ti 6 GiB under Docker Desktop/WSL2. Qwen3-ASR and both TTS voices run on the local GPU; Silero VAD and application orchestration run locally on CPU. The main Gemma LLM runs on Ollama Cloud, so its RAM, GPU and compute are excluded.

| Observation | Value |
| --- | ---: |
| Container memory, highest observed | 2,600.96 MiB |
| Docker CPU, mean / highest observed | 42.00% / 210.77% |
| Whole-device VRAM, highest observed | 4,954 MiB |
| Whole-device GPU utilization, highest observed | 44% |

CPU 100% means one logical core. These 38 coarse observations include startup; sampled peaks can miss brief spikes. Container memory excludes the host browser/evaluator. VRAM is whole-device usage, not a separate allocation measurement for each model. Native and container latency runs are separate observations, not a controlled container-versus-native speedup comparison. Provider usage was recorded for 13 completed responses: 7,279 input and 415 output tokens. No monetary cost estimate is inferred from these counts.

## Failure found and corrected

The preceding candidate passed all 52 STT recordings but only 15/17 browser tests. A brief burst of microphone frames could trigger the immediate 64 KiB backlog stop. The client now preserves frame order in a bounded queue (64 pending frames, 8 ms drain interval), allowing a short burst to drain while sustained overflow still stops capture. A Node regression checks burst delivery order, overflow and cleanup. The affected sessions passed 8/8 in a focused check, followed by 17/17 in the final image. The evaluation input step also has a duration-plus-15-second timeout, preventing a stopped audio context from hanging the test indefinitely. Failed raw runs remain archived locally; they were not replaced with passing outputs.

[Machine-readable cohort, per-case, latency and resource measurements](metrics.json). Model output interpretation, synthetic fixtures and cloud variability remain limitations.
