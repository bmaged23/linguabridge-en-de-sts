# LinguaBridge automated evaluation — 24 October 2026

Historical baseline. See the [later teaching and voice-marker fix](../teaching-voice-fix-2026-10-24/README.md) for updated results and the remaining noisy recognition issue. Original measurements below are preserved.

The current automated development evaluation is complete across STT, TTS, VAD, translation and full browser journeys. **Completion does not mean all quality checks passed.** Human listening, physical microphone/speaker echo, long-session endurance and cross-device reproducibility remain outside this pass. Results are local and have not been pushed.

## Coverage and outcomes

| Stage | Coverage | Outcome |
|---|---|
| Qwen STT component | 24 human clips + 48 clean/noisy synthetic recordings + 2 non-speech controls; fresh and progressive variants | 148 decodes completed; detailed corpus metrics in component-v1 |
| Kokoro/Piper TTS component | 24 texts × 2 synthesis paths | All 24 production-streamed outputs completed; one unbounded English baseline OOM, 47/48 overall |
| Independent Whisper component audit | Whole recordings, native chunks, human calibration and controls | 191 decodes completed; mixed-whole and native-chunk scores kept separate |
| Silero/endpointing | 200 mode/noise/control cases | No misses, false speech or premature splits in this set |
| Translation | 20 source/reference pairs | chrF++90.55 EN→DE /84.45 DE→EN; original judge truncations and repair retained |
| Corrected full browser run | 17 turns across 4 sessions | All completed; 15/17 frozen functional checks passed; no JavaScript errors |
| Actual conversation output audit | 38 generated native chunks | All 38 independently decoded with Whisper |
| Semantic review | 14 actual LLM answers | 14 valid assessments after 4 targeted controls; 3 local control replies use deterministic checks |
| CPU/GPU synthesis timing | 6 matched texts × 2 variants | All 12 CPU outputs completed; timing comparison below |
| Application regression tests | 65 tests | All passed |

The inputs cover English, German, English→German→English and German→English→German, including a connected noisy story. Human calibration is only 12 FLEURS read-speech clips per language; natural human code-switching is not represented. Synthetic recordings reuse application voices, and repeated variants are not independent samples.

## Full-flow recognition and latency

| Metric | Measured value |
|---|---:|
| Corpus input WER / CER, 17 turns | 3.47% / 1.03% |
| First audio after input recording ends, median /p95 | 2.63s / 4.37s |
| Cloud first text, median /p95 over14 calls | 0.49s / 1.53s |
| Spoken interruption, input start to interrupt packet | 0.908s |

The recording includes0.4s trailing silence. First-audio latency includes endpointing, final recognition, cloud and synthesis; it is not measured from an exactly labelled phoneme boundary. Browser output uses a silent real-time sink. A four-second post-interruption idle check produced no false turn and kept the microphone active.

Fresh final transcription fixes committed preview errors while preserving streaming previews. On the noisy story, WER changed from 25.58% to 9.30% in the full rerun, and its answer correctly says the train was cancelled. The mixed teaching request changed from 69.23% to 0%. These are targeted development cases; model weights are unchanged.

## Audio agreement

Whisper uses the language declared for each generated chunk:

| Chunk label | Chunks | Literal WER | Literal CER |
|---|---:|---:|---:|
| en | 24 | 7.61% | 4.42% |
| de | 14 | 14.29% | 10.60% |

These are text-to-generated-audio agreement proxies, not human voice-quality scores. They include generated cancelled/unplayed chunks; playback status is saved separately. An upstream German sentence mislabelled [en] is included in the English-labelled group, so the score cannot be attributed solely to Kokoro. Literal digits/spelled numbers inflate some edit counts. Short-word recognizer errors also need listening review.

## Model-judge scores

Scores below are mean ordinal ratings divided by 4, **not accuracy percentages**. All 14answers were assessed on each dimension.

| Dimension | Mean normalized score |
|---|---:|
| source fidelity | 0.911 |
| transcript faithfulness | 1.000 |
| completeness | 0.946 |
| voice markers | 0.929 |
| memory grounding | 1.000 |

The judge is GPT-OSS 120B on Ollama Cloud, low reasoning, 1536-token cap, temperature 0. Four simple controls check correct mixed markers, wrong language tags, an omitted requested item and a changed amount. They passed, but do not establish human calibration. The judge can over-penalize acceptable contextual examples or confuse an omission with source fidelity. Scores are advisory and are not a release gate.

Initial flow-judge-v1 lacked required explanations; v3 assessments also misread valid markers/actions. Those runs remain locally preserved and are excluded from these means. The later protocol explicitly explains marker persistence and supplies action evidence. No valid old low score was overwritten.

## Resource measurements

Full browser-run samples include the driver, browser, server and local workers:

| Metric | Measured value |
|---|---:|
| Peak process-tree RSS | 3835MiB |
| Peak whole-device VRAM | 4974MiB |
| Mean CPU, one-core scale / machine-normalized | 64.8% / 5.4% |
| Mean /peak GPU utilization | 9.3% / 55.0% |

RSS can double-count shared memory. VRAM is device-wide. Remote LLM compute is excluded. Wall-clock delivery, playback and cloud waiting reduce average GPU utilization; this is not a sustained-throughput benchmark.

## CPU versus GPU TTS

For the six matched production-streamed texts, median generation was **2.64s CPU versus0.47s GPU**, a5.61× ratio of medians. Model loading/warmup is excluded. The unbounded-span variant measured 5.32× on the same six texts. This is a synthesis-only timing comparison, not an end-to-end speedup. CPU audio did not receive a separate independent intelligibility audit. See `cpu-gpu-tts.json` for every pair.

## Outstanding findings

1. The tutor omits the requested explanation of because after explaining although. Its instruction to teach one idea at a time conflicts with explicit multi-item requests.
2. A German teaching answer introduces English examples and marks a German price-question sentence as [en], causing the wrong voice. The all-German expectation is stricter than the general bilingual-teaching policy, so an English example alone is not conclusive; the German text under [en] is a confirmed marker error.
3. Original keyword-only checks missed semantic damage in mixed STT. The supplemental semantic review improves coverage but still needs human calibration.
4. One unbounded long-text synthesis baseline exceeded its configured memory arena; production chunked synthesis completed its current cases.

## Evidence and reproduction

Adjacent JSON files contain sanitized per-case outputs, scores, timings, resource statistics and matched CPU/GPU timings. Raw native WAVs, actual STT captures, cloud outputs and resource traces remain in ignored `.evaluation-local/`; normal user memory was never used. See [evaluation instructions](../../README.md) for commands, budgets and normalization. The local listening page is `.evaluation-local/conversation-audio-v1/listening-review.html`; no human MOS is claimed.
