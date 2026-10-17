# Local ASR comparison

Measured on 2026-10-16 using the GTX 1660 Ti (6 GiB).

These are synthetic diagnostics, not a public benchmark or an estimate of human microphone accuracy. English Kokoro and German Piper voices change at language boundaries. WER/CER lowercase and remove punctuation, but do not normalize numbers or German compound spelling. Zero errors means an exact match after this normalization.

## Selected experimental configuration

Qwen3-ASR-0.6B, CUDA FP32, eager attention, one request at a time, two-second partial-update interval. Earlier generated text is reused after two partials with five trailing tokens allowed to change. Audio remains cumulative; this is a Transformers prototype, not native vLLM streaming. It skips stale snapshots rather than accumulating a queue.

Fixed ASR context (no reference transcript is supplied):

> Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate.

This context was selected on the original four mixed examples and then frozen before evaluating four new examples. No model weights were trained.

## Original mixed examples

| Recording | Whisper WER / CER | Qwen WER / CER | First partial from start | Final after audio end |
| --- | ---: | ---: | ---: | ---: |
| en_de_en_travel | 66.7% / 67.5% | 0.0% / 0.0% | 2.98 s | 0.49 s |
| de_en_de_travel | 34.6% / 38.6% | 0.0% / 0.0% | 2.83 s | 1.53 s |
| en_de_en_learning | 0.0% / 0.0% | 0.0% / 0.0% | 2.66 s | 0.69 s |
| de_en_de_work | 44.4% / 46.5% | 0.0% / 0.0% | 2.77 s | 0.83 s |

These final delays exclude VAD endpoint waiting, cloud LLM latency, and TTS. Peak PyTorch allocated memory in this run: 3196 MiB. CUDA context and other process allocations are additional.

## New held-out mixed examples

| Recording | Whisper WER / CER | Qwen WER / CER | First partial from start | Final after audio end |
| --- | ---: | ---: | ---: | ---: |
| en_de_en_booking | 69.0% / 66.7% | 3.5% / 1.2% | 2.74 s | 0.54 s |
| de_en_de_grammar | 50.0% / 46.5% | 0.0% / 0.0% | 2.94 s | 0.96 s |
| en_de_en_invoice | 59.3% / 52.4% | 11.1% / 5.4% | 2.88 s | 0.56 s |
| de_en_de_hotel | 54.8% / 49.7% | 0.0% / 0.0% | 2.73 s | 0.62 s |

These final delays exclude VAD endpoint waiting, cloud LLM latency, and TTS. Peak PyTorch allocated memory in this run: 3209 MiB. CUDA context and other process allocations are additional.

## Separate English and German

| Recording | Qwen WER | Qwen CER | Complete-utterance decode |
| --- | ---: | ---: | ---: |
| english_complex | 0.0% | 0.0% | 2.45 s |
| german_complex | 0.0% | 0.0% | 3.00 s |
| english_numbers | 0.0% | 0.0% | 2.38 s |
| german_numbers | 0.0% | 0.0% | 3.88 s |

The same verbatim hint was used. Completed-utterance decoding still dropped mixed speech in some examples, so the progressive prefix-reuse behavior matters; the hint alone is insufficient.

## Other configurations tested

- Whisper large-v3-turbo: about 1.17–1.43 seconds per completed 7.4–9.9-second utterance in the existing app decoding test. It dropped whole language spans in several mixed examples.
- Qwen0.6B FP16: roughly 2.4–3.8 seconds for complete utterances except a prematurely truncated result. Short partials produced punctuation and severe latency; this setting was rejected.
- Qwen1.7B FP16: roughly 3.4–5.0 seconds per complete utterance, with translation and omission errors on mixed examples. Larger was not better for this diagnostic.
- Qwen0.6B FP32 without context: complete utterances took roughly 1.8–2.9 seconds except the truncated result. Progressive prefix reuse corrected three of four mixed examples, but lost the German → English → German travel passage.
- A generic bilingual hint recovered that travel passage but translated a German grammar example into English. The explicit verbatim hint above was selected instead.

All trial references, transcripts, checkpoint revisions, package versions, update timing, and measured allocation are preserved in [asr-evaluation-results.json](data/asr-evaluation-results.json). These trials are development experiments; prompt selection on the original examples can overfit.

## Reproduction

Use the separate `.venv-asr` with [exact evaluation dependencies](../../requirements/archive/asr-evaluation.txt). It does not replace the app environment.

```bash
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.evaluate_mixed_language_asr
.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.evaluate_asr_holdout
.venv-asr/bin/python -m scripts.evaluation.evaluate_qwen_asr --dtype float32 --progressive --rollback-tokens 5 --context "Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate."
.venv-asr/bin/python -m scripts.evaluation.evaluate_qwen_asr --suite codeswitch-holdout --dtype float32 --progressive --rollback-tokens 5 --context "Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate."
.venv/bin/python -m scripts.reports.export_asr_comparison
```

## Practical limits

Earlier text can be retained incorrectly by prefix reuse. Partials must remain replaceable and must not be sent to the conversation LLM as independent user turns. A completed turn still requires VAD endpoint detection. Test real human English, German, and mixed microphone speech before treating these measurements as representative.

This experiment does not change the app recognizer, prompts, or user memory. See [primary sources and alternative models](asr-model-research.md).
