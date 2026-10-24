# Teaching completeness and voice-marker fix — 24 October 2026

Historical checkpoint. The [later mixed-ASR fix](../mixed-asr-fix-2026-10-24/README.md) resolves the reported train-story recognition failure and adds explicit meaning checks; it retains broader noise and latency limitations.

The two reported failures are resolved in the tested cases: the tutor explains every requested item before checking understanding, and German text is routed to the German voice rather than treated as English.

## What changed

- The tutor previously said to explain one idea and then ask a question. Explicit multi-item requests now take priority, so asking about both although and because receives both explanations and examples.
- Voice examples now follow the main reply language. German-led examples show an explicit return to `[de]` after an English quotation. `[en]` is explicitly a switch into English, never a closing tag for German.
- A current-turn language rule follows memory context. An earlier bilingual lesson no longer implicitly requests English examples in a later German-only question.

This keeps the same Gemma streaming call, parser, GPU Kokoro/Piper models and phrase pipeline. No classifier, new dependency, extra application cloud call or model download was added.

## Verification

| Check | Result |
|---|---|
| Ordered text-level teaching regressions | 12/12 in each of two passes; 12 unique cases, 24 requests |
| Full browser speech-to-speech suite | 17/17 frozen functional checks passed; no JavaScript errors |
| Application tests | 65 passed |
| Independent Whisper audit of generated audio | 32/32 chunks scored without execution failures |
| Audio from the two previously failing teaching turns | All 10 chunks had 0% normalized WER and CER against their synthesis text |

The 12 text cases cover multi-item requests, switching back to one language after bilingual context, English–German–English, and German–English–German. The first four reproduce earlier failures; the other eight were new when the regression suite was introduced. They became development regressions during tuning, not an independent held-out benchmark. The v2 dataset adds stricter checks after v1 keyword checks missed wrong language spans. Early unsuccessful prompt revisions are retained locally.

The actual synthesis trace shows English explanations/examples using Kokoro and German examples using Piper. The German station question uses Piper for its complete reply. Both requested concepts are explained before the follow-up question. The independent recognizer agrees with all 10 generated chunks after the documented normalization; this is an intelligibility proxy, not human voice-quality scoring.

## Separate recognition issue retained

The noisy mixed-language story remains variable. Its WER was 25.58% in this rerun, compared with 9.30% in an earlier fresh-final run, and the bad transcript led to an incorrect full-train summary. No STT settings or model weights changed in this teaching fix. The 17/17 number describes functional checks, which miss that semantic failure; it is not an overall correctness score.

Across this run's 17 synthetic inputs, corpus WER/CER was **7.51% / 4.02%**. Median first audio after the recording ended was **2.70 s**. The earlier baseline's 3.47%/1.03% and 2.63 s remain historical measurements, not a guarantee. A physical microphone/speaker or human listening evaluation has not been claimed.

## Evidence

Adjacent JSON files contain both successful regression passes, per-turn raw voice markers and text, native-audio recognition results, edit counts and timings. Raw recordings and earlier attempts stay in ignored `.evaluation-local/`. The ordinary user memory database was not used. Changes are local and have not been pushed.
