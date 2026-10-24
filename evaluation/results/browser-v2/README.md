# Browser evaluation v2 — fresh final transcription

All eight selected story/memory/teaching turns completed, with no JavaScript errors. Six passed all functional checks. This follow-up changes final STT decoding; the previous full 17-turn baseline remains preserved.

| Mixed-language case | v1 WER | v2 WER | v1 first audio after recording | v2 first audio after recording |
|---|---:|---:|---:|---:|
| mixed-connected-story | 25.58% | 11.63% | 1.92 s | 4.85 s |
| teaching-de-en-de | 69.23% | 0.00% | 1.89 s | 2.73 s |

The mixed story answer now correctly says the train was cancelled. The mixed teaching input is transcribed exactly after normalization. Browser captures are new real-time deliveries of the same source recordings, not byte-identical captured audio; cloud timing/answers also vary. A separate controlled decode of the original captures reduced story WER to9.30% and teaching WER to0%. These are targeted development examples, not an overall generalization claim.

## Remaining failures

- The mixed teaching answer discusses although but omits because, failing the completeness keyword check.
- The German teaching answer emits an English voice where the frozen reference expects only German. The v1 run additionally demonstrated a German sentence incorrectly tagged as English by the LLM.
- Fresh decoding improves these two STT cases but adds finalization latency. It does not solve language-marker reliability or answer completeness.

Application regression suite: **58 passed**. No new model weights were installed. These results have not yet been published to GitHub or Docker Hub.
