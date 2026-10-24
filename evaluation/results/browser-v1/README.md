# Browser evaluation v1 — before fresh-final STT correction

All 17 turns completed; 16 passed the frozen functional checks and there were no JavaScript errors. This is **not 16 semantically correct replies**: manual trace review found two mixed-language STT failures which the original keyword checks missed.

| Check | Result |
|---|---|
| Translation request, confirmation and persistence | 6/6 functional checks passed |
| Explicit note and browser-reload recall | Passed |
| English→German→English voice order | Passed |
| Spoken interruption | Old chunk cancelled, 0.904 s from prerecorded input start |
| Four-second idle check after interruption | No false turn; microphone remained active |
| Median first audio after input recording ends | 2.43 s |

Latency includes endpointing and recognition after the recording finishes. The input has 0.4 s trailing silence, so this is not latency from the exact last spoken phoneme. The browser uses real-time prerecorded synthetic input and a silent output sink, not a physical microphone/speaker test.

## Quality failures retained

1. A noisy continuous mixed-language story had 25.58% WER. German “Unser Zug fiel aus” became “Our zug filled out”; the answer then incorrectly described a full train. Simple keyword checks passed despite this semantic error.
2. A German–English–German teaching request had 69.23% WER; STT translated German spans into English. Functional voice-presence checks did not catch that transcription failure.
3. The German teaching reply included an unsolicited English example, and the LLM marked the German sentence “Wie viel kostet bitte eine Fahrkarte nach Berlin?” as `[en]`. The parser correctly followed this incorrect marker, producing German text with the English voice. This failed the no-unrequested-voice check.

## Controlled diagnosis

Fresh Qwen decoding on the **same captured input WAVs** reduced the mixed story WER from 25.58% to 9.30%, and the teaching request from 69.23% to 0%. This is a targeted diagnostic, not a new aggregate benchmark. Final decoding now starts without the committed preview prefix; previews remain progressive. A separate browser-v2 experiment tests the changed application. Original v1 outputs and checks remain unchanged.

The wrong language marker remains an open issue. No human listening quality score is claimed. The next evaluation revision needs richer semantic reference assertions; passing infrastructure and keyword checks is insufficient.
