# Validation notes

The practical configuration uses a multilingual Whisper Turbo GPU worker,
Gemma 4 31B on Ollama Cloud, and Kokoro af_heart for English and Piper Thorsten High for German in one CPU worker.

Local checks use synthetic test speech and English/German sentences. They are
integration checks, not a human-speech benchmark or a WER/CER evaluation.

- Offline: persistence/reopen, independent histories, deterministic mode rules,
  explicit notes, summary boundaries, source/target routing, VAD pauses,
  microphone frames/resampling, playback backpressure, interruption history.
- Speech: synthesize each language using its own Piper voice; transcribe with
  the GPU model; assert EN/DE confidence and expected content.
- Cloud: verify the configured model is accessible and translates both ways.
- Transport: send real-time microphone frames, receive live translation and
  synthesized PCM, acknowledge playback, and wait through a six-second pause.
- Browser: virtual WebAudio microphone, real AudioWorklet/WebSocket/VAD/STT/
  cloud/TTS/audio playback, voice mode change, memory surviving reload, explicit
  note recall in teaching, both translation directions, responsive layout.

Test reports, screenshots and synthetic audio are written under ignored `logs/`.
A future narrated-free demo should follow `description/PROJECT_BRIEF.md`; no
new demo video is claimed by these checks. A physical microphone/speaker test
is still needed for room echo, device permissions, accents and real turn-taking.

## Current local run

- 22 offline tests passed, including weak-confidence routing, spoken clarification, confirmation expiry/cancellation/persistence, and command isolation.
- Kokoro English and Piper German with GPU Whisper Turbo round trips passed.
- Gemma 4 cloud access and live translation passed.
- Live transport EN→DE, EN→DE, DE→EN and six-second idle pause passed.
- Browser voice mode change, saved-note recall, both translation directions,
  audio playback, desktop/mobile layout, and animated diagram controls passed.
- Exact requirements match the installed environment; dependency compatibility
  and shareable-file credential checks passed.

Synthetic test history was removed from the initial development database after
verification. The default mode is conversation. Translation and reply-language changes require explicit confirmation. No physical microphone
recording or real-world translation benchmark has been claimed.

## Confirmation and translation regression update

The old English Piper voice was replaced with the original Voca Kokoro ONNX
weights, af_heart, speed 1.05; SHA256 hashes match the original files. Both
engines remain local and the new project owns its model copies.

Twenty live translation behavior checks passed: questions vs answers, both
negation directions, names, quantities, euro decimals, dates/times, weekdays,
mixed-language terms, instruction-like source text, and repeated English turns.
These use semantic anchors and are not a human translation-quality evaluation.

Real browser testing passed spoken translation requests, declining a request,
confirming a later request, unchanged settings before confirmation, and cancelling
a German reply-language request. No per-utterance confirmation is required once
translation is explicitly enabled. Mode/context still persist across sessions.

The specific reported regression also passed a live session check: ordinary
English input in conversation mode produced an English answer and Kokoro audio,
without enabling translation. The app was left in conversation mode after QA.
