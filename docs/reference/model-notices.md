# Model notices and privacy

- [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen implementation](https://github.com/QwenLM/Qwen3-ASR), Apache-2.0.
- Optional legacy baseline: [Whisper Turbo original model](https://huggingface.co/openai/whisper-large-v3-turbo) and [CTranslate2 conversion](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo), MIT.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), MIT.
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT.
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) and [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) provide English speech; retain their respective model/runtime notices.
- [Piper runtime](https://github.com/OHF-Voice/piper1-gpl), GPL-3.0; retain its license obligations when distributing software.
- Voice licenses are separate from the engine: the [Thorsten card](https://huggingface.co/rhasspy/piper-voices/blob/main/de/de_DE/thorsten/high/MODEL_CARD) names a CC0 dataset. Its model card is downloaded beside the German voice. English uses Kokoro’s model and voice embeddings. Do not assume the repository-wide license covers every voice's training material.
- [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B-it); hosted by Ollama Cloud. Availability, quota, pricing and latency depend on your account/provider.

Live audio stays local. Test scripts can save audio under ignored `logs/`; reviewed synthetic reports/audio under `docs/` are shareable Git content. Finalized text, selected recent history, and relevant notes/summaries are sent to Ollama Cloud. Raw conversation text persists locally by default for memory; it is not written to ordinary server logs. `.env`, `description/`, models, data, caches and logs are excluded from Git.
