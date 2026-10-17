# Complex and mixed-language STT check

Tested locally on 2026-10-16 with the configured GPU Whisper large-v3-turbo.
Reproduce: `.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.evaluate_mixed_language_asr`.
Audio and full reference/hypothesis results: `logs/codeswitch-check/` (ignored).

Eight synthetic utterances were produced using Kokoro English and Piper German.
Mixed utterances join three language spans at 16 kHz with 120 ms pauses. Voice
identity changes at language switches: this is a diagnostic, not a human speech
benchmark or an estimate of microphone/accent accuracy.

Compared current app decoding with previous language English and German, native
automatic language detection, and automatic decoding with `multilingual=True`.
All four strategies produced identical text on these eight recordings.

| Case | WER | CER | Observation |
| --- | ---: | ---: | --- |
| English complex appointment | 0% | 0% | Full sentence preserved |
| German complex appointment | 0% | 0% | Full sentence preserved |
| English booking and numbers | 8.0% | 7.3% | “twenty third” rendered as “23rd” |
| German booking and numbers | 13.0% | 13.5% | Numeric formatting plus “Buchung” → “Bohrung”, “geprüft” → “geprügt” |
| English → German → English travel | 66.7% | 67.5% | Both English spans omitted |
| German → English → German travel | 34.6% | 38.6% | English span omitted |
| English → German → English learning | 0% | 0% | All three spans preserved |
| German → English → German work | 44.4% | 46.5% | English span omitted |

Metrics lowercase and remove punctuation; they do not normalize written versus
numeric numbers. WER therefore penalizes equivalent number formatting. No aggregate
accuracy claim is appropriate for this small synthetic sample.

The recognizer cannot currently be described as reliable for arbitrary English/German
switches within one utterance. Removing previous-language bias remains useful but
did not resolve these failures. Before selecting a replacement or segmentation
strategy, compare the same audio and real human mixed-language recordings. Do not
recover missing text by asking the conversation LLM to guess it.

This test changes no app behavior, prompts, memory, or running-server configuration.
