# TTS accuracy and computation comparison

Measured 2026-10-17 on GTX 1660 Ti 6 GiB. Same text and ASR scoring across variants. English/German each have four sentences; mixed has four (two in each switching direction).

| Scorer | Language | Voice / strategy | WER | CER | Generation mean | First segment mean | Audio mean | RTF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen | de | piper | 3.77% | 0.26% | 0.257 s | 0.253 s | 5.134 s | 0.050 |
| qwen | de | qwen_float32 | 0.00% | 0.00% | 19.866 s | 19.866 s | 5.300 s | 3.748 |
| qwen | en | kokoro | 0.00% | 0.00% | 0.540 s | 0.540 s | 4.736 s | 0.114 |
| qwen | en | qwen_float32 | 0.00% | 0.00% | 18.377 s | 18.376 s | 4.480 s | 4.102 |
| qwen | mixed | current_single_voice | 34.33% | 13.77% | 0.464 s | 0.462 s | 5.681 s | 0.082 |
| qwen | mixed | hybrid_gold_spans | 8.96% | 6.49% | 0.473 s | 0.128 s | 5.255 s | 0.090 |
| qwen | mixed | qwen_float32_auto | 34.33% | 12.73% | 23.212 s | 23.212 s | 5.400 s | 4.298 |
| qwen | mixed | qwen_float32_gold_spans | 17.91% | 8.05% | 24.553 s | 5.872 s | 5.360 s | 4.581 |
| whisper | de | piper | 0.00% | 0.00% | 0.257 s | 0.253 s | 5.134 s | 0.050 |
| whisper | de | qwen_float32 | 0.00% | 0.00% | 19.866 s | 19.866 s | 5.300 s | 3.748 |
| whisper | en | kokoro | 0.00% | 0.00% | 0.540 s | 0.540 s | 4.736 s | 0.114 |
| whisper | en | qwen_float32 | 0.00% | 0.00% | 18.377 s | 18.376 s | 4.480 s | 4.102 |
| whisper | mixed | current_single_voice | 25.37% | 8.57% | 0.464 s | 0.462 s | 5.681 s | 0.082 |
| whisper | mixed | hybrid_gold_spans | 47.76% | 47.79% | 0.473 s | 0.128 s | 5.255 s | 0.090 |
| whisper | mixed | qwen_float32_auto | 2.99% | 2.60% | 23.212 s | 23.212 s | 5.400 s | 4.298 |
| whisper | mixed | qwen_float32_gold_spans | 1.49% | 0.26% | 24.553 s | 5.872 s | 5.360 s | 4.581 |

## Computation

CPU percentages divide by all 12 logical CPUs. RAM uses proportional set size (PSS). GPU metrics are whole-device. Short inference runs yield few samples.

| Language / strategy | Peak VRAM | Peak RAM PSS | Mean CPU | Peak CPU | Mean GPU | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- |
| de / piper | 1276 MiB | 1025 MiB | 7.8% | 8.1% | 44.4% | 47.0% |
| de / qwen_float32 | 5026 MiB | 1724 MiB | 8.6% | 9.4% | 33.1% | 63.0% |
| en / kokoro | 982 MiB | 1008 MiB | 8.1% | 8.7% | 34.2% | 42.0% |
| en / qwen_float32 | 4908 MiB | 1719 MiB | 8.6% | 9.9% | 31.2% | 63.0% |
| mixed / current_single_voice | 1314 MiB | 1030 MiB | 8.3% | 9.1% | 49.8% | 56.0% |
| mixed / hybrid_gold_spans | 1314 MiB | 1030 MiB | 8.5% | 9.2% | 47.9% | 56.0% |
| mixed / qwen_float32_auto | 5026 MiB | 1734 MiB | 8.7% | 9.6% | 31.9% | 62.0% |
| mixed / qwen_float32_gold_spans | 5026 MiB | 1735 MiB | 8.7% | 10.3% | 27.0% | 61.0% |

## One voice across languages

Internal diagnostic: cosine similarity to the synthetic Kokoro reference using Qwen TTS own speaker encoder. This is not an independent speaker-verification or listening score.

| Language | Strategy | Mean cosine to reference |
| --- | --- | --- |
| de | piper | 0.918 |
| de | qwen_float32 | 0.988 |
| en | kokoro | 0.996 |
| en | qwen_float32 | 0.991 |
| mixed | current_single_voice | 0.959 |
| mixed | hybrid_gold_spans | 0.961 |
| mixed | qwen_float32_auto | 0.989 |
| mixed | qwen_float32_gold_spans | 0.990 |

## GPU compatibility

FP16 failed during warmup with invalid sampling probabilities. FP32 generated all test outputs. With FP32 Qwen TTS resident, loading the deployed FP32 Qwen ASR **passed**.

[Full co-residency measurement](../tts-comparison/co-residency.json).

Running actual speech inference with both models resident **passed**.

Both models loaded with 5924 / 6144 MiB reported by nvidia-smi. The real ASR check took 4.118 s, and the short TTS reply took 7.536 s to generate 1.520 s of audio. This is a sequential resident-model check, not a concurrent stress test; near-capacity device usage is not proof that every request fits without paging. The measured standalone TTS speed is already slower than playback, so the present Kokoro/Piper runtime remains preferable for live response speed.

The released model is named 0.6B. Counting the loaded PyTorch modules gives 914,643,008 parameters in talker plus speaker encoder, and 153,714,657 in the separate audio tokenizer. The release size label does not describe the entire inference pipeline.

## Mixed-language scoring caveat

Whisper omitted complete English sections from multiple correctly split hybrid samples. For example, it transcribed only the German middle of an English–German–English sentence. Compare both scorer tables and listen to the audio; do not interpret that mixed-language score as proof that the TTS omitted those words.

## Interpretation and limits

- 12 authored sentences, 4 per language group; one seeded run per case; synthetic content-consistency smoke test, not a general benchmark.
- ASR-mediated WER/CER confounds recognition and synthesis and can miss mispronunciation. Both recognizers score all variants. Whisper is independent of the Qwen family; Qwen ASR provides a second view.
- Gold spans are manually supplied evaluation boundaries, not a tested automatic LLM segmentation implementation. Whisper dropped whole English sections from several correct hybrid outputs; mixed-language WER is especially ASR-dependent and is not a reliable standalone pronunciation ranking.
- The reference voice is synthetic Kokoro af_heart; no claim of cloning a real person or measured speaker similarity.
- All timing measurements exclude loading and one warmup. Full waveform calls: first audio means first completed segment, not audio-token streaming.
- Models run sequentially without STT/LLM/browser playback. CPU/RAM refer to the generation process; whole-device GPU sampling can miss brief peaks.
- Both Kokoro and Piper share one loaded process. Their resources cannot be attributed as independent resident footprints.
- RTF below 1 means faster than playback. Naturalness, accent authenticity, and speaker consistency require listening.

[Listen to the comparison](../tts-comparison/index.html) · [Raw scores, transcripts and resources](../tts-comparison/report.json) · [Fixed test sentences](../../scripts/evaluation/speech_synthesis_test_cases.json)

Qwen source and API: [official repository](https://github.com/QwenLM/Qwen3-TTS). Qwen TTS checkpoint revision `5d83992436eae1d760afd27aff78a71d676296fc`. See `tts-comparison/qwen-settings.json` and `compatibility.json` for local precision and compatibility details.
