# Completed-segment reuse and GPU batching

Qwen3-ASR still decodes both languages locally. While speech arrives, the worker saves exact completed pause segments in a per-utterance bounded cache. It always decodes the unfinished tail again. Audio hashes and decoding settings must match before reuse; the cache clears at utterance end. Remaining segments run in GPU batches of two, with serial retry on CUDA out-of-memory. The original monolingual whole-input fallback and meaning/plausibility guards remain. No model, dependencies, voices or cloud request count were added.

## Accuracy gate

All 52 archived final-decode transcripts were unchanged with batch size two. Strict per-recording WER/CER, routing and source-fact non-regression gates passed. The 12 cumulative replays also preserved normalized text, routing and facts for all three profiles. These reuse existing evaluation recordings; they are not a new independent human mixed-language benchmark. Existing noise-related recognition errors remain.

## Offline replay measurements

| Profile | Median final decode | Median summed decode compute | Final model calls across 12 recordings | Cache hits | CUDA OOM fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 5.649 s | 15.571 s | 60 | 0 | 0 |
| Cache, serial | 2.326 s | 16.766 s | 20 | 40 | 0 |
| Cache, batch two | 2.300 s | 12.975 s | 17 | 40 | 0 |

Eight recordings are mixed-language stories/captures and four are human monolingual controls. For the eight mixed recordings, median final decoding fell from 6.561 to 1.511 seconds. Prefixes arrive every two seconds in the replay schedule, but the script does not sleep: final timing excludes real arrival, endpointing, queues, cloud and TTS. Summed compute is not elapsed conversation time. Caching alone moves computation earlier and can increase total work; batching reduced it in this run.

See [replay assessment](replay-assessment.json) and [batch quality comparison](batch-quality-comparison.json). Real-time browser measurements are reported separately below after the complete run.

## Matched real-time story

The same 15.10-second prerecorded story and waveform hash were supplied through the browser audio path. First output audio after recording ended fell from **7.771 to 2.690 seconds** (about 65% less delay) in these two runs. The candidate final ASR took 1.245 seconds, reused five segments and made one model call. Both runs passed the source-grounded cancellation and ticket-negation checks, with unchanged 4.65% WER / 1.24% CER.

For this story, candidate peak summed process RSS was 3,880.51 MiB and whole-device VRAM 4,780 MiB. Mean CPU was 70.65% of one core / 5.89% of machine capacity. RSS may count shared pages; GPU numbers include other device allocations. Cloud LLM hardware is excluded. This is one paired case, not a repeated-trial latency guarantee. Browser output used a silent sink, so physical microphone/speaker echo and personal accent testing remain pending. See [paired measurements](browser-paired-story.json).

## Full browser regression

All **17/17** scripted turns passed, including bilingual translation, mixed speech, teaching, saved memory, reload, silence and interruption. There were no browser JavaScript errors. Median first audio after recording ended was **2.624 seconds** across cases emitting audio. See [per-case metrics](browser-candidate-summary.json). The cloud model is variable and the suite is small; this does not establish a universal speedup. The application test suite also passed **89/89** tests.

## Independent generated-audio audit

Installed Whisper large-v3 scored **37/37** native output chunks, with **0 execution failures**. Reviewing every nonzero-WER transcript showed number/time spelling differences (for example ‘ten’ versus ‘10’ and ‘nine forty-five’ versus ‘9.45’). Literal WER/CER retain these differences; no scores were silently normalized away. This is text/audio agreement with known-language decoding, not a human listening or accent-quality score. See [audio audit metrics](audio-audit-summary.json).
