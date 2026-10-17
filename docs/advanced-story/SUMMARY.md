# Advanced continuous-story audit

Every input is a single connected story WAV. Conditions include English, German, repeated language switches, noise, low volume, pauses, repeated inputs, bilingual retelling requests, and longer stories. These are synthetic diagnostic runs of one related narrative, not independent human speakers or a public benchmark.

50 pipeline runs; 38 mixed-language runs; 239 generated browser audio chunks; 239 completed playback acknowledgments.

## Saved outputs

- [story-20261017: every sample, raw STT/LLM tokens, native and combined TTS audio, resource measurements](story-20261017/index.html)
- [story-extended-20261017: every sample, raw STT/LLM tokens, native and combined TTS audio, resource measurements](story-extended-20261017/index.html)

## Recognition comparison

WER/CER below normalize case and punctuation but retain spelling and number-format differences. Scores are weighted by reference length. Candidate models receive the same PCM16 browser-captured WAVs; the live application originally receives float32 samples. Candidates decode every captured segment, including segments the application canceled. The live application therefore measures both recognition and turn-handling losses.

| Configuration | Mixed N | Mixed WER | Mixed CER | All N | All WER | All CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Live Qwen progressive pipeline | 38 | 26.5% | 20.8% | 50 | 21.7% | 16.5% |
| large-v3 (30s multilingual) | 38 | 53.1% | 51.0% | 50 | 43.4% | 41.2% |
| large-v3 (5s multilingual) | 28 | 48.2% | 43.9% | 40 | 39.1% | 34.7% |
| qwen-fresh | 38 | 19.5% | 13.0% | 50 | 16.5% | 10.6% |
| qwen-replay | 24 | 21.5% | 13.4% | 26 | 20.3% | 12.5% |
| turbo (30s multilingual) | 38 | 53.5% | 51.2% | 50 | 43.9% | 41.5% |
| turbo (5s multilingual) | 28 | 50.1% | 45.2% | 40 | 40.1% | 35.6% |

### Mixed stories with one completed captured segment

This subset removes cases with canceled or split final transcriptions, making decoder comparisons easier to interpret.

| Configuration | N | WER | CER |
| --- | ---: | ---: | ---: |
| Live Qwen progressive pipeline | 32 | 15.0% | 8.1% |
| large-v3 (30s multilingual) | 32 | 53.9% | 52.0% |
| large-v3 (5s multilingual) | 25 | 48.7% | 44.5% |
| qwen-fresh | 32 | 22.2% | 14.9% |
| qwen-replay | 18 | 23.4% | 14.7% |
| turbo (30s multilingual) | 32 | 54.5% | 52.4% |
| turbo (5s multilingual) | 25 | 50.5% | 45.9% |

`qwen-replay` is deliberately restricted to cases failing in the live or fresh-final run, plus clean controls, so its aggregate is not directly comparable to the full-suite aggregates. It replays cumulative previews and prefix reuse on the same frozen WAVs. Fresh Qwen is a full independent final decode. Whisper candidates use automatic per-window language detection (`multilingual=True`); 30-second standard and 5-second experimental windows are identified separately and a transcription task. Their reported language label is the initial window, not a complete language segmentation.

## Interpretation

On the identical complete-capture mixed subset, the live progressive path has lower error than independent fresh finals. The better overall fresh-final score partly comes from recovering segments the application canceled. Neither fresh-final decoding nor the tested Whisper configurations is a demonstrated drop-in fix. Preserve user speech and enforce voice metadata before attributing all failures to model choice.

## Timing and resource interpretation

The first part of the base run encountered a GPU held in P8 at 300 MHz under load. It later reached P0 at roughly 1,650–1,860 MHz. Do not pool those final-STT timings as normal GPU performance. Per-sample stage clocks and raw power/clock samples are saved where available. GPU clocks normally also drop during idle/playback; the reviewer correlates clock state specifically with final STT.

| Local candidate in isolation | Samples | Median decoding seconds | Peak sampled RAM MiB | Peak sampled whole-GPU VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| large-v3 (30s multilingual) | 50 | 2.903 | 1867.8 | 2321.0 |
| large-v3 (5s multilingual) | 40 | 5.180 | 1879.05 | 2225.0 |
| qwen-fresh | 50 | 5.143 | 1806.38 | 3881.0 |
| qwen-replay | 26 | 10.455 | 1809.13 | 3903.0 |
| turbo (30s multilingual) | 50 | 1.430 | 1168.45 | 1233.0 |
| turbo (5s multilingual) | 40 | 3.196 | 1167.4 | 1233.0 |

Candidate decoding excludes microphone arrival, VAD, cloud generation and playback. Replay decoding includes all its preview computation. Live end-to-end timings include the recorded input duration and playback. Local pipeline RAM covers the server and speech workers; system RAM is the WSL guest rather than Windows host; cloud LLM RAM/VRAM are unavailable. GPU memory/utilization cover the whole device. Sampling can miss short peaks. RSS sums may count shared pages more than once. See each sample for CPU percentages, per-model RAM, GPU utilization, individual STT/LLM/TTS timings, and full completion time.

## Findings requiring attention

- 40 runs: LLM/memory infers a bus journey that the current transcript only describes as finding a bus.
- 38 runs: LLM/memory adds rescheduling; the current story only corrects ten rather than nine.
- 24 runs: LLM omitted all voice markers; fallback routed the complete reply through one default voice.
- 14 runs: WER exceeds 15 percent.
- 11 runs: de vocabulary largely missing or translated; inspect original and recognized text.
- 5 runs: Final STT timing measured with GPU in low-power P8 state.
- 5 runs: 1 detected speech segments were canceled before their final transcription was delivered.
- 4 runs: en vocabulary largely missing or translated; inspect original and recognized text.
- 3 runs: Voice-span mismatch candidate: en chunk contains firma.
- 2 runs: TTS re-recognition disagrees with intended phrase: hattet ihr mehr Zeit..
- 2 runs: STT changed finding a bus into a bus journey; the summary propagates that change.
- 2 runs: Voice-span mismatch candidate: en chunk contains gestern, seid, vorstellungsgespräch.
- 2 runs: Voice-span mismatch candidate: en chunk contains bahnhof.
- 2 runs: Voice-span mismatch candidate: en chunk contains angekommen, fünfundvierzig, seid.
- 2 runs: TTS re-recognition disagrees with intended phrase: Gestern seid ihr für ein Vorstellungsgespräch nach Berlin gereist..
- 2 runs: TTS re-recognition disagrees with intended phrase: um die Zeit von neun auf zehn Uhr zu ändern,.
- 2 runs: Voice-span mismatch candidate: de chunk contains yesterday, you.
- 2 runs: Voice-span mismatch candidate: de chunk contains the.
- 2 runs: Voice-span mismatch candidate: de chunk contains the, you.
- 2 runs: Voice-span mismatch candidate: de chunk contains you.
- 2 runs: Voice-span mismatch candidate: de chunk contains because, called, the.
- 2 runs: Voice-span mismatch candidate: de chunk contains arrived, you.
- 2 runs: TTS re-recognition disagrees with intended phrase: Yesterday, you and Anna traveled to Berlin for an interview..
- 2 runs: TTS re-recognition disagrees with intended phrase: sie solle noch keine Tickets kaufen. At the station,.
- 2 runs: TTS re-recognition disagrees with intended phrase: Anna said the train was cancelled, so you found a bus,.
- 2 runs: TTS re-recognition disagrees with intended phrase: but you told her not to buy tickets yet..
- 2 runs: TTS re-recognition disagrees with intended phrase: The company called because the interview was at ten instead of nine,.
- 2 runs: TTS re-recognition disagrees with intended phrase: and you arrived at nine forty-five..
- 1 runs: 2 detected speech segments were canceled before their final transcription was delivered.
- 1 runs: TTS re-recognition disagrees with intended phrase: so you looked for a bus. Am Bahnhof sagte Anna,.
- 1 runs: TTS re-recognition disagrees with intended phrase: dass der Zug ausfällt, also habt ihr einen Bus gesucht..
- 1 runs: TTS re-recognition disagrees with intended phrase: und ihr seid um neun Uhr fünfundvierzig angekommen..
- 1 runs: Voice-span mismatch candidate: en chunk contains gesagt.
- 1 runs: Voice-span mismatch candidate: en chunk contains keine, soll.
- 1 runs: TTS re-recognition disagrees with intended phrase: You told Anna not to buy tickets. Du hast Anna gesagt,.
- 1 runs: TTS re-recognition disagrees with intended phrase: dass sie keine Tickets kaufen soll. Then the company called to change the interview from nine to ten..
- 1 runs: TTS re-recognition disagrees with intended phrase: Dann hat die Firma angerufen, um das Gespräch von neun auf zehn Uhr zu verschieben..
- 1 runs: TTS re-recognition disagrees with intended phrase: You arrived at nine forty-five. Ihr seid um neun Uhr fünfundvierzig angekommen..
- 1 runs: Voice-span mismatch candidate: en chunk contains keine.
- 1 runs: Voice-span mismatch candidate: en chunk contains fünfundvierzig.
- 1 runs: TTS re-recognition disagrees with intended phrase: Am Bahnhof sagte Anna, dass der Zug ausfällt,.
- 1 runs: TTS re-recognition disagrees with intended phrase: also habt ihr einen Bus gefunden, aber du sagtest,.
- 1 runs: TTS re-recognition disagrees with intended phrase: sie solle noch keine Tickets kaufen. The company called to change the time from nine to ten,.
- 1 runs: TTS re-recognition disagrees with intended phrase: and you arrived at nine forty-five. Die Firma rief an,.
- 1 runs: TTS re-recognition disagrees with intended phrase: und ihr kamt um neun Uhr fünfundvierzig an..
- 1 runs: TTS re-recognition disagrees with intended phrase: Schließlich hat euch der Manager begrüßt und euch etwas Wasser angeboten..

### Voice-marker recovery diagnostic

A separate labeling-only LLM pass was tested on 5 actual unmarked bilingual replies. 5/5 retained every non-whitespace character and returned valid English/German spans. Observed request time: 0.837–1.368 seconds. The inspected language boundaries matched the bilingual sentences. This prototype was not deployed and does not correct factual content.

[Raw labeling outputs](story-extended-20261017/marker-recovery-diagnostic.json).

These findings separate speech loss/translation, canceled user segments, summary inferences and audio disagreement. Semantic inference flags are transparent review heuristics, not a certified semantic score. The same story is repeated while application memory persists, which can conceal missing current STT input; full LLM prompt history is retained in the traces.

TTS WAV integrity is checked and actual WAV bytes are re-recognized; identical bytes may share an audit result. Automatic ASR agreement cannot replace human listening. Browser playback uses real-time WebAudio with a silent output sink, so physical speaker/microphone behavior is not verified.

## Reproduction

See [test instructions](../guides/advanced-story-testing.md). The normal conversation database remains separate from the test databases. No credentials or user conversation history are exported.
