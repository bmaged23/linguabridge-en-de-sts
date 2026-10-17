# Confirmed pipeline findings

50 continuous story runs were completed, including 38 mixed-language runs. All 239 received browser audio chunks completed playback. Fourteen runs exceeded 15% normalized WER. Completing playback is a structural check, not an accuracy pass.

## Recognition changes the spoken language

Several complete mixed recordings were rendered partly in translation rather than verbatim. In `story_en_de_en_pause`, German passages became English while essentially the whole waveform was captured. [Inspect the input, final transcript and raw reply](story-20261017/index.html#story_en_de_en_pause).

On the 32 mixed stories with a single complete capture, live progressive Qwen scored 15.0% WER versus 22.2% for fresh finals. The lower overall fresh-final score partly recovers segments the application discarded and is not evidence that replacing final decoding fixes the app. A fresh final decode helps some samples but still translates others. One quiet recording produced repeated phrases that were not spoken. Cumulative replay and independent-final decoding are measured separately; a language whitelist alone does not address these failures.

The clean monolingual English controls mostly differ in valid spellings such as travelled/traveled and cancelled/canceled. Raw WER deliberately counts those differences, so inspect the word diff before interpreting a small score as a meaning error.

## Turn handling discards recorded speech

The roughly 39-second stories hit the configured 30-second utterance cap. For `long_story_en_de_en`, the application saved captures of 30.016 and 9.024 seconds, but delivered only the ending's final transcript. The earlier response task was canceled when continued speech was treated as a new turn. Natural pauses caused a similar loss in some shorter inputs. [Inspect the long story](story-extended-20261017/index.html#long_story_en_de_en).

The LLM reused prior story information, so its answer sounded more complete than the current transcription. New details in the canceled section were unavailable to it. Preserving or merging unfinished user input is a separate requirement from improving the recognizer.

## Missing voice markers select the wrong voice

Five explicit bilingual-retelling replies omitted every voice marker. The parser's default-language fallback then assigned mixed English/German text to one voice. Examples include English-voice chunks containing “Am Bahnhof sagte Anna” and German-voice chunks containing English clauses. [Inspect an unmarked reply](story-extended-20261017/index.html#bilingual_retelling_en_de_en_quiet).

The earlier fallback prevents a formatting crash, but is insufficient for safe bilingual voice routing. Past assistant messages are also replayed as clean spoken text without markers, which may weaken format compliance; that contribution has not been isolated.

A separate labeling-only LLM diagnostic correctly split all five actual unmarked replies, preserved their non-whitespace text, and took 0.837–1.368 seconds. This is a tested recovery prototype, not an application change. [Raw recovery results](story-extended-20261017/marker-recovery-diagnostic.json).

## Summaries infer actions and time changes

The story says a bus was found and tickets should not yet be bought. Many replies assert that the bus was taken. The source corrects the interview time to ten rather than nine; many replies describe a rescheduling. Some bus-journey claims originate in STT (“fanden” becoming “fuhren”); others are added by the LLM or inherited from memory. Review heuristics flag these distinctions without treating every reasonable inference as a certified factual error. [Inspect the English control](story-20261017/index.html#story_english_clean).

## Resource measurements

Peak sampled server/worker RSS across the 50 live runs: **3037 MiB**. Peak sampled whole-GPU VRAM during those runs: **5768 MiB**. Full per-sample CPU, RAM, VRAM, GPU utilization, STT queue/decode, cloud first-text, TTS inference and overall completion measurements are in the viewers and JSON files.

The GPU initially stayed at P8/300 MHz under load, then reached P0/high clocks. Timing results must be separated by power state. Whole-GPU measurements include other GPU consumers; RSS sums can count shared pages twice. Cloud LLM RAM/VRAM are unavailable. Post-test audio audits have separate resource labels and are not part of production response latency.

## Scope

These tests use synthetic voices in continuous WAVs and real-time silent WebAudio rendering. They do not verify a physical microphone, speakers, a human accent, or general ASR benchmark performance. Each generated WAV is saved, structurally inspected and re-recognized; model agreement does not replace listening. No user history or credentials are exported. Production recognition and turn-taking behavior have not been changed by this audit.
