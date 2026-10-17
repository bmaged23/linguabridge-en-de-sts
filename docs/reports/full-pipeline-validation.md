# Full pipeline test

Local synthetic diagnostic on 2026-10-17. This is not a public benchmark.

[Open the raw trace and audio viewer](../full-qa/index.html).

## Results

- browser-report: 14 recorded turns; 0 unexpected pipeline failures; 0 intentionally injected failures; 0 JavaScript errors.
- recovery-report: 20 recorded turns; 0 unexpected pipeline failures; 1 intentionally injected failures; 0 JavaScript errors.
  - KNOWN FAILURE: short recall question lost my in STT; testing explicit recall separately
  - Saved memory survives reload and is recalled in speech
  - Eight repeated real microphone/cloud/playback turns without reconnecting
  - Suspended audio resumes and completes another real turn
  - Cloud timeout is visible and the next real turn recovers
  - Length-limited replies finish their queued audio and recover
  - Voice interruption cancels playback and the next reply completes
  - Stopped audio clock is detected and a fresh session works
  - Microphone worklet failure is visible and restart works
  - Mute and unmute preserve microphone capture
  - Repeated stop and start do not leave a stale session lock
- baseline-browser (before fixes): 3 recorded turns; 1 unexpected pipeline failures; 0 intentionally injected failures; 0 JavaScript errors.
- mixed-browser: 8 recorded turns; 0 unexpected pipeline failures; 0 intentionally injected failures; 0 JavaScript errors.
- targeted-browser: 3 recorded turns; 0 unexpected pipeline failures; 0 intentionally injected failures; 0 JavaScript errors.
  - Semantic failure: Spoken Yes, confirm was recognized as Yes, firm; confirmation did not activate.
  - PASS: UI confirmation activates translation; conversation restored.

Cloud marker diagnostic: nine of nine synthetic prompts parsed successfully after the repair. See [raw cloud responses](../full-qa/cloud-protocol-diagnostic.json).

Component tests: 47 passed. Threaded tests require unrestricted execution in this environment; the restricted sandbox caused executor timeouts.

## Repairs

- Repeated, mismatched, optional closing language tags and boundary punctuation no longer cancel speech. Opening language switches select the appropriate TTS voice; unmarked text falls back to the requested main language.
- Live audio auditing found the confirmation arrow being pronounced as left right arrow; confirmation wording now uses ordinary spoken words. An isolated-vocabulary prompt example now asks the LLM to mark individual German words as German, verified in the targeted cloud/browser checks.
- Token-limited replies finish queued speech and show a continuation notice.
- Microphone processor errors use an explicit browser event listener; the fault-injection test verified the visible error and restart. Paused audio exposes a Resume audio control. A stopped audio clock or failed microphone processor ends the session with a visible recovery message.
- Optional local pipeline tracing records synthetic test inputs, partial/final recognition, raw cloud output, parsed language segments, native TTS WAVs and playback acknowledgments. It remains disabled in normal sessions.

## Limitations found

- The short synthetic question “What is my German level?” lost “my” in browser-captured STT. A fresh decode of the same audio had the same omission; prefix reuse was not the cause in that example. The model then answered the altered question. An explicit recall question correctly retrieved the saved A2 level. This remains a recognition accuracy limitation.
- In a later run, spoken Yes, confirm became Yes, firm, so translation was not activated. UI confirmation was tested successfully; the earlier full translation sequence also passed. Spoken confirmation remains dependent on ASR accuracy.
- One original German test utterance pronounced English borrow/lend with the German TTS voice. That fixture is unsuitable for judging English word recognition; properly voiced code-switch recordings are tested separately.
- Qwen assigns one language to a whole utterance. A German–English–German example received an English label despite its German opening; the mixed transcript retained both languages. Automatic main reply language is therefore imperfect for mixed inputs.
- The single generated German word obwohl was recognized as Upon by automatic ASR, although its complete German example sentences transcribed correctly. Short isolated-word pronunciation still needs human listening; the automatic check cannot establish whether the voice or recognizer caused that mismatch.
- TTS audio re-recognition scores measure agreement between two models, not human pronunciation quality. See individual phrases and listen to the WAVs in the viewer.
- Virtual microphone input and silent browser output exercise capture/rendering and completion events. Physical microphone, speaker and room echo behavior still require a local device test.
- The user memory database was kept separate from the QA database. No credentials or user history are included in the exported artifacts.


## Complex microphone input recognition

| Input | WER | CER | Audio played |
| --- | ---: | ---: | ---: |
| english_complex | 0.0% | 0.0% | 1/1 |
| german_complex | 10.5% | 4.2% | 1/1 |
| english_numbers | 0.0% | 0.0% | 1/1 |
| german_numbers | 0.0% | 0.0% | 1/1 |
| en_de_en_travel | 0.0% | 0.0% | 4/4 |
| de_en_de_travel | 3.9% | 2.6% | 3/3 |
| en_de_en_learning | 0.0% | 0.0% | 10/10 |
| de_en_de_work | 7.4% | 4.7% | 5/5 |

## Generated audio diagnostic

- en: 71 distinct synthesized phrases checked; mean phrase WER 2.6%; mean phrase CER 1.2%. These are unweighted phrase means.
- de: 37 distinct synthesized phrases checked; mean phrase WER 3.1%; mean phrase CER 2.3%. These are unweighted phrase means.

## Reproducing the checks

Use the installed project environments and model files described in the README. Run the QA server separately from the normal GPU app. QA uses port 8767, a separate memory database, explicit synthetic traces and test-only fault injection. Never expose that test server publicly.

```bash
.venv/bin/python -m scripts.validation.prepare_browser_test_inputs
.venv/bin/python -m scripts.application.start_voice_app -m scripts.validation.start_test_server
# In another terminal, with PLAYWRIGHT_BROWSERS_PATH pointing to your Chromium installation:
.venv/bin/python -m scripts.validation.run_browser_pipeline_tests
.venv/bin/python -m scripts.validation.run_browser_recovery_tests
.venv/bin/python -m scripts.validation.run_browser_pipeline_tests --mixed
.venv/bin/python -m scripts.validation.run_browser_pipeline_tests --targeted
# Stop the QA server before running the extra GPU recognizer:
.venv/bin/python -m scripts.validation.audit_synthesized_audio
.venv/bin/python -m scripts.reports.export_pipeline_report
```
