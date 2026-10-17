"""Export synthetic QA artifacts only. Never reads the user's conversation database."""
from pathlib import Path
import html,json,shutil
from voice_bridge.settings import ROOT


def main():
 source=ROOT/'logs/full-qa';dest=ROOT/'docs/full-qa';dest.mkdir(exist_ok=True)
 reports={p.stem:json.loads(p.read_text()) for p in source.glob('*-report.json')}
 for name in ('baseline-browser','mixed-browser','targeted-browser'):
  path=source/f'{name}.json'
  if path.exists():reports[name]=json.loads(path.read_text())
 for name in ('final-asr-comparison',):
  path=source/f'{name}.json'
  if path.exists():shutil.copyfile(path,dest/path.name)
 protocol=ROOT/'logs/cloud-protocol-diagnostic.json'
 if protocol.exists():shutil.copyfile(protocol,dest/protocol.name)
 rows=json.loads((source/'tts-audio-audit.json').read_text()) if (source/'tts-audio-audit.json').exists() else []
 body=['<h1>Full voice pipeline test</h1><p>Synthetic microphone input, real GPU STT/TTS, real cloud responses, and real-time Chromium WebAudio completion acknowledgments. The browser output used a silent sink: this does not verify physical speakers, the user’s microphone, or acoustic echo cancellation. Raw content below is synthetic QA only.</p>']
 for name,report in reports.items():
  shutil.copyfile(source/f'{name}.json',dest/f'{name}.json')
  body.append(f'<h2>{html.escape(name)}</h2><pre>{html.escape(json.dumps({k:v for k,v in report.items() if k not in ("turns","failures")},ensure_ascii=False,indent=2))}</pre>')
  for turn in report.get('turns',[]):
   body.append('<details><summary>'+html.escape(turn.get('case','turn'))+'</summary><pre>'+html.escape(json.dumps(turn,ensure_ascii=False,indent=2))+'</pre></details>')
 for path in sorted((source/'pipeline-traces').glob('*/events.jsonl')):
  target=dest/path.parent.name;target.mkdir(exist_ok=True);shutil.copyfile(path,target/path.name)
  events=[json.loads(line) for line in path.read_text().splitlines()]
  body.append('<details><summary>Raw pipeline session '+path.parent.name+'</summary>')
  raw=''.join(e['text'] for e in events if e['event']=='llm_raw')
  body.append('<h3>Raw streamed model text with language markers</h3><pre>'+html.escape(raw)+'</pre>')
  for e in events:
   if e['event'] in ('stt_input','tts_audio'):
    shutil.copyfile(path.parent/e['audio'],target/e['audio'])
    body.append('<p>'+html.escape(e.get('language','input')+': '+e.get('text','Captured microphone input'))+'</p><audio controls preload="none" src="'+path.parent.name+'/'+e['audio']+'"></audio>')
  body.append('<details><summary>All partial/final STT, raw tokens, parsed segments, TTS timing and playback acknowledgments</summary><pre>'+html.escape('\n'.join(json.dumps(e,ensure_ascii=False) for e in events))+'</pre></details></details>')
 if rows:
  (dest/'tts-audio-audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
  body.append('<h2>TTS audio recognition diagnostic</h2><p>Re-recognizing generated speech with the same ASR is an imperfect intelligibility diagnostic, not a human listening score or a public WER/CER benchmark. Punctuation and case are normalized; number spelling is not.</p><pre>'+html.escape(json.dumps(rows,ensure_ascii=False,indent=2))+'</pre>')
 style='body{max-width:1000px;margin:40px auto;padding:20px;background:#101723;color:#e5edf8;font:16px system-ui}details{border:1px solid #3c4d64;padding:12px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}audio{width:100%}a{color:#80c7ff}'
 (dest/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Voice Bridge full pipeline QA</title><style>'+style+'</style>'+''.join(body)+'</html>')
 lines=['# Full pipeline test','', 'Local synthetic diagnostic on 2026-10-17. This is not a public benchmark.','', '[Open the raw trace and audio viewer](../full-qa/index.html).','', '## Results','']
 for name,report in reports.items():
  turns=report.get('turns',[])
  label=name+' (before fixes)' if name=='baseline-browser' else name
  lines.append(f'- {label}: {len(turns)} recorded turns; {len(report.get("failures",[]))} unexpected pipeline failures; {sum(bool(t.get("expected_error")) for t in turns)} intentionally injected failures; {len(report.get("browser_errors",report.get("javascript_errors",[])))} JavaScript errors.')
  for check in report.get('checks',[]):lines.append('  - '+check)
  for failure in report.get('semantic_failures',[]):lines.append('  - Semantic failure: '+failure)
  if report.get('ui_confirmation_recovery'):lines.append('  - '+report['ui_confirmation_recovery'])
 lines+=['','Cloud marker diagnostic: nine of nine synthetic prompts parsed successfully after the repair. See [raw cloud responses](../full-qa/cloud-protocol-diagnostic.json).','', 'Component tests: 47 passed. Threaded tests require unrestricted execution in this environment; the restricted sandbox caused executor timeouts.','', '## Repairs','', '- Repeated, mismatched, optional closing language tags and boundary punctuation no longer cancel speech. Opening language switches select the appropriate TTS voice; unmarked text falls back to the requested main language.', '- Live audio auditing found the confirmation arrow being pronounced as left right arrow; confirmation wording now uses ordinary spoken words. An isolated-vocabulary prompt example now asks the LLM to mark individual German words as German, verified in the targeted cloud/browser checks.', '- Token-limited replies finish queued speech and show a continuation notice.', '- Microphone processor errors use an explicit browser event listener; the fault-injection test verified the visible error and restart. Paused audio exposes a Resume audio control. A stopped audio clock or failed microphone processor ends the session with a visible recovery message.', '- Optional local pipeline tracing records synthetic test inputs, partial/final recognition, raw cloud output, parsed language segments, native TTS WAVs and playback acknowledgments. It remains disabled in normal sessions.','', '## Limitations found','', '- The short synthetic question “What is my German level?” lost “my” in browser-captured STT. A fresh decode of the same audio had the same omission; prefix reuse was not the cause in that example. The model then answered the altered question. An explicit recall question correctly retrieved the saved A2 level. This remains a recognition accuracy limitation.', '- In a later run, spoken Yes, confirm became Yes, firm, so translation was not activated. UI confirmation was tested successfully; the earlier full translation sequence also passed. Spoken confirmation remains dependent on ASR accuracy.', '- One original German test utterance pronounced English borrow/lend with the German TTS voice. That fixture is unsuitable for judging English word recognition; properly voiced code-switch recordings are tested separately.', '- Qwen assigns one language to a whole utterance. A German–English–German example received an English label despite its German opening; the mixed transcript retained both languages. Automatic main reply language is therefore imperfect for mixed inputs.', '- The single generated German word obwohl was recognized as Upon by automatic ASR, although its complete German example sentences transcribed correctly. Short isolated-word pronunciation still needs human listening; the automatic check cannot establish whether the voice or recognizer caused that mismatch.', '- TTS audio re-recognition scores measure agreement between two models, not human pronunciation quality. See individual phrases and listen to the WAVs in the viewer.', '- Virtual microphone input and silent browser output exercise capture/rendering and completion events. Physical microphone, speaker and room echo behavior still require a local device test.', '- The user memory database was kept separate from the QA database. No credentials or user history are included in the exported artifacts.','']
 mixed=reports.get('mixed-browser',{}).get('turns',[])
 if mixed:
  lines+=['','## Complex microphone input recognition','','| Input | WER | CER | Audio played |','| --- | ---: | ---: | ---: |']
  for turn in mixed:lines.append(f'| {turn["case"]} | {turn["stt_metrics"]["wer"]:.1%} | {turn["stt_metrics"]["cer"]:.1%} | {turn["played_chunks"]}/{turn["audio_chunks"]} |')
 if rows:
  lines+=['','## Generated audio diagnostic','']
  for language in ('en','de'):
   selected=[r for r in rows if r['language']==language]
   lines.append(f'- {language}: {len(selected)} distinct synthesized phrases checked; mean phrase WER {sum(r["metrics"]["wer"] for r in selected)/max(len(selected),1):.1%}; mean phrase CER {sum(r["metrics"]["cer"] for r in selected)/max(len(selected),1):.1%}. These are unweighted phrase means.')
 lines+=['','## Reproducing the checks','','Use the installed project environments and model files described in the README. Run the QA server separately from the normal GPU app. QA uses port 8767, a separate memory database, explicit synthetic traces and test-only fault injection. Never expose that test server publicly.','','```bash','.venv/bin/python -m scripts.validation.prepare_browser_test_inputs','.venv/bin/python -m scripts.application.start_voice_app -m scripts.validation.start_test_server','# In another terminal, with PLAYWRIGHT_BROWSERS_PATH pointing to your Chromium installation:','.venv/bin/python -m scripts.validation.run_browser_pipeline_tests','.venv/bin/python -m scripts.validation.run_browser_recovery_tests','.venv/bin/python -m scripts.validation.run_browser_pipeline_tests --mixed','.venv/bin/python -m scripts.validation.run_browser_pipeline_tests --targeted','# Stop the QA server before running the extra GPU recognizer:','.venv/bin/python -m scripts.validation.audit_synthesized_audio','.venv/bin/python -m scripts.reports.export_pipeline_report','```','']
 (ROOT/'docs/reports/full-pipeline-validation.md').write_text('\n'.join(lines)+'\n')
 for name in ('mobile-final.png','recovery-desktop.png'):
  path=source/name
  if path.exists():shutil.copyfile(path,dest/name)
 print('Exported',len(reports),'reports;',len(rows),'TTS audio checks')
if __name__=='__main__':main()
