"""Write an inspectable report from synthetic story runs, including native audio."""
import argparse,difflib,html,json,shutil,wave
import numpy as np
from pathlib import Path
from voice_bridge.settings import ROOT


def main(folder):
 folder=Path(folder);data=json.loads((folder/'results.json').read_text());rows=data['samples']
 from scripts.story_tests.review_story_results import review
 reviewed=review(folder);reviews={r['case']:r for r in reviewed['samples']}
 out=ROOT/'docs/advanced-story'/folder.name;out.mkdir(parents=True,exist_ok=True)
 for directory in ('inputs','samples','pipeline-traces'):
  if (folder/directory).exists():shutil.copytree(folder/directory,out/directory,dirs_exist_ok=True)
 for path in folder.glob('candidate-*'):shutil.copyfile(path,out/path.name)
 for name in ('results.json','resources.jsonl','cases.json','final-screen.png','candidate-results.json','environment.json','review.json','gpu-clocks.jsonl','gpu-clocks-continuous.jsonl','process-map.json','hardware.json','marker-recovery-diagnostic.json'):
  if (folder/name).exists():shutil.copyfile(folder/name,out/name)
 owners={}
 for path in (folder/'pipeline-traces').glob('*/events.jsonl'):
  for line in path.read_text().splitlines():
   event=json.loads(line)
   if event['event']=='tts_audio':owners[event['monotonic_seconds']]=path.parent.name
 candidates={}
 for path in folder.glob('candidate-*.json'):
  candidate=json.loads(path.read_text())
  if isinstance(candidate,dict) and 'samples' in candidate:candidates[path.stem]={r['case']:r for r in candidate['samples']}
 escape=html.escape
 fragments=['<h1>Continuous bilingual story audit</h1><p>'+escape(data['method'])+'</p><p>'+escape(data['resource_scope'])+'</p><p>Every input is one continuous WAV containing a connected story. The reference text is used for scoring only. These synthetic voices do not establish human microphone accuracy.</p><p><a href="results.json">All measurements (JSON)</a> · <a href="resources.jsonl">Raw resource samples</a></p>']
 markdown=['# Continuous bilingual story audit','',data['method'],'',data['resource_scope'],'','[Open raw text, audio and resource measurements](index.html).','','| Story run | WER | CER | Seconds | RAM peak MiB | Whole GPU VRAM peak MiB | Audio finished | GPU state | Review flags |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
 fragments.append('<h2>Raw comparison artifacts</h2><ul>')
 for path in sorted(folder.glob('candidate-*.json')):fragments.append('<li><a href="'+path.name+'">'+escape(path.name)+'</a></li>')
 fragments.append('</ul>')
 for row in rows:
  row['review_flags']=reviews[row['case']['id']]['findings']
  c=row['case'];resources=row['resources'];ram=(resources.get('ram_tree_rss_mb') or {}).get('peak');vram=(resources.get('gpu_vram_mb') or {}).get('peak')
  markdown.append(f'| {c["id"]} | {row["stt_metrics"]["wer"]:.1%} | {row["stt_metrics"]["cer"]:.1%} | {row["timings"]["whole_sample_seconds"]:.2f} | {ram} | {vram} | {row["completed_playback_chunks"]}/{row["audio_chunks"]} | {','.join(reviews[c['id']]['gpu_clock_state']['states']) or 'unobserved'} | {escape("; ".join(row["review_flags"]))} |')
  fragments.append('<details id="'+escape(c['id'],quote=True)+'"><summary>'+escape(c['id'])+f' — WER {row["stt_metrics"]["wer"]:.1%}, CER {row["stt_metrics"]["cer"]:.1%}, {row["timings"]["whole_sample_seconds"]:.1f}s'+('</summary><p class="flag">'+escape('; '.join(row['review_flags']))+'</p>'))
  fragments.append('<h3>Continuous input</h3><audio controls preload="none" src="'+escape(c['input'])+'"></audio><p>'+escape(c['reference'])+'</p><h3>STT final</h3><p>'+escape(row['stt_final'])+'</p>')
  for candidate_name,transcripts in candidates.items():
   match=transcripts.get(c['id'])
   if match:fragments.append('<details><summary>'+escape(candidate_name)+f' — WER {match["metrics"]["wer"]:.1%}, CER {match["metrics"]["cer"]:.1%}, {match["decode_seconds"]:.3f}s decoding</summary><p>'+escape(match['text'])+'</p></details>')
  diff=difflib.HtmlDiff(wrapcolumn=65).make_table(c['reference'].split(),row['stt_final'].split(),fromdesc='Reference words',todesc='Recognized words',context=True,numlines=2)
  fragments.append('<details><summary>Word differences</summary>'+diff+'</details><h3>Raw LLM output, including language markers</h3><pre>'+escape(row['raw_llm'])+'</pre><h3>TTS output in playback order</h3>')
  combined=[]
  for event in row['native_audio']:
   owner=owners.get(event.get('monotonic_seconds'))
   if owner:
    with wave.open(str(folder/'pipeline-traces'/owner/event['audio'])) as handle:
     rate=handle.getframerate();samples=np.frombuffer(handle.readframes(handle.getnframes()),'<i2')
    combined.append(np.interp(np.arange(0,len(samples),rate/24000),np.arange(len(samples)),samples))
   if owner:fragments.append('<p>'+escape(event['language']+': '+event['text'])+'</p><audio controls preload="none" src="pipeline-traces/'+owner+'/'+event['audio']+'"></audio>')
  if combined:
   name=row['case']['id']+'-combined-tts.wav'
   with wave.open(str(out/name),'wb') as handle:handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(24000);handle.writeframes(np.clip(np.concatenate(combined),-32768,32767).astype('<i2').tobytes())
   fragments.append('<p>All generated chunks joined in order; not a recording of speaker playback or its timing gaps.</p><audio controls preload="none" src="'+name+'"></audio>')
  fragments.append('<h3>Timings, RAM, GPU memory and utilization</h3><pre>'+escape(json.dumps({'timings':row['timings'],'resources':resources,'model_ram':reviews[c['id']]['model_ram'],'gpu_clock_state':reviews[c['id']]['gpu_clock_state']},indent=2))+'</pre><details><summary>Every STT preview, raw token, parsed segment, generated chunk and playback acknowledgment</summary><pre>'+escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details></details>')
 style='body{max-width:1100px;margin:30px auto;padding:20px;background:#101824;color:#edf4ff;font:16px system-ui}summary{cursor:pointer}details{border:1px solid #40536b;border-radius:8px;margin:12px 0;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px monospace}audio{width:100%}a{color:#86caff}.flag{color:#ffd294}.diff{font:12px monospace}.diff_add{background:#164f35}.diff_sub{background:#6a2029}.diff_chg{background:#655015}'
 (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Story pipeline audit</title><style>'+style+'</style>'+''.join(fragments)+'<script>function openHash(){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target){target.open=true;target.scrollIntoView();}}addEventListener("hashchange",openHash);openHash();</script></html>')
 markdown+=['','## Review boundaries','','- These are synthetic voices and artificial noise, not a human speech benchmark.', '- The browser uses real-time silent WebAudio rendering and native playback completion callbacks. Physical speaker and microphone behavior is not measured.', '- Resource samples cover all local worker processes. Whole-GPU figures cannot be attributed exclusively to one model, especially during overlapping stages.', '- A phase shorter than the sampling interval may have no resource sample. Sampled peaks are lower bounds.', '- The cloud model runs remotely: cloud RAM/VRAM are unavailable; only local resource use and network response latency are measured.', '- WER/CER normalize case and punctuation, not number spelling or German compounds. Review flags indicate candidates for inspection, not automatically proven semantic errors.','']
 (out/'REPORT.md').write_text('\n'.join(markdown)+'\n');print('Report:',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();main(ROOT/a.folder)
