"""Inspect every saved stage, retaining heuristic findings for manual review."""
import argparse,json,re,wave,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from voice_bridge.settings import ROOT
from scripts.evaluation.evaluate_mixed_language_asr import normalized


def review(folder):
 folder=Path(folder);data=json.loads((folder/'results.json').read_text());owners={}
 for path in (folder/'pipeline-traces').glob('*/events.jsonl'):
  for line in path.read_text().splitlines():
   event=json.loads(line)
   if event['event']=='tts_audio':owners[event['monotonic_seconds']]=path.parent/event['audio']
 clocks=[json.loads(l) for l in (folder/'gpu-clocks-continuous.jsonl').read_text().splitlines()] if (folder/'gpu-clocks-continuous.jsonl').exists() else []
 process_map={p['pid']:p['role'] for p in json.loads((folder/'process-map.json').read_text())} if (folder/'process-map.json').exists() else {}
 resources=[json.loads(l) for l in (folder/'resources.jsonl').read_text().splitlines()]
 rows=[]
 for row in data['samples']:
  case=row['case'];findings=list(row['review_flags']);inputs=[e for e in row['pipeline_trace'] if e['event']=='stt_input'];finals=[e for e in row['pipeline_trace'] if e['event']=='stt_final'];retention={}
  with wave.open(str(folder/case['input'])) as source:
   input_pcm=source.readframes(source.getnframes())
  if hashlib.sha256(input_pcm).hexdigest()!=case['sha256']:findings.append('Input WAV hash does not match the frozen manifest')
  if row['timings']['whole_sample_seconds']<case['seconds']+1.3:findings.append('Test may have accepted an early reply before final microphone silence; rerun with the post-input settling guard')
  if len(inputs)>len(finals):findings.append(f'{len(inputs)-len(finals)} detected speech segments were canceled before their final transcription was delivered')
  if case['kind']=='mixed':
   words={lang:set(normalized(' '.join(text for language,text in case['spans'] if language==lang))) for lang in ('en','de')};recognized=set(normalized(row['stt_final']))
   for lang in ('en','de'):
    unique=words[lang]-words['de' if lang=='en' else 'en'];retention[lang]=round(len(unique&recognized)/max(len(unique),1),3)
    if retention[lang]<.35:findings.append(f'{lang} vocabulary largely missing or translated; inspect original and recognized text')
  if any(e['event']=='response_limit' for e in row['pipeline_trace']):findings.append('Cloud reply reached its token limit; generated speech may be an incomplete answer')
  markers=re.findall(r'\[(en|de)\]',row['raw_llm'])
  if row['raw_llm'].strip() and not markers:findings.append('LLM omitted all voice markers; fallback routed the complete reply through one default voice')
  german_words={'gestern','bahnhof','seid','vorstellungsgespräch','gesagt','keine','soll','firma','angekommen','fünfundvierzig','schließlich','willkommen'}
  english_words={'yesterday','you','the','called','arrived','welcomed','please','before','because','finally'}
  for event in row['native_audio']:
   words=set(normalized(event['text']));wrong=(german_words if event['language']=='en' else english_words)&words
   if wrong:findings.append('Voice-span mismatch candidate: '+event['language']+' chunk contains '+', '.join(sorted(wrong)))
  answer=row['reply'].casefold()
  if re.search(r'took a bus|taken a bus|bus genommen|nahmen.*bus|mit dem bus gefahren',answer):
   if re.search(r'took a bus|fuhren.{0,15}bus|nahmen.{0,15}bus|mit dem bus gefahren',row['stt_final'].casefold()):findings.append('STT changed finding a bus into a bus journey; the summary propagates that change')
   else:findings.append('LLM/memory infers a bus journey that the current transcript only describes as finding a bus')
  if re.search(r'(interview|appointment|interview time).{0,35}(moved|changed|rescheduled)|termin.{0,30}(verschoben|verlegt)',answer):findings.append('LLM/memory adds rescheduling; the current story only corrects ten rather than nine')
  if re.search(r'\[/?(?:en|de)\]',row['reply']):findings.append('Language marker leaked into displayed transcript')
  audio=[]
  for event in row['native_audio']:
   with wave.open(str(owners[event['monotonic_seconds']])) as f:
    rate=f.getframerate();duration=f.getnframes()/rate;signal=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
   rms=float(np.sqrt(np.mean(signal**2))) if len(signal) else 0
   audio.append({'id':event['id'],'language':event['language'],'text':event['text'],'seconds':round(duration,4),'rate':rate,'rms':round(rms,5),'peak':round(float(np.max(np.abs(signal))) if len(signal) else 0,5)})
   if rms<.00001:findings.append('Generated audio is silent: '+event['id'])
   if not any(c.isalnum() for c in event['text']):findings.append('Punctuation-only voice chunk: '+event['id'])
  for check in row.get('tts_audio_audit',[]):
   if check.get('metrics',{}).get('wer',0)>.2:findings.append('TTS re-recognition disagrees with intended phrase: '+check['reference'])
  case_resources=[r for r in resources if r.get('case')==case['id']]
  model_ram={}
  for role in ('stt','tts','server'):
   values=[sum(p['rss_mb'] for p in r['processes'] if process_map.get(p['pid'],p['role'])==role) for r in case_resources]
   model_ram[role]={'peak_mb':round(max(values),2),'mean_mb':round(sum(values)/len(values),2)} if values else None
  case_clocks=[r for r in clocks if r.get('case')==case['id']]
  clock_values=[r['gpu_sm_clock_mhz'] for r in case_clocks if isinstance(r.get('gpu_sm_clock_mhz'),(int,float))]
  power_states=sorted({r.get('pstate','unknown') for r in case_clocks})
  clock_summary={'states':power_states,'min_sm_mhz':min(clock_values) if clock_values else None,'max_sm_mhz':max(clock_values) if clock_values else None,'sample_count':len(case_clocks)}
  intervals=[]
  for event in inputs:
   finish=next((e for e in finals if e.get('turn')==event.get('turn')),None)
   if finish:intervals.append((event['monotonic_seconds'],finish['monotonic_seconds']))
  decode_clocks=[r for r in case_clocks if any(a<=r['monotonic_seconds']<=b for a,b in intervals)]
  decode_states=sorted({r.get('pstate','unknown') for r in decode_clocks});clock_summary['final_stt_states']=decode_states
  if decode_states==['P8']:findings.append('Final STT timing measured with GPU in low-power P8 state')
  if 'P8' in decode_states and 'P0' in decode_states:findings.append('GPU power state transitioned during final STT; inspect clocks before comparing timing')
  rows.append({'case':case['id'],'wer':row['stt_metrics']['wer'],'cer':row['stt_metrics']['cer'],'language_vocabulary_retention':retention,'captured_segments':len(inputs),'delivered_finals':len(finals),'findings':list(dict.fromkeys(findings)),'audio_integrity':audio,'ram_vram':row['resources'],'model_ram':model_ram,'gpu_clock_state':clock_summary,'timings':row['timings'],'review_scope':'Structural checks and transparent semantic heuristics; automatic ASR agreement is not a human pronunciation judgment.'})
 report={'samples_reviewed':len(rows),'findings_counts':dict(Counter(f for row in rows for f in row['findings'])),'samples':rows}
 (folder/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('Reviewed',len(rows),'continuous story runs')
 for text,count in report['findings_counts'].items():print(count,text)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();review(ROOT/a.folder)
