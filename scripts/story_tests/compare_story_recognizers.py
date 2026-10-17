"""Compare alternative recognizers on the exact captured story PCM, no translation task."""
import argparse,gc,json,os,time,wave,hashlib
from pathlib import Path
import numpy as np
from collections import Counter
from voice_bridge.settings import ROOT
from scripts.evaluation.evaluate_mixed_language_asr import metrics,normalized
from scripts.story_tests.resource_monitor import Resources


def main():
 parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('--model',choices=('large-v3','turbo','qwen-fresh','qwen-replay'),default='large-v3');parser.add_argument('--only-failures',action='store_true');parser.add_argument('--quality-only',action='store_true');parser.add_argument('--whisper-chunk',type=int,choices=(5,30),default=30);a=parser.parse_args();folder=ROOT/a.folder
 data=json.loads((folder/'results.json').read_text());owners={}
 for path in (folder/'pipeline-traces').glob('*/events.jsonl'):
  for line in path.read_text().splitlines():
   event=json.loads(line)
   if event['event']=='stt_input':owners[event['monotonic_seconds']]=path.parent/event['audio']
 artifact_name=a.model if a.model.startswith('qwen-') else a.model+'-'+str(a.whisper_chunk)+'s'
 fresh_path=folder/'candidate-qwen-fresh.json';fresh_scores={r['case']:r['metrics']['wer'] for r in json.loads(fresh_path.read_text())['samples']} if fresh_path.exists() else {}
 from faster_whisper import WhisperModel
 path=ROOT/'models/whisper-large-v3' if a.model=='large-v3' else Path('/mnt/d/voice-bridge-models/faster-whisper-large-v3-turbo')
 monitor=Resources(os.getpid(),folder/f'candidate-{artifact_name}-resources.jsonl');monitor.start();started=time.monotonic()
 if a.model.startswith('qwen-'):
  from voice_bridge.qwen_bridge import QwenBridge
  from voice_bridge.settings import load_config
  config=load_config();config['stt']['model_path']=config['paths']['stt_model'];config['stt']['audit_raw_outputs']=True;path=ROOT/config['paths']['stt_model'];model=QwenBridge(config['stt'],ROOT)
 else:model=WhisperModel(str(path),device='cuda',compute_type='int8_float32',cpu_threads=2,num_workers=1,local_files_only=True)
 language_audit=[]
 if not a.model.startswith('qwen-'):
  class LanguageAudit:
   def __init__(self,real):self.real=real
   def __getattr__(self,name):return getattr(self.real,name)
   def detect_language(self,*args,**kwargs):
    result=self.real.detect_language(*args,**kwargs);language_audit.append({'monotonic_seconds':time.monotonic(),'top_languages':result[0][:5]});return result
  model.model=LanguageAudit(model.model)
 load_seconds=time.monotonic()-started;rows=[]
 try:
  for row in data['samples']:
   case=row['case']
   if a.quality_only:continue
   if a.only_failures and max(row['stt_metrics']['wer'],fresh_scores.get(case['id'],0))<=.15 and case['variant']!='clean':continue
   monitor.label=case['id'];language_before=len(language_audit);texts=[];languages=[];updates=[];raw_finals=[];input_errors=[];decode_start=time.monotonic()
   inputs=[e for e in row['pipeline_trace'] if e['event']=='stt_input']
   for event in inputs:
    with wave.open(str(owners[event['monotonic_seconds']])) as f:
     assert f.getframerate()==16000 and f.getnchannels()==1
     audio=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
    try:
     if a.model.startswith('qwen-'):
      identity=None
      if a.model=='qwen-replay':
       identity=case['id']+':'+str(event['monotonic_seconds'])
       for end in np.arange(2.2,len(audio)/16000,2):
        before=time.monotonic();partial=model.transcribe(audio[:int(end*16000)],identity,False);updates.append({'audio_seconds':float(end),'decode_seconds':round(time.monotonic()-before,4),**partial})
      decoded=model.transcribe(audio,identity,True);raw_finals.append(decoded);texts.append(decoded['text']);languages.append(decoded['language'])
     else:
      segments,info=model.transcribe(audio,language=None,task='transcribe',beam_size=3,vad_filter=False,condition_on_previous_text=False,multilingual=True,chunk_length=a.whisper_chunk)
      texts.append(' '.join(s.text.strip() for s in segments));languages.append(info.language)
    except Exception as error:
     input_errors.append(type(error).__name__+': '+str(error));texts.append('');languages.append('error')
   ended=time.monotonic();text=' '.join(texts)
   words=normalized(text);reference_words=normalized(case['reference']);ngrams=Counter(tuple(words[i:i+8]) for i in range(max(0,len(words)-7)));reference_ngrams=Counter(tuple(reference_words[i:i+8]) for i in range(max(0,len(reference_words)-7)));repetitions=[{'text':' '.join(k),'count':v} for k,v in ngrams.items() if v>=3 and v>reference_ngrams[k]]
   result={'case':case['id'],'reference':case['reference'],'text':text,'languages':languages,'metrics':metrics(case['reference'],text),'decode_seconds':round(ended-decode_start,4),'resources':monitor.summarize(decode_start,ended),'captured_segments':len(inputs),'progressive_replay':a.model=='qwen-replay','partials':updates,'raw_final_outputs':raw_finals,'errors':input_errors,'repeated_unheard_ngrams':repetitions,'language_detection_events':language_audit[language_before:]}
   if not a.model.startswith('qwen-') and case['variant']=='clean':
    control_chunk=5 if a.whisper_chunk==30 else 30;monitor.label=case['id']+':'+str(control_chunk)+'s-control';control_start=time.monotonic();control_text=[];control_language_before=len(language_audit)
    for event in inputs:
     with wave.open(str(owners[event['monotonic_seconds']])) as f:control_audio=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
     segments,info=model.transcribe(control_audio,language=None,task='transcribe',beam_size=3,vad_filter=False,condition_on_previous_text=False,multilingual=True,chunk_length=control_chunk)
     control_text.append(' '.join(s.text.strip() for s in segments))
    joined=' '.join(control_text);control_end=time.monotonic()
    result['alternative_window_control']={'chunk_seconds':control_chunk,'text':joined,'metrics':metrics(case['reference'],joined),'decode_seconds':round(control_end-control_start,4),'language_detection_events':language_audit[control_language_before:],'resources':monitor.summarize(control_start,control_end)}
   rows.append(result)
   (folder/f'candidate-{artifact_name}.json').write_text(json.dumps({'model':str(path),'compute_type':'float32' if a.model.startswith('qwen-') else 'int8_float32','task':'transcribe','language':None,'beam_size':3,'whisper_multilingual':not a.model.startswith('qwen-'),'whisper_chunk_seconds':a.whisper_chunk if not a.model.startswith('qwen-') else None,'load_seconds':load_seconds,'scope':'Same browser-captured audio as the application Qwen final; final decoding only, without VAD waiting, microphone arrival, cloud LLM or playback.','samples':rows},ensure_ascii=False,indent=2)+'\n')
   print(artifact_name,case['id'],result['metrics'],result['decode_seconds'],'seconds',flush=True)
  if a.model=='qwen-fresh':
   # Check every actual generated WAV; reuse only identical audio bytes and text.
   audio_owners={}
   for path in (folder/'pipeline-traces').glob('*/events.jsonl'):
    for line in path.read_text().splitlines():
     event=json.loads(line)
     if event['event']=='tts_audio':audio_owners[event['monotonic_seconds']]=path.parent/event['audio']
   cache={}
   for saved in (folder/'samples').glob('*.json'):
    for audit in json.loads(saved.read_text()).get('tts_audio_audit',[]):
     if audit.get('wav_sha256') and not audit.get('error'):cache[(audit['wav_sha256'],audit['reference'],audit['expected_language'])]={k:v for k,v in audit.items() if k!='id'}
   for row in data['samples']:
    monitor.label='tts-audit:'+row['case']['id'];audits=[]
    for event in row['native_audio']:
     path=audio_owners[event['monotonic_seconds']];digest=hashlib.sha256(path.read_bytes()).hexdigest();key=(digest,event['text'],event['language'])
     if key not in cache:
      with wave.open(str(path)) as f:
       rate=f.getframerate();samples=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
      audio=np.interp(np.arange(0,len(samples),rate/16000),np.arange(len(samples)),samples).astype(np.float32)
      try:
       before=time.monotonic();decoded=model.transcribe(audio,None,True)
       cache[key]={'expected_language':event['language'],'reference':event['text'],'recognized':decoded['text'],'detected_language':decoded['language'],'metrics':metrics(event['text'],decoded['text']),'seconds':round(time.monotonic()-before,4),'wav_sha256':digest}
      except Exception as error:cache[key]={'error':type(error).__name__+': '+str(error),'wav_sha256':digest,'reference':event['text'],'expected_language':event['language']}
     audits.append({'id':event['id'],**cache[key]})
    row['tts_audio_audit']=audits
    (folder/'samples'/f'{row["case"]["id"]}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    print('Exact-WAV audio audit',row['case']['id'],len(audits),'chunks',flush=True)
   (folder/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 finally:
  monitor.stop()
  if a.model.startswith('qwen-'):model.close()
  del model;gc.collect()
if __name__=='__main__':main()
