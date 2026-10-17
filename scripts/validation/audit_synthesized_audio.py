"""Re-recognize synthetic QA TTS WAVs; diagnostic intelligibility, not ground truth."""
import json,time,wave
import numpy as np
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.qwen_bridge import QwenBridge
from scripts.evaluation.evaluate_mixed_language_asr import metrics

if __name__=='__main__':
 folder=ROOT/'logs/full-qa';config=load_settings();config['stt']['model_path']=config['paths']['stt_model']
 bridge=QwenBridge(config['stt'],ROOT);path=folder/'tts-audio-audit.json'
 rows=json.loads(path.read_text()) if path.exists() else [];seen={(e['language'],e['text']) for e in rows}
 try:
  for path in sorted((folder/'pipeline-traces').glob('*/events.jsonl')):
   for line in path.read_text().splitlines():
    event=json.loads(line)
    if event['event']!='tts_audio':continue
    key=(event['language'],event['text'])
    if key in seen:continue
    seen.add(key)
    with wave.open(str(path.parent/event['audio'])) as f:
     rate=f.getframerate();audio=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
    samples=np.interp(np.arange(0,len(audio),rate/16000),np.arange(len(audio)),audio).astype(np.float32)
    started=time.perf_counter();result=bridge.transcribe(samples,None,True)
    row={**event,'session':path.parent.name,'recognized':result['text'],'detected_language':result['language'],
         'metrics':metrics(event['text'],result['text']),'recognition_seconds':round(time.perf_counter()-started,3),
         'peak':round(float(np.max(np.abs(audio))),4),'rms':round(float(np.sqrt(np.mean(audio*audio))),4)}
    rows.append(row)
    (folder/'tts-audio-audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(len(rows),event['language'],row['metrics'],event['text'],flush=True)
 finally:bridge.close()
