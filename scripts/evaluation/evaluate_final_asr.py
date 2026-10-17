"""Compare frozen progressive-prefix finals with independent full-audio decoding."""
import json,runpy,wave
import numpy as np
from voice_bridge.settings import ROOT
from voice_bridge.qwen_bridge import QwenBridge
from scripts.evaluation.evaluate_mixed_language_asr import metrics

if __name__=='__main__':
 c=runpy.run_path(str(ROOT/'config.py'))['CONFIG'];s=c['stt'];s['model_path']=c['paths']['stt_model']
 bridge=QwenBridge(s,ROOT);prompts=json.loads((ROOT/'logs/full-qa/prompts.json').read_text());rows=[]
 try:
  for name in ('coffee_lesson','memory_recall','recovery','translate_english','german_lesson'):
   text=next(t for n,_,t in prompts if n==name)
   with wave.open(str(ROOT/f'logs/full-qa/{name}.wav')) as f:
    rate=f.getframerate();audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
   audio=np.interp(np.arange(0,len(audio),rate/16000),np.arange(len(audio)),audio).astype(np.float32)
   audio=np.concatenate([np.zeros(512*10,np.float32),audio,np.zeros(16000,np.float32)])
   for end in np.arange(2.2,len(audio)/16000,2):bridge.transcribe(audio[:int(end*16000)],name,False)
   progressive=bridge.transcribe(audio,name,True)['text']
   fresh=bridge.transcribe(audio,name+'-fresh',True)['text']
   row={'case':name,'reference':text,'progressive':progressive,'fresh':fresh,
        'progressive_metrics':metrics(text,progressive),'fresh_metrics':metrics(text,fresh)}
   rows.append(row);print(row,flush=True)
 finally:bridge.close()
 (ROOT/'logs/full-qa/final-asr-comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
