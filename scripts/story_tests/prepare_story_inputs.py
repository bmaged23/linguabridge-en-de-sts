"""Generate frozen inputs using correct per-language voices, independently of STT."""
import argparse,json,wave,hashlib
from pathlib import Path
import numpy as np
from voice_bridge.settings import ROOT,load_config
from voice_bridge.workers import init_tts,synthesize
from scripts.story_tests.story_test_cases import cases


def prepare(folder,rows=None):
 folder=Path(folder);(folder/'inputs').mkdir(parents=True,exist_ok=True)
 config=load_config();config['tts']['device']='cpu';init_tts(config['tts'],ROOT)
 rng=np.random.default_rng(731904);cache={};rows=cases() if rows is None else rows
 for case in rows:
  parts=[]
  for language,text in case['spans']:
   key=(language,text)
   if key not in cache:
    pcm,rate=synthesize(text,language);a=np.frombuffer(pcm,'<i2').astype(np.float32)/32768
    cache[key]=np.interp(np.arange(0,len(a),rate/16000),np.arange(len(a)),a).astype(np.float32)
   parts.extend([cache[key],np.zeros(7200 if 'pause' in case['variant'] else 1920,np.float32)])
  audio=np.concatenate(parts) if parts else np.zeros(16000*6,np.float32)
  if 'noise' in case['variant']:
   snr=int(case['variant'].rsplit('noise',1)[1]);power=np.sqrt(np.mean(audio**2));audio=audio+rng.normal(0,power/(10**(snr/20)),len(audio))
  if 'quiet' in case['variant']:audio=audio*.18
  elif case['variant']=='background':audio=rng.normal(0,.0007,len(audio))
  audio=np.concatenate([np.zeros(4800),audio,np.zeros(1600)])
  pcm=(np.clip(audio,-1,1)*32767).astype('<i2').tobytes();path=folder/'inputs'/f'{case["id"]}.wav'
  with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm)
  case.update(input=str(path.relative_to(folder)),sha256=hashlib.sha256(pcm).hexdigest(),seconds=round(len(audio)/16000,3),peak=round(float(np.max(np.abs(audio))),4),rms=round(float(np.sqrt(np.mean(audio**2))),5))
  print('Prepared',case['id'],case['seconds'],flush=True)
 (folder/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 return rows
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('--extended',action='store_true');args=parser.parse_args()
 if args.extended:
  from scripts.story_tests.story_test_cases import extended_cases
  prepare(args.folder,extended_cases())
 else:prepare(args.folder)
