"""Prepare synthetic microphone inputs on CPU; no second GPU voice process."""
import json,wave
import numpy as np
from scripts.evaluation.evaluate_mixed_language_asr import CASES as MIXED_CASES
from voice_bridge.settings import ROOT,load_config
from voice_bridge.workers import init_tts,synthesize
from scripts.validation.inspect_cloud_protocol import CASES

PROMPTS=[*CASES,
 ('enable_translation','en','Bridge, translator mode.'),
 ('confirm_translation','en','Yes, confirm.'),
 ('translate_english','en','Where is the nearest train station?'),
 ('translate_german','de','Ich möchte einen Kaffee, bitte.'),
 ('short_response','en','Please answer with just one short sentence.'),
 ('interrupt','en','Stop. Tell me the time in German instead.'),
 ('interrupt_short','en','Stop. Please say hello in one short sentence.'),
 ('memory_recall','en','What is my German level? Please answer in one short sentence.'),
 ('memory_explicit','en','Do you remember my saved German language proficiency level?'),
]
PROMPTS=[(name,lang,'Explain the difference between although and because using German words and examples.' if name=='isolated_vocabulary' else text) for name,lang,text in PROMPTS]

if __name__=='__main__':
    folder=ROOT/'logs/full-qa';folder.mkdir(exist_ok=True)
    c=load_config();c['tts']['device']='cpu';init_tts(c['tts'],ROOT)
    for name,lang,text in PROMPTS:
        path=folder/f'{name}.wav'
        if path.exists():continue
        pcm,rate=synthesize(text,lang)
        with wave.open(str(path),'wb') as handle:
            handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate);handle.writeframes(pcm)
        print('Prepared',name,flush=True)
    for name,spans in MIXED_CASES:
        path=folder/f'{name}.wav'
        if path.exists():continue
        pieces=[]
        for lang,text in spans:
            pcm,rate=synthesize(text,lang);audio=np.frombuffer(pcm,'<i2').astype(np.float32)/32768
            pieces.extend([np.interp(np.arange(0,len(audio),rate/16000),np.arange(len(audio)),audio),np.zeros(1920)])
        with wave.open(str(path),'wb') as handle:
            handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(16000);handle.writeframes((np.clip(np.concatenate(pieces),-1,1)*32767).astype('<i2').tobytes())
    (folder/'prompts.json').write_text(json.dumps(PROMPTS,ensure_ascii=False,indent=2)+'\n')
