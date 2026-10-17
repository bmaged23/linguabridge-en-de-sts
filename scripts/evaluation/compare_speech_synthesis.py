"""Reproducible local TTS round-trip and performance comparison; no user memory/cloud.

Run stages in separate processes to release GPU weights between models.
Gold language spans evaluate synthesis, not automatic language classification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'docs/tts-comparison'
CASES=json.loads((ROOT/'scripts/evaluation/speech_synthesis_test_cases.json').read_text())
REFERENCE='Hello, I am your language assistant. We can practice together and take one step at a time.'


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_rows(stage):
    path=FOLDER/f'{stage}.json'
    return json.loads(path.read_text()) if path.exists() else []


def write_wav(path,audio,rate):
    audio=np.asarray(audio,dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all():raise ValueError('Invalid generated waveform')
    with wave.open(str(path),'wb') as handle:
        handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate)
        handle.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())


def read_wav(path):
    with wave.open(str(path)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768,handle.getframerate()


def resample(audio,rate,target=24000):
    if rate==target:return audio
    return np.interp(np.arange(0,len(audio),rate/target),np.arange(len(audio)),audio).astype(np.float32)


class Resources:
    def __enter__(self):
        import psutil
        self.samples=[];self.stop=threading.Event();self.process=psutil.Process()
        def run():
            self.process.cpu_percent()
            while not self.stop.is_set():
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0].split(',')
                    self.samples.append({'vram_mib':float(gpu[0]),'gpu_percent':float(gpu[1]),
                        'cpu_machine_percent':self.process.cpu_percent()/psutil.cpu_count(),
                        'pss_mib':self.process.memory_full_info().pss/1024**2})
                except (OSError,subprocess.SubprocessError):pass
                self.stop.wait(.25)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
    def result(self):
        if not self.samples:return {}
        return {key:{'mean':round(sum(s[key] for s in self.samples)/len(self.samples),2),
                     'peak':round(max(s[key] for s in self.samples),2)} for key in self.samples[0]}


def record(stage,case,variant,generate,rows):
    destination=FOLDER/f'{case["name"]}-{variant}.wav'
    key=(case['name'],variant)
    if any((r['name'],r['variant'])==key and (FOLDER/r['audio']).exists() for r in rows):return
    started=time.monotonic()
    with Resources() as monitor:
        audio,rate,first_seconds=generate()
    elapsed=time.monotonic()-started
    write_wav(destination,audio,rate)
    seconds=len(audio)/rate
    row={'name':case['name'],'group':case['group'],'variant':variant,
         'reference':' '.join(text for _,text in case['spans']),
         'audio':destination.name,'sample_rate':rate,'audio_seconds':round(seconds,3),
         'generation_seconds':round(first_seconds['total'],3),
         'first_available_audio_seconds':round(first_seconds['first'],3),
         'measurement_wall_seconds':round(elapsed,3),'rtf':round(first_seconds['total']/seconds,3),
         'resources':monitor.result(),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
    rows.append(row);save_json(FOLDER/f'{stage}.json',rows)
    print(f'{variant} {case["name"]}: {row["generation_seconds"]}s for {seconds:.2f}s audio, RTF {row["rtf"]}',flush=True)


def baseline():
    from voice_bridge.settings import load_config
    from voice_bridge import workers
    config=load_config();workers.init_tts(config['tts'],ROOT);workers.warmup_tts()
    pcm,rate=workers.synthesize(REFERENCE,'en')
    write_wav(FOLDER/'reference-kokoro.wav',np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768,rate)
    save_json(FOLDER/'reference.json',{'text':REFERENCE,'source':'Synthetic Kokoro af_heart; not a recording of the user. Same reference for every Qwen language and case.'})
    rows=load_rows('baseline')
    def generate(spans):
        start=time.monotonic();pieces=[];first=None
        for lang,text in spans:
            pcm,rate=workers.synthesize(text,lang)
            if first is None:first=time.monotonic()-start
            audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
            pieces.append(resample(audio,rate))
        return np.concatenate(pieces),24000,{'total':time.monotonic()-start,'first':first}
    for case in CASES:
        spans=case['spans']
        record('baseline',case,'hybrid_gold_spans' if case['group']=='mixed' else ('kokoro' if case['group']=='en' else 'piper'),lambda:generate(spans),rows)
        if case['group']=='mixed':
            language=spans[0][0]
            record('baseline',case,'current_single_voice',lambda:generate([(language,' '.join(t for _,t in spans))]),rows)


def qwen(args):
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(2)
    dtype=getattr(torch,args.dtype)
    model=Qwen3TTSModel.from_pretrained(args.model,device_map='cuda:0',dtype=dtype,
                                       attn_implementation='eager',local_files_only=True)
    print('Loaded Qwen TTS',args.dtype,flush=True)
    prompt=model.create_voice_clone_prompt(ref_audio=str(FOLDER/'reference-kokoro.wav'),ref_text=REFERENCE,x_vector_only_mode=False)
    def generate(spans):
        pieces=[];first=None;start=time.monotonic()
        for index,(lang,text) in enumerate(spans):
            torch.manual_seed(20261016+index)
            wavs,rate=model.generate_voice_clone(text=text,language={'en':'English','de':'German','auto':'Auto'}[lang],
                voice_clone_prompt=prompt,max_new_tokens=512)
            torch.cuda.synchronize()
            if first is None:first=time.monotonic()-start
            pieces.append(resample(wavs[0],rate))
        return np.concatenate(pieces),24000,{'total':time.monotonic()-start,'first':first}
    generate([('en','Hello, I am ready to help.')])
    metadata={'model':'Qwen/Qwen3-TTS-12Hz-0.6B-Base','revision':'5d83992436eae1d760afd27aff78a71d676296fc',
      'dtype':args.dtype,'attention':'eager','reference':'reference-kokoro.wav','seed':20261016,
      'warmup':True,'max_new_tokens':512,'parameters_total':sum(p.numel() for p in model.model.parameters()),
      'generation_defaults':model.generate_defaults,
      'streaming':'SDK generate_voice_clone returns full waveforms, not streaming audio. First-audio timings are per completed call/segment.'}
    save_json(FOLDER/'qwen-settings.json',metadata)
    rows=load_rows('qwen-'+args.dtype)
    for case in CASES:
        spans=case['spans'];language=case['group'] if case['group']!='mixed' else 'auto'
        record('qwen-'+args.dtype,case,'qwen_'+args.dtype+'_auto' if language=='auto' else 'qwen_'+args.dtype,
               lambda:generate([(language,' '.join(t for _,t in spans))]),rows)
        if case['group']=='mixed':
            record('qwen-'+args.dtype,case,'qwen_'+args.dtype+'_gold_spans',lambda:generate(spans),rows)


def judge(args):
    from scripts.evaluation.evaluate_mixed_language_asr import metrics,normalized,distance
    if args.scorer=='qwen':
        import torch
        from qwen_asr import Qwen3ASRModel
        model=Qwen3ASRModel.from_pretrained(str(ROOT/'models/qwen3-asr-0.6b'),dtype=torch.float32,
            device_map='cuda:0',attn_implementation='eager',max_inference_batch_size=1,max_new_tokens=256,local_files_only=True)
    else:
        from faster_whisper import WhisperModel
        model=WhisperModel('/mnt/d/voice-bridge-models/faster-whisper-large-v3-turbo',device='cuda',compute_type='int8_float32',cpu_threads=2,local_files_only=True)
    rows=load_rows('baseline')+load_rows('qwen-'+args.dtype)
    outputs=load_rows('scored-'+args.scorer+'-'+args.dtype)
    for row in rows:
        if any(r['sha256']==row['sha256'] and r['variant']==row['variant'] for r in outputs):continue
        audio,rate=read_wav(FOLDER/row['audio'])
        if args.scorer=='qwen':
            result=model.transcribe(audio=(audio,rate),language=None,
                context='Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate.')[0]
            text=result.text
        else:
            samples=resample(audio,rate,16000)
            segments,_=model.transcribe(samples,language=None,task='transcribe',multilingual=True,
                beam_size=3,condition_on_previous_text=False,vad_filter=False)
            text=' '.join(segment.text.strip() for segment in segments)
        rw,hw=normalized(row['reference']),normalized(text)
        rc,hc=' '.join(rw),' '.join(hw)
        output={**row,'scorer':args.scorer,'hypothesis':text,**metrics(row['reference'],text),'reference_words':len(rw),
                'word_errors':distance(rw,hw),'reference_characters':len(rc),'character_errors':distance(rc,hc)}
        outputs.append(output);save_json(FOLDER/('scored-'+args.scorer+'-'+args.dtype+'.json'),outputs)
        print(f'{row["variant"]} {row["name"]}: WER {output["wer"]:.1%}, CER {output["cer"]:.1%}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['baseline','qwen','judge'])
    parser.add_argument('--scorer',choices=['qwen','whisper'],default='qwen')
    parser.add_argument('--dtype',choices=['float16','float32'],default='float16')
    parser.add_argument('--model',default='/mnt/d/voice-bridge-models/qwen3-tts-0.6b')
    args=parser.parse_args();FOLDER.mkdir(parents=True,exist_ok=True)
    {'baseline':baseline,'qwen':lambda:qwen(args),'judge':lambda:judge(args)}[args.stage]()
