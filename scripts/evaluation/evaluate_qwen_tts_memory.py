"""Check actual FP32 Qwen TTS / deployed FP32 Qwen ASR GPU co-residency."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def gpu():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0].split(',')
    return {'used_mib':float(values[0]),'total_mib':float(values[1])}


def main():
    import torch
    torch.set_num_threads(2)
    from qwen_tts import Qwen3TTSModel
    import runpy
    from voice_bridge.qwen_bridge import QwenBridge
    result={'dtype':'float32','attention':'eager','baseline':gpu()}
    model=Qwen3TTSModel.from_pretrained('/mnt/d/voice-bridge-models/qwen3-tts-0.6b',
        device_map='cuda:0',dtype=torch.float32,attn_implementation='eager',local_files_only=True)
    result['tts_loaded']=gpu()
    result['components']={name:sum(p.numel() for p in child.parameters()) for name,child in model.model.named_children()}
    tokenizer=model.model.speech_tokenizer
    result['audio_tokenizer_parameters']=sum(p.numel() for p in tokenizer.model.parameters())
    result['audio_tokenizer_dtype']=str(tokenizer.model.dtype)
    from scripts.evaluation.compare_speech_synthesis import FOLDER,read_wav,load_rows
    import torch.nn.functional as functional
    reference,rate=read_wav(FOLDER/'reference-kokoro.wav')
    reference_embedding=model.model.extract_speaker_embedding(reference,rate).float()
    similarity=[]
    for row in load_rows('baseline')+load_rows('qwen-float32'):
        audio,rate=read_wav(FOLDER/row['audio'])
        embedding=model.model.extract_speaker_embedding(audio,rate).float()
        cosine=float(functional.cosine_similarity(reference_embedding,embedding,dim=0))
        similarity.append({'name':row['name'],'variant':row['variant'],'group':row['group'],
                           'audio':row['audio'],'cosine_to_reference':round(cosine,4)})
    (FOLDER/'speaker-similarity.json').write_text(json.dumps({'method':'Cosine similarity in Qwen TTS own speaker encoder; internal diagnostic, not independent speaker verification or a human similarity score. Reference is synthetic Kokoro af_heart. Short clips and language affect embeddings.','cases':similarity},indent=2)+'\n')
    print('Speaker similarity diagnostic complete',flush=True)
    settings=runpy.run_path(str(ROOT/'config.py'))['CONFIG'];settings['stt']['model_path']=settings['paths']['stt_model']
    started=time.monotonic();bridge=None
    try:
        bridge=QwenBridge(settings['stt'],ROOT)
        result['asr_initialization']='passed';result['both_loaded']=gpu()
        from scripts.evaluation.compare_speech_synthesis import resample
        audio,rate=read_wav(FOLDER/'en_booking-kokoro.wav')
        started_inference=time.monotonic()
        packet=bridge.transcribe(resample(audio,rate,16000))
        result['asr_real_audio_seconds']=round(time.monotonic()-started_inference,3)
        result['asr_real_audio_words']=len(packet.get('text','').split())
        result['after_asr_inference']=gpu()
        prompt=model.create_voice_clone_prompt(ref_audio=str(FOLDER/'reference-kokoro.wav'),ref_text='Hello, I am your language assistant. We can practice together and take one step at a time.')
        torch.manual_seed(20261016)
        started_inference=time.monotonic()
        wavs,rate=model.generate_voice_clone(text='Hello, I am ready to help.',language='English',voice_clone_prompt=prompt,max_new_tokens=256)
        torch.cuda.synchronize()
        result['tts_real_audio_generation_seconds']=round(time.monotonic()-started_inference,3)
        result['tts_real_audio_seconds']=round(len(wavs[0])/rate,3)
        result['after_tts_inference']=gpu()
        result['real_inference_check']='passed'
    except Exception as error:
        result.setdefault('asr_initialization','failed');result['real_inference_check']='failed';result['error']=str(error)
        result['after_attempt']=gpu()
    finally:
        if bridge is not None:bridge.close()
    result['asr_load_attempt_seconds']=round(time.monotonic()-started,3)
    path=ROOT/'docs/tts-comparison/co-residency.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
