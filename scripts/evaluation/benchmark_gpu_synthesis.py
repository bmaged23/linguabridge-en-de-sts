"""Compare warmed GPU serial versus concurrent chunks alongside resident Qwen."""
import json
import time
import os
from pathlib import Path
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.workers import init_tts,synthesize,synthesize_batch,warmup_tts
from voice_bridge.core import PhraseBuffer


def main():
    config=load_settings();settings=config['tts']
    settings.update(device='cuda',parallel_chunks=2,cuda={'device_id':0,'gpu_mem_limit':768*1024**2,
        'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC',
        'cudnn_conv_use_max_workspace':'0','do_copy_in_default_stream':'1'})
    from scripts.evaluation.profile_voice_pipeline import Monitor
    monitor=Monitor(os.getpid());monitor.start()
    init_tts(settings,ROOT);warmup_tts()
    texts={'en':'I can help you move your appointment to Friday afternoon. Please check whether the later train arrives before your meeting, and let me know if you would like another option.',
        'de':'Ich kann Ihnen helfen, den Termin auf Freitagnachmittag zu verschieben. Bitte prüfen Sie, ob der spätere Zug vor Ihrem Treffen ankommt, und sagen Sie mir, ob Sie eine andere Verbindung benötigen.'}
    result={}
    for language,text in texts.items():
        phrases=PhraseBuffer(40,180).push(text,flush=True);result[language]={}
        for strategy in ('serial','parallel2'):
            monitor.phase=language+'_'+strategy
            rows=[]
            for _ in range(3):
                start=time.monotonic()
                first=synthesize(phrases[0],language);first_seconds=time.monotonic()-start
                rest=([synthesize(p,language) for p in phrases[1:]] if strategy=='serial' else synthesize_batch(phrases[1:],language))
                elapsed=time.monotonic()-start
                audio=[first,*rest]
                row={'first_chunk_seconds':round(first_seconds,3),'total_seconds':round(elapsed,3),
                    'audio_seconds':round(sum(len(pcm)/2/rate for pcm,rate in audio),3),'chunks':len(audio)}
                rows.append(row);print(language,strategy,row,flush=True)
            result[language][strategy]=rows
            (ROOT/'logs/gpu-tts-benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    result['resources']=monitor.finish()
    (ROOT/'logs/gpu-tts-benchmark.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
