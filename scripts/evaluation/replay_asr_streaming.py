"""Exact-waveform cumulative replay: measure final work after preview computation.

This offline replay does not sleep or simulate a physical microphone queue.
Use the real browser suite to validate actual end-of-recording latency.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from voice_bridge.settings import ROOT, load_config
from voice_bridge.qwen_bridge import QwenBridge
from scripts.evaluation.audio_tools import read_audio
from scripts.evaluation.semantic_checks import check_meaning
from scripts.evaluation.speech_metrics import speech_metrics, normalized_words
from scripts.story_tests.resource_monitor import Resources
from evaluation.config import ASR_LATENCY_EVALUATION as CONFIG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=CONFIG['inputs'])
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch2',action='store_true')
    args=parser.parse_args()
    inputs=ROOT/args.inputs;output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    if (output/'results.jsonl').exists():raise RuntimeError('Preserve existing replay results')
    manifest=json.loads((inputs/'inputs.json').read_text())
    cases=[r for r in manifest['cases'] if r['input_id'] in args.case]
    if len(cases)!=len(set(args.case)) or len(cases)>CONFIG['maximum_recordings']:
        raise ValueError('Unknown, duplicate or excessive replay selection')
    cfg=load_config();stt=dict(cfg['stt'],model_path=cfg['paths']['stt_model'],audit_raw_outputs=True)
    for path in ['config.py','scripts/runtime/qwen_asr_worker.py','src/voice_bridge/qwen_bridge.py','src/voice_bridge/workers.py']:
        target=output/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    profiles={'reference':{'stream_cache':False,'batch_size':1},'cached_serial':{'stream_cache':True,'batch_size':1}}
    if args.batch2:profiles['cached_batch2']={'stream_cache':True,'batch_size':2}
    (output/'configuration.json').write_text(json.dumps({'parameters':CONFIG,'profiles':profiles,
        'selected_ids':[r['input_id'] for r in cases],
        'input_manifest_sha256':hashlib.sha256((inputs/'inputs.json').read_bytes()).hexdigest()},indent=2)+'\n')
    monitor=Resources(os.getpid(),output/'resources.jsonl',CONFIG['resource_interval_seconds']);monitor.start()
    bridge=None;rows=[]
    try:
        bridge=QwenBridge(stt,ROOT)
        first=read_audio(ROOT/cases[0]['audio'])
        bridge.transcribe(first[:32000],None,True,{'strategy':'whole','padding_ms':None,'num_beams':1})
        for index,case in enumerate(cases):
            path=ROOT/case['audio']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==case['audio_sha256']
            audio=read_audio(path)
            order=list(profiles)
            if index%2:order.reverse()
            for profile in order:
                options={'strategy':'adaptive_pause_chunks','num_beams':3,'padding_ms':200,'speech_gate':True,**profiles[profile]}
                identity=case['input_id']+':'+profile
                monitor.label=identity;previews=[];began=time.monotonic()
                for end in range(round(CONFIG['preview_interval_seconds']*16000),len(audio),round(CONFIG['preview_interval_seconds']*16000)):
                    tick=time.monotonic();result=bridge.transcribe(audio[:end],identity,False,options)
                    previews.append({'audio_seconds':end/16000,'seconds':time.monotonic()-tick,'result':result})
                tick=time.monotonic()
                result=bridge.transcribe(audio,identity if options['stream_cache'] else None,True,options)
                final_seconds=time.monotonic()-tick
                row={'input_id':case['input_id'],'kind':case['kind'],'profile':profile,
                     'audio_sha256':case['audio_sha256'],'reference':case['reference'],
                     'previews':previews,'final':result,'final_seconds':final_seconds,
                     'total_compute_seconds':time.monotonic()-began,
                     'metrics':speech_metrics(case['reference'],result['text']),
                     'semantic_checks':check_meaning(result['text'],case['semantic_checks'])}
                rows.append(row)
                with (output/'results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'input':case['input_id'],'profile':profile,'final_seconds':final_seconds,
                                 'wer':row['metrics']['wer'],'cache_hits':result.get('cached_segment_hits'),
                                 'calls':result.get('model_decode_calls')}),flush=True)
    finally:
        if bridge:bridge.close()
        monitor.stop()
    reference={r['input_id']:r for r in rows if r['profile']=='reference'}
    baseline={r['input_id']:r for r in [json.loads(line) for line in
        (ROOT/CONFIG['accuracy_baseline']/'results.jsonl').read_text().splitlines()]}
    frozen_accuracy={identity:normalized_words(row['final']['text'])==normalized_words(baseline[identity]['hypothesis'])
                     for identity,row in reference.items()}
    checks=[]
    for r in rows:
        if r['profile']=='reference':continue
        old=reference[r['input_id']]
        checks.append({'input_id':r['input_id'],'profile':r['profile'],
            'normalized_text_identical':normalized_words(r['final']['text'])==normalized_words(old['final']['text']),
            'routing_unchanged':all(r['final'].get(k)==old['final'].get(k) for k in ('language','can_route','certain')),
            'word_errors_not_worse':r['metrics']['word_errors']<=old['metrics']['word_errors'],
            'character_errors_not_worse':r['metrics']['character_errors']<=old['metrics']['character_errors'],
            'meaning_not_worse':all(not value or r['semantic_checks'].get(key,False) for key,value in old['semantic_checks'].items())})
    summary={'recordings':len(cases),'profiles':list(profiles),'checks':checks,'reference_matches_frozen_accuracy':frozen_accuracy,
             'scope':'Final compute after offline cumulative previews; excludes real-time arrival, worker queue, endpointing, LLM and TTS.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return int(not all(frozen_accuracy.values()) or any(not r['word_errors_not_worse'] or not r['character_errors_not_worse'] or not r['meaning_not_worse'] or not r['routing_unchanged'] for r in checks))


if __name__=='__main__':raise SystemExit(main())
