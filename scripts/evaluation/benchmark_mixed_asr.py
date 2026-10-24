"""Frozen mixed-speech ablation: one GPU model, exact WAVs, no reference hints."""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from voice_bridge.settings import ROOT, load_config
from voice_bridge.qwen_bridge import QwenBridge
from scripts.evaluation.audio_tools import read_audio, resample, write_audio
from scripts.evaluation.run_component_benchmarks import streamed_synthesis
from scripts.evaluation.speech_metrics import speech_metrics, aggregate_metrics
from scripts.evaluation.semantic_checks import check_meaning
from scripts.runtime.qwen_asr_worker import pause_segments
from scripts.story_tests.resource_monitor import Resources
from evaluation.config import MIXED_ASR_EVALUATION as CONFIG


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    from voice_bridge import workers
    folder = args.inputs
    folder.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT/CONFIG['dataset']
    if (folder/'inputs.json').exists():
        value = load(folder/'inputs.json')
        assert value['dataset_sha256'] == sha(dataset_path)
        assert all(sha(ROOT/r['audio']) == r['audio_sha256'] for r in value['cases'])
        print('Frozen input hashes verified.', flush=True)
        return
    dataset = load(dataset_path)
    rows = []
    cfg = load_config()
    workers.init_tts(cfg['tts'], ROOT)
    workers.warmup_tts()
    for case in dataset['cases']:
        captured, _, _ = asyncio.run(streamed_synthesis(case, cfg, workers, folder/(case['case_id']+'.sqlite3')))
        pieces = [np.zeros(CONFIG['padding_ms']*16, np.float32)]
        for _, _, (pcm, rate) in captured:
            if len(pieces)>1:
                pieces.append(np.zeros(CONFIG['gap_ms']*16, np.float32))
            pieces.append(resample(np.frombuffer(pcm, '<i2').astype(np.float32)/32768, rate, 16000))
        pieces.append(np.zeros(CONFIG['padding_ms']*16, np.float32))
        clean = np.concatenate(pieces)
        for snr in dataset['snr_db']:
            identity = case['case_id']+('--clean' if snr is None else f'--noise{snr}')
            audio = clean.copy()
            if snr is not None:
                seed = dataset['noise_seed'] ^ int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
                rms = float(np.sqrt(np.mean(clean*clean)))
                audio += np.random.default_rng(seed).normal(0, rms/(10**(snr/20)), len(clean)).astype(np.float32)
            path = folder/(identity+'.wav')
            fingerprint = write_audio(path, audio, 16000)
            rows.append({**case, 'input_id':identity, 'kind':'new_synthetic', 'snr_db':snr,
                         'audio':str(path.relative_to(ROOT)), 'audio_sha256':fingerprint,
                         'audio_seconds':len(audio)/16000})
        print('prepared',case['case_id'],flush=True)
    omitted=[]
    captures=args.capture_run or [Path('.evaluation-local')/name for name in
        ('full-flow-v1','fresh-final-flow-v2','full-flow-v3','teaching-fix-flow-v1')]
    for capture in captures:
        source=(ROOT/capture).resolve();name=source.name
        if not (source/'cases/mixed-connected-story.json').is_file():
            omitted.append(str(capture));continue
        case = load(source/'cases/mixed-connected-story.json')
        event = next(e for e in case['pipeline_trace'] if e['event']=='stt_input')
        original = source/'pipeline-traces'/event['trace_session']/event['audio']
        path = folder/('capture-'+name+'.wav')
        shutil.copyfile(original,path)
        rows.append({'case_id':'original-noisy-story','input_id':'capture-'+name, 'split':'development',
                     'kind':'captured', 'reference':case['reference'], 'audio':str(path.relative_to(ROOT)),
                     'audio_sha256':sha(path),'audio_seconds':event['audio_seconds'],
                     'semantic_checks':[
                         {'name':'cancelled_train','any_patterns':[r'(?:zug|train).*fiel aus',r'train.*cancelled',r'zug.*ausgefallen'],
                          'forbidden_patterns':[r'filled out',r'vollen zug',r'full train',r'train was full']},
                         {'name':'tickets_not_bought','any_patterns':[r'(?:noch )?kein(?:e|en) fahrkarten',r'(?:no|not).*tickets']},
                         {'name':'ten_not_nine','any_patterns':[r'(?:ten|10).*not (?:nine|9)']}]})
    human = load(ROOT/'evaluation/datasets/fleurs-calibration-v1.json')['cases']
    for language in ('en','de'):
        for case in [c for c in human if c['language']==language]:
            if not (ROOT/case['audio']).is_file():
                omitted.append(case['audio']);continue
            rows.append({**case, 'input_id':case['case_id'],'split':'control','kind':'human_control','semantic_checks':[]})
    save(folder/'inputs.json',{'dataset_sha256':sha(dataset_path),'cases':rows,
        'noise_scope':'Noise SNR relative to complete clean recording RMS, including short inserted silence.',
        'capture_scope':'Captured variants of one source story are not independent stories.',
        'unavailable_optional_inputs':omitted})


def evaluate(args):
    folder = args.output
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.jsonl').exists():
        raise RuntimeError('Preserve previous experiment; choose a new output directory')
    for source in ('config.py','evaluation/config.py','scripts/runtime/qwen_asr_worker.py',
                   'src/voice_bridge/qwen_bridge.py','scripts/evaluation/benchmark_mixed_asr.py',
                   'requirements/stt-cuda.txt','requirements.txt'):
        target=folder/'source'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,target)
    manifest = load(args.inputs/'inputs.json')
    assert manifest['dataset_sha256']==sha(ROOT/CONFIG['dataset'])
    cases = [r for r in manifest['cases'] if args.split=='all' or r['split'] in (args.split,'control')]
    if args.case:
        cases = [r for r in cases if r['input_id'] in args.case]
    methods = args.method or list(CONFIG['methods'])
    if not cases or len(cases)*len(methods)>CONFIG['maximum_decodes']:
        raise RuntimeError('Invalid selection or decode budget exceeded')
    cfg = load_config()
    stt = dict(cfg['stt'], model_path=cfg['paths']['stt_model'], audit_raw_outputs=True)
    options = {name:{'context':CONFIG['reference_context'] if CONFIG['methods'][name]['context']=='configured' else CONFIG['methods'][name]['context'],
                     'num_beams':CONFIG['methods'][name]['num_beams'],
                     'strategy':'adaptive_pause_chunks' if CONFIG['methods'][name].get('strategy')=='adaptive_pause_chunks' else 'whole',
                     'speech_gate':CONFIG['methods'][name].get('speech_gate',False),
                     'batch_size':CONFIG['methods'][name].get('batch_size',1),
                     'stream_cache':False,
                     'padding_ms':CONFIG['methods'][name].get('padding_ms')} for name in methods}
    save(folder/'configuration.json',{'input_manifest_sha256':sha(args.inputs/'inputs.json'),
        'worker_source_sha256':sha(ROOT/'scripts/runtime/qwen_asr_worker.py'), 'options':options,
        'method_definitions':{m:CONFIG['methods'][m] for m in methods},
        'pause_segmentation':CONFIG['pause_segmentation'],
        'selected_ids':[r['input_id'] for r in cases],'split':args.split})
    if any(CONFIG['methods'][m].get('strategy')=='adaptive_pause_chunks' for m in methods):
        assert stt['final_segmentation']==CONFIG['pause_segmentation']
    monitor = Resources(os.getpid(), folder/'resources.jsonl',CONFIG['resource_interval_seconds'])
    monitor.start()
    started = time.monotonic()
    bridge = None
    rows = []
    try:
        bridge = QwenBridge(stt,ROOT)
        model_load = time.monotonic()-started
        for case in cases:
            path = ROOT/case['audio']
            assert sha(path)==case['audio_sha256']
            audio = read_audio(path)
            for method in methods:
                began = time.monotonic()
                monitor.label = case['input_id']+'/'+method
                row = {k:case[k] for k in ('input_id','case_id','split','kind','reference','audio_sha256')}
                row.update(method=method,completed=False)
                try:
                    if CONFIG['methods'][method].get('strategy')=='pause_chunks':
                        pieces=[]
                        for left,right in pause_segments(audio,CONFIG['pause_segmentation']):
                            segment=bridge.transcribe(audio[left:right],None,True,decode_options=options[method])
                            pieces.append({'start_seconds':left/16000,'end_seconds':right/16000,**segment})
                        first=next((p for p in pieces if p['text']),pieces[0])
                        result={**first,'text':' '.join(p['text'].strip() for p in pieces if p['text'].strip()),'pieces':pieces,
                                'detected_languages':list(dict.fromkeys(p['detected_language'] for p in pieces if p['text']))}
                    else:
                        result = bridge.transcribe(audio,None,True,decode_options=options[method])
                    row.update(completed=True,hypothesis=result['text'],recognizer_output=result,
                        metrics=speech_metrics(case['reference'],result['text']),
                        semantic_checks=check_meaning(result['text'],case['semantic_checks']))
                except Exception as error:
                    row['error']={'type':type(error).__name__,'message':str(error)[:600]}
                row.update(seconds=time.monotonic()-began,audio_seconds=len(audio)/16000)
                row['real_time_factor']=row['seconds']/row['audio_seconds']
                row['resources']=monitor.summarize(began,time.monotonic())
                rows.append(row)
                with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'input':case['input_id'],'method':method,'wer':row.get('metrics',{}).get('wer'),
                                  'semantics':row.get('semantic_checks'),'error':row.get('error')}),flush=True)
    finally:
        if bridge:bridge.close()
        monitor.stop()
    summary={'expected_decodes':len(cases)*len(methods),'completed':sum(r['completed'] for r in rows),
             'model_load_seconds':model_load,'methods':{}}
    import statistics
    for method in methods:
        valid=[r for r in rows if r['method']==method and r['completed']]
        groups={}
        for kind in ('new_synthetic','captured','human_control'):
            group=[r for r in valid if r['kind']==kind]
            if not group:continue
            semantic=[v for r in group for v in r['semantic_checks'].values()]
            groups[kind]={'metrics':aggregate_metrics([r['metrics'] for r in group]),
                'semantic_passed':sum(semantic),'semantic_checks':len(semantic),
                'median_seconds':statistics.median(r['seconds'] for r in group),
                'median_real_time_factor':statistics.median(r['real_time_factor'] for r in group)}
        summary['methods'][method]=groups
    summary['resources']=monitor.summarize(started,time.monotonic())
    save(folder/'summary.json',summary)
    return int(summary['completed']!=summary['expected_decodes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','evaluate'],required=True)
    parser.add_argument('--inputs',type=Path,default=CONFIG['input_directory'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--split',choices=['development','validation','all'],default='development')
    parser.add_argument('--method',action='append',choices=list(CONFIG['methods']))
    parser.add_argument('--case',action='append')
    parser.add_argument('--capture-run',type=Path,action='append',help='Optional saved browser run containing the mixed-story capture')
    args=parser.parse_args()
    args.inputs=(ROOT/args.inputs).resolve()
    if args.stage=='prepare':prepare(args)
    else:
        if not args.output:parser.error('--output is required for evaluation')
        args.output=(ROOT/args.output).resolve()
        return evaluate(args)


if __name__=='__main__':raise SystemExit(main())
