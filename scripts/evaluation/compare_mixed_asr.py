"""Compare exact frozen waveforms and enforce ASR deployment gates."""
import argparse
import json
import statistics
from pathlib import Path

from voice_bridge.settings import ROOT
from scripts.evaluation.speech_metrics import aggregate_metrics
from evaluation.config import MIXED_ASR_EVALUATION as CONFIG


def read_rows(folder):
    return [json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]


def summarize(rows):
    semantic=[value for row in rows for value in row['semantic_checks'].values()]
    return {'recordings':len(rows),'metrics':aggregate_metrics([row['metrics'] for row in rows]),
            'meaning_checks_passed':sum(semantic),'meaning_checks':len(semantic),
            'median_processing_seconds':statistics.median(row['seconds'] for row in rows),
            'median_processing_rtf':statistics.median(row['real_time_factor'] for row in rows)}


def compare(candidate,baselines,baseline_method='configured_greedy',strict_quality=False):
    baseline={}
    # Explicit source order: later experiment wins; never select a row by its score.
    for folder in baselines:
        for row in read_rows(folder):
            if row['method']==baseline_method:
                baseline[row['input_id']]={**row,'baseline_experiment':folder.name}
    rows=read_rows(candidate)
    if not rows or any(not row['completed'] for row in rows):
        raise ValueError('Missing/failed candidate decodes cannot pass deployment')
    if len({row['input_id'] for row in rows})!=len(rows):
        raise ValueError('Compare one candidate method per waveform')
    for row in rows:
        old=baseline[row['input_id']]
        if not old['completed'] or old['audio_sha256']!=row['audio_sha256'] or old['reference']!=row['reference']:
            raise ValueError('Baseline/reference/waveform mismatch')
    cohorts={
        'development_stories':lambda r:r['kind']=='new_synthetic' and r['split']=='development',
        'validation_stories':lambda r:r['kind']=='new_synthetic' and r['split']=='validation',
        'captured_regressions':lambda r:r['kind']=='captured',
        'human_all':lambda r:r['kind']=='human_control',
        'human_en':lambda r:r['kind']=='human_control' and r['input_id'].startswith('fleurs-en_us'),
        'human_de':lambda r:r['kind']=='human_control' and r['input_id'].startswith('fleurs-de_de'),
    }
    for suffix in ('clean','noise20','noise10'):
        cohorts['validation_'+suffix]=lambda r,suffix=suffix:r['kind']=='new_synthetic' and r['split']=='validation' and r['input_id'].endswith('--'+suffix)
    report={'candidate':candidate.name,'waveforms':len(rows),'cohorts':{},'matched_pairs':[]}
    for name,predicate in cohorts.items():
        selected=[r for r in rows if predicate(r)]
        if selected:
            report['cohorts'][name]={'baseline':summarize([baseline[r['input_id']] for r in selected]),
                                     'candidate':summarize(selected)}
    for row in rows:
        old=baseline[row['input_id']]
        report['matched_pairs'].append({'input_id':row['input_id'],'kind':row['kind'],'split':row['split'],
            'audio_sha256':row['audio_sha256'],'baseline_experiment':old['baseline_experiment'],
            'reference':row['reference'],
            'baseline_metrics':old['metrics'],'candidate_metrics':row['metrics'],
            'baseline_text':old['hypothesis'],'candidate_text':row['hypothesis'],
            'baseline_meaning':old['semantic_checks'],'candidate_meaning':row['semantic_checks'],
            'baseline_seconds':old['seconds'],'candidate_seconds':row['seconds']})
    control=report['cohorts']['human_all'];validation=report['cohorts']['validation_stories'];captures=report['cohorts']['captured_regressions']
    report['deployment_gates']={
        'all_expected_waveforms':len(rows)==CONFIG['expected_guarded_waveforms'],
        'validation_wer_not_worse':validation['candidate']['metrics']['wer']<=validation['baseline']['metrics']['wer'],
        'validation_cer_not_worse':validation['candidate']['metrics']['cer']<=validation['baseline']['metrics']['cer'],
        'validation_meaning_not_worse':validation['candidate']['meaning_checks_passed']>=validation['baseline']['meaning_checks_passed'],
        'all_captured_facts_retained':captures['candidate']['meaning_checks_passed']==captures['candidate']['meaning_checks'],
        'human_wer_not_worse':control['candidate']['metrics']['wer']<=control['baseline']['metrics']['wer'],
        'no_individual_human_word_error_regression':all(row['metrics']['word_errors']<=baseline[row['input_id']]['metrics']['word_errors'] for row in rows if row['kind']=='human_control'),
    }
    report['passed']=all(report['deployment_gates'].values())
    if strict_quality:
        report['deployment_gates']['no_recording_word_or_character_regression']=all(
            row['metrics']['word_errors']<=baseline[row['input_id']]['metrics']['word_errors'] and
            row['metrics']['character_errors']<=baseline[row['input_id']]['metrics']['character_errors'] for row in rows)
        report['deployment_gates']['no_recording_fact_regression']=all(
            all(not passed or row['semantic_checks'].get(name,False) for name,passed in baseline[row['input_id']]['semantic_checks'].items())
            for row in rows)
        report['deployment_gates']['routing_unchanged']=all(
            all(row.get('recognizer_output',{}).get(key)==baseline[row['input_id']].get('recognizer_output',{}).get(key)
                for key in ('language','can_route','certain')) for row in rows)
        report['passed']=all(report['deployment_gates'].values())
    report['scope']='Exact WAV hashes; repeated captures/clean-noisy variants are not independent stories. Latencies are serial offline final-processing runs, not microphone delivery or cloud response time.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-method',default='configured_greedy')
    parser.add_argument('--strict-quality',action='store_true')
    args=parser.parse_args()
    output=ROOT/args.output
    if output.exists():raise RuntimeError('Preserve previous comparison; choose a new output')
    report=compare(ROOT/args.candidate,[ROOT/path for path in args.baseline],args.baseline_method,args.strict_quality)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'gates':report['deployment_gates']},indent=2))
    return int(not report['passed'])


if __name__=='__main__':raise SystemExit(main())
