"""Review speed profiles against frozen accuracy before selecting runtime defaults."""
import argparse
import json
import statistics
from pathlib import Path
from voice_bridge.settings import ROOT
from scripts.evaluation.speech_metrics import normalized_words


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=ROOT/args.replay;output=ROOT/args.output
    if output.exists():raise RuntimeError('Preserve prior speed assessment')
    rows=[json.loads(l) for l in (folder/'results.jsonl').read_text().splitlines()]
    reference={r['input_id']:r for r in rows if r['profile']=='reference'}
    frozen={r['input_id']:r for r in [json.loads(l) for l in
        (ROOT/'.evaluation-local/mixed-asr-guarded-v3/results.jsonl').read_text().splitlines()]}
    profiles={};checks=[]
    reference_valid=all(normalized_words(r['final']['text'])==normalized_words(frozen[i]['hypothesis'])
        and all(r['final'].get(k)==frozen[i]['recognizer_output'].get(k) for k in ('language','can_route','certain'))
        for i,r in reference.items())
    for profile in sorted({r['profile'] for r in rows}):
        values=[r for r in rows if r['profile']==profile]
        complete=len(values)==12 and {r['input_id'] for r in values}==set(reference)
        quality=True
        for row in values:
            old=reference[row['input_id']]
            same=normalized_words(row['final']['text'])==normalized_words(old['final']['text'])
            routing=all(row['final'].get(k)==old['final'].get(k) for k in ('language','can_route','certain'))
            meaning=all(not v or row['semantic_checks'].get(k,False) for k,v in old['semantic_checks'].items())
            checks.append({'input_id':row['input_id'],'profile':profile,'text_identical':same,'routing_identical':routing,'meaning_retained':meaning})
            quality=quality and same and routing and meaning
        profiles[profile]={'recordings':len(values),'coverage_complete':complete,'accuracy_unchanged':quality,
            'median_final_seconds':statistics.median(r['final_seconds'] for r in values),
            'median_total_compute_seconds':statistics.median(r['total_compute_seconds'] for r in values),
            'final_model_calls':sum(r['final']['model_decode_calls'] for r in values),
            'final_cache_hits':sum(r['final']['cached_segment_hits'] for r in values),
            'batch_oom_fallbacks':sum(r['final'].get('batch_oom_fallbacks',0)+sum(p['result'].get('batch_oom_fallbacks',0) for p in r['previews']) for r in values),
            'mixed':{},'human':{}}
        for group,kind in [('mixed',False),('human',True)]:
            selected=[r for r in values if (r['kind']=='human_control')==kind]
            profiles[profile][group]={'recordings':len(selected),
                'median_final_seconds':statistics.median(r['final_seconds'] for r in selected),
                'median_total_compute_seconds':statistics.median(r['total_compute_seconds'] for r in selected)}
    ready=reference_valid and all(p['coverage_complete'] and p['accuracy_unchanged'] for p in profiles.values())
    report={'reference_matches_frozen_accuracy':reference_valid,'all_profiles_accuracy_passed':ready,'profiles':profiles,'checks':checks,
        'scope':'Offline cumulative replay. Final timing excludes actual arrival/worker queue/endpointing/cloud/TTS; total compute is not elapsed conversation time.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    return int(not ready)


if __name__=='__main__':raise SystemExit(main())
