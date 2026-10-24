import json
from scripts.evaluation import compare_mixed_asr as comparison
from scripts.evaluation.speech_metrics import speech_metrics


def test_average_improvement_cannot_hide_a_human_regression(tmp_path,monkeypatch):
    monkeypatch.setitem(comparison.CONFIG,'expected_guarded_waveforms',4)
    old=tmp_path/'baseline';new=tmp_path/'candidate';old.mkdir();new.mkdir()
    def row(identity,kind,split,text,method):
        return {'input_id':identity,'kind':kind,'split':split,'completed':True,
                'reference':'one two','hypothesis':text,'audio_sha256':identity,
                'metrics':speech_metrics('one two',text),'semantic_checks':{'fact':True},
                'seconds':1,'real_time_factor':.1,'method':method}
    specs=[('val','new_synthetic','validation','one too','one two'),
           ('capture','captured','development','one two','one two'),
           ('fleurs-en_us-a','human_control','control','one two','one too'),
           ('fleurs-de_de-b','human_control','control','one too','one two')]
    baseline=[row(i,k,s,a,'configured_greedy') for i,k,s,a,b in specs]
    candidate=[row(i,k,s,b,'speech_gated_pause_beam3') for i,k,s,a,b in specs]
    (old/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in baseline))
    (new/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidate))
    report=comparison.compare(new,[old])
    assert report['deployment_gates']['human_wer_not_worse']
    assert not report['deployment_gates']['no_individual_human_word_error_regression']
    assert not report['passed']
