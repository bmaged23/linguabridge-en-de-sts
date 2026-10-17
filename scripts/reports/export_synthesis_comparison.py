"""Export scored TTS results as tables and a local audio comparison page."""
import html
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'docs/tts-comparison'


def main():
    paths=sorted(FOLDER.glob('scored-*.json'))
    if not paths:raise RuntimeError('Run compare_tts.py judge first')
    rows=[];seen=set()
    for path in paths:
        for row in json.loads(path.read_text()):
            key=(row.get('scorer','qwen'),row['name'],row['variant'])
            if key not in seen:rows.append(row);seen.add(key)
    groups=[]
    for scorer,group,variant in sorted({(r.get('scorer','qwen'),r['group'],r['variant']) for r in rows}):
        cases=[r for r in rows if (r.get('scorer','qwen'),r['group'],r['variant'])==(scorer,group,variant)]
        resource_keys=('vram_mib','gpu_percent','cpu_machine_percent','pss_mib')
        resources={key:{'peak':max(r['resources'].get(key,{}).get('peak',0) for r in cases),
                        'mean_of_case_means':round(statistics.mean(r['resources'].get(key,{}).get('mean',0) for r in cases),2)} for key in resource_keys}
        groups.append({'scorer':scorer,'language':group,'variant':variant,'cases':len(cases),
            'wer_percent':round(100*sum(r['word_errors'] for r in cases)/sum(r['reference_words'] for r in cases),2),
            'cer_percent':round(100*sum(r['character_errors'] for r in cases)/sum(r['reference_characters'] for r in cases),2),
            'word_errors':sum(r['word_errors'] for r in cases),'reference_words':sum(r['reference_words'] for r in cases),
            'generation_seconds_mean':round(statistics.mean(r['generation_seconds'] for r in cases),3),
            'generation_seconds_min':min(r['generation_seconds'] for r in cases),
            'generation_seconds_max':max(r['generation_seconds'] for r in cases),
            'first_audio_seconds_mean':round(statistics.mean(r['first_available_audio_seconds'] for r in cases),3),
            'audio_seconds_mean':round(statistics.mean(r['audio_seconds'] for r in cases),3),
            'rtf_total':round(sum(r['generation_seconds'] for r in cases)/sum(r['audio_seconds'] for r in cases),3),
            'resources':resources})
    report={'date':'2026-10-17','hardware':'GTX 1660 Ti 6 GiB; i7-9750H, 12 logical CPUs; WSL 7.66 GiB RAM',
        'asr':'Whisper large-v3-turbo int8_float32, beam 3, multilingual auto; Qwen3-ASR-0.6B FP32 eager auto as cross-check, fixed verbatim hint. No reference transcripts supplied to either recognizer.',
        'normalization':'Unicode casefold, word tokens, punctuation removed; CER includes normalized word spaces; no number expansion',
        'aggregation':'Micro-average edit counts divided by total reference words/characters; speed ratio uses total generation / total audio duration',
        'limitations':['12 authored sentences, 4 per language group; one seeded run per case; synthetic content-consistency smoke test, not a general benchmark.',
          'ASR-mediated WER/CER confounds recognition and synthesis and can miss mispronunciation. Both recognizers score all variants. Whisper is independent of the Qwen family; Qwen ASR provides a second view.',
          'Gold spans are manually supplied evaluation boundaries, not a tested automatic LLM segmentation implementation. Whisper dropped whole English sections from several correct hybrid outputs; mixed-language WER is especially ASR-dependent and is not a reliable standalone pronunciation ranking.',
          'The reference voice is synthetic Kokoro af_heart; no claim of cloning a real person or measured speaker similarity.',
          'All timing measurements exclude loading and one warmup. Full waveform calls: first audio means first completed segment, not audio-token streaming.',
          'Models run sequentially without STT/LLM/browser playback. CPU/RAM refer to the generation process; whole-device GPU sampling can miss brief peaks.',
          'Both Kokoro and Piper share one loaded process. Their resources cannot be attributed as independent resident footprints.',
          'RTF below 1 means faster than playback. Naturalness, accent authenticity, and speaker consistency require listening.'],
        'summary':groups,'cases':rows}
    for name,key in [('compatibility.json','precision_compatibility'),('co-residency.json','co_residency'),('speaker-similarity.json','speaker_similarity')]:
        if (FOLDER/name).exists():report[key]=json.loads((FOLDER/name).read_text())
    (FOLDER/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# TTS accuracy and computation comparison','','Measured 2026-10-17 on GTX 1660 Ti 6 GiB. Same text and ASR scoring across variants. English/German each have four sentences; mixed has four (two in each switching direction).','',
      '| Scorer | Language | Voice / strategy | WER | CER | Generation mean | First segment mean | Audio mean | RTF |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in groups:lines.append(f"| {r['scorer']} | {r['language']} | {r['variant']} | {r['wer_percent']:.2f}% | {r['cer_percent']:.2f}% | {r['generation_seconds_mean']:.3f} s | {r['first_audio_seconds_mean']:.3f} s | {r['audio_seconds_mean']:.3f} s | {r['rtf_total']:.3f} |")
    lines+=['','## Computation','','CPU percentages divide by all 12 logical CPUs. RAM uses proportional set size (PSS). GPU metrics are whole-device. Short inference runs yield few samples.','',
      '| Language / strategy | Peak VRAM | Peak RAM PSS | Mean CPU | Peak CPU | Mean GPU | Peak GPU |',
      '| --- | --- | --- | --- | --- | --- | --- |']
    for r in groups:
        if r['scorer']!='whisper':continue
        a=r['resources'];lines.append(f"| {r['language']} / {r['variant']} | {a['vram_mib']['peak']:.0f} MiB | {a['pss_mib']['peak']:.0f} MiB | {a['cpu_machine_percent']['mean_of_case_means']:.1f}% | {a['cpu_machine_percent']['peak']:.1f}% | {a['gpu_percent']['mean_of_case_means']:.1f}% | {a['gpu_percent']['peak']:.1f}% |")
    if 'speaker_similarity' in report:
        lines+=['','## One voice across languages','','Internal diagnostic: cosine similarity to the synthetic Kokoro reference using Qwen TTS own speaker encoder. This is not an independent speaker-verification or listening score.','', '| Language | Strategy | Mean cosine to reference |','| --- | --- | --- |']
        items=report['speaker_similarity']['cases']
        for group,variant in sorted({(r['group'],r['variant']) for r in items}):
            values=[r['cosine_to_reference'] for r in items if (r['group'],r['variant'])==(group,variant)]
            lines.append(f'| {group} | {variant} | {statistics.mean(values):.3f} |')
    if 'co_residency' in report:
        fit=report['co_residency']
        lines+=['','## GPU compatibility','',f"FP16 failed during warmup with invalid sampling probabilities. FP32 generated all test outputs. With FP32 Qwen TTS resident, loading the deployed FP32 Qwen ASR **{fit['asr_initialization']}**.",'', '[Full co-residency measurement](../tts-comparison/co-residency.json).']
        if 'real_inference_check' in fit:
            lines+=['',f"Running actual speech inference with both models resident **{fit['real_inference_check']}**."]
        if fit['asr_initialization']=='failed' or fit.get('real_inference_check')=='failed':
            lines+=['','The tested Qwen TTS runtime is not a drop-in replacement for this 6 GiB deployment. Keep the existing fast voices and fix language-based segment routing. Quantization, a different runtime, or offloading would require a separate benchmark.']
        elif fit.get('real_inference_check')=='passed':
            lines+=['',f"Both models loaded with {fit['both_loaded']['used_mib']:.0f} / {fit['both_loaded']['total_mib']:.0f} MiB reported by nvidia-smi. The real ASR check took {fit['asr_real_audio_seconds']:.3f} s, and the short TTS reply took {fit['tts_real_audio_generation_seconds']:.3f} s to generate {fit['tts_real_audio_seconds']:.3f} s of audio. This is a sequential resident-model check, not a concurrent stress test; near-capacity device usage is not proof that every request fits without paging. The measured standalone TTS speed is already slower than playback, so the present Kokoro/Piper runtime remains preferable for live response speed."]
        lines+=['',f"The released model is named 0.6B. Counting the loaded PyTorch modules gives {sum(fit['components'].values()):,} parameters in talker plus speaker encoder, and {fit['audio_tokenizer_parameters']:,} in the separate audio tokenizer. The release size label does not describe the entire inference pipeline."]
    lines+=['','## Mixed-language scoring caveat','','Whisper omitted complete English sections from multiple correctly split hybrid samples. For example, it transcribed only the German middle of an English–German–English sentence. Compare both scorer tables and listen to the audio; do not interpret that mixed-language score as proof that the TTS omitted those words.','', '## Interpretation and limits','']+['- '+s for s in report['limitations']]
    lines+=['','[Listen to the comparison](../tts-comparison/index.html) · [Raw scores, transcripts and resources](../tts-comparison/report.json) · [Fixed test sentences](../../scripts/evaluation/speech_synthesis_test_cases.json)','',
      'Qwen source and API: [official repository](https://github.com/QwenLM/Qwen3-TTS). Qwen TTS checkpoint revision `5d83992436eae1d760afd27aff78a71d676296fc`. See `tts-comparison/qwen-settings.json` and `compatibility.json` for local precision and compatibility details.']
    (ROOT/'docs/reports/speech-synthesis-comparison.md').write_text('\n'.join(lines)+'\n')
    cards=[]
    for name in dict.fromkeys(r['name'] for r in rows):
        cases=[r for r in rows if r['name']==name]
        variants=[]
        for variant in dict.fromkeys(r['variant'] for r in cases):
            scores=[r for r in cases if r['variant']==variant]
            r=scores[0]
            score_text=' · '.join(f"{x['scorer']}: WER {100*x['wer']:.1f}%, CER {100*x['cer']:.1f}%" for x in scores)
            transcripts=''.join(f"<p>{html.escape(x['scorer'])}: {html.escape(x['hypothesis'])}</p>" for x in scores)
            variants.append(f'''<div class="variant"><h3>{html.escape(r['variant'])}</h3><p>{score_text} · generation {r['generation_seconds']:.2f}s</p><audio controls preload="none" src="{html.escape(r['audio'])}"></audio><details><summary>ASR transcript</summary>{transcripts}</details></div>''')
        cards.append(f'''<section><h2>{html.escape(name.replace('_',' '))}</h2><p class="reference">{html.escape(cases[0]['reference'])}</p><div class="variants">{''.join(variants)}</div></section>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bridge · TTS comparison</title><style>body{font-family:system-ui,sans-serif;background:#f4f6f1;color:#20382c;margin:0;padding:32px}main{max-width:1200px;margin:auto}h1{letter-spacing:-1px}p{line-height:1.6;color:#536657}section{background:white;border:1px solid #dce6d7;border-radius:18px;padding:24px;margin:24px 0}.reference{font-size:18px}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px}.variant{background:#f5f8f1;padding:16px;border-radius:12px;min-width:0}audio{width:100%}h3{font-size:14px;overflow-wrap:anywhere}summary{cursor:pointer;margin-top:14px}a{color:#245b46}</style><main><h1>One sentence. Different voices.</h1><p>English, German, and mixed-language TTS · GTX 1660 Ti · 17 October 2026</p><p>Compare the original text, audio, and automated transcription. WER/CER are a recognition-based proxy for content accuracy—not a human pronunciation or naturalness score. The cloned voice uses one synthetic Kokoro reference in both languages.</p><p><a href="../reports/speech-synthesis-comparison.md">Full measurement report</a> · <a href="report.json">Raw results</a></p>'''+''.join(cards)+'''</main><script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''
    (FOLDER/'index.html').write_text(page)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
