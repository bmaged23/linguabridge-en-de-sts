"""Aggregate complete story runs with explicit scoring and timing scopes."""
import json,statistics
from pathlib import Path
from voice_bridge.settings import ROOT
from scripts.evaluation.evaluate_mixed_language_asr import normalized,distance


def score(rows):
 words=chars=word_errors=char_errors=0
 for reference,text in rows:
  a=normalized(reference);b=normalized(text);x=' '.join(a);y=' '.join(b)
  words+=len(a);chars+=len(x);word_errors+=distance(a,b);char_errors+=distance(x,y)
 return {'n':len(rows),'wer':word_errors/max(words,1),'cer':char_errors/max(chars,1)}


def main():
 folders=[ROOT/'logs/advanced-story'/n for n in ('story-20261017','story-extended-20261017')]
 reports=[(p,json.loads((p/'results.json').read_text())) for p in folders if (p/'results.json').exists()]
 samples=[s for _,r in reports for s in r['samples']];lookup={s['case']['id']:s for s in samples}
 lines=['# Advanced continuous-story audit','','Every input is a single connected story WAV. Conditions include English, German, repeated language switches, noise, low volume, pauses, repeated inputs, bilingual retelling requests, and longer stories. These are synthetic diagnostic runs of one related narrative, not independent human speakers or a public benchmark.','',f'{len(samples)} pipeline runs; {sum(s["case"]["kind"]=="mixed" for s in samples)} mixed-language runs; {sum(s["audio_chunks"] for s in samples)} generated browser audio chunks; {sum(s["completed_playback_chunks"] for s in samples)} completed playback acknowledgments.','', '## Saved outputs','']
 for folder,data in reports:
  lines.append(f'- [{folder.name}: every sample, raw STT/LLM tokens, native and combined TTS audio, resource measurements]({folder.name}/index.html)')
 lines+=['','## Recognition comparison','','WER/CER below normalize case and punctuation but retain spelling and number-format differences. Scores are weighted by reference length. Candidate models receive the same PCM16 browser-captured WAVs; the live application originally receives float32 samples. Candidates decode every captured segment, including segments the application canceled. The live application therefore measures both recognition and turn-handling losses.','','| Configuration | Mixed N | Mixed WER | Mixed CER | All N | All WER | All CER |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 configurations={'Live Qwen progressive pipeline':samples}
 for folder,_ in reports:
  for path in sorted(folder.glob('candidate-*.json')):
   candidate=json.loads(path.read_text())
   if not isinstance(candidate,dict) or 'samples' not in candidate:continue
   if 'qwen-fresh' in path.name:name='qwen-fresh'
   elif 'qwen-replay' in path.name:name='qwen-replay'
   else:
    architecture='large-v3' if 'large-v3' in path.name else 'turbo'
    name=architecture+' ('+str(candidate.get('whisper_chunk_seconds','unknown'))+'s multilingual)'
   configurations.setdefault(name,[]).extend(candidate['samples'])
 aggregate={}
 for name,rows in configurations.items():
  pairs=[(r['case']['reference'],r['stt_final']) if 'stt_final' in r else (r['reference'],r['text']) for r in rows]
  mixed=[p for r,p in zip(rows,pairs) if (r['case']['kind'] if 'stt_final' in r else lookup[r['case']]['case']['kind'])=='mixed']
  total=score(pairs);part=score(mixed);aggregate[name]={'all':total,'mixed':part}
  lines.append(f'| {name} | {part["n"]} | {part["wer"]:.1%} | {part["cer"]:.1%} | {total["n"]} | {total["wer"]:.1%} | {total["cer"]:.1%} |')
 complete_mixed={s['case']['id'] for s in samples if s['case']['kind']=='mixed' and len([e for e in s['pipeline_trace'] if e['event']=='stt_input'])==1 and len([e for e in s['pipeline_trace'] if e['event']=='stt_final'])==1}
 lines+=['','### Mixed stories with one completed captured segment','','This subset removes cases with canceled or split final transcriptions, making decoder comparisons easier to interpret.','','| Configuration | N | WER | CER |','| --- | ---: | ---: | ---: |']
 complete_scores={}
 for name,rows in configurations.items():
  selected=[r for r in rows if (r['case']['id'] if 'stt_final' in r else r['case']) in complete_mixed]
  pairs=[(r['case']['reference'],r['stt_final']) if 'stt_final' in r else (r['reference'],r['text']) for r in selected];values=score(pairs)
  complete_scores[name]=values
  lines.append(f'| {name} | {values["n"]} | {values["wer"]:.1%} | {values["cer"]:.1%} |')
 lines+=['','`qwen-replay` is deliberately restricted to cases failing in the live or fresh-final run, plus clean controls, so its aggregate is not directly comparable to the full-suite aggregates. It replays cumulative previews and prefix reuse on the same frozen WAVs. Fresh Qwen is a full independent final decode. Whisper candidates use automatic per-window language detection (`multilingual=True`); 30-second standard and 5-second experimental windows are identified separately and a transcription task. Their reported language label is the initial window, not a complete language segmentation.','','## Interpretation\n\nOn the identical complete-capture mixed subset, the live progressive path has lower error than independent fresh finals. The better overall fresh-final score partly comes from recovering segments the application canceled. Neither fresh-final decoding nor the tested Whisper configurations is a demonstrated drop-in fix. Preserve user speech and enforce voice metadata before attributing all failures to model choice.\n\n## Timing and resource interpretation','','The first part of the base run encountered a GPU held in P8 at 300 MHz under load. It later reached P0 at roughly 1,650–1,860 MHz. Do not pool those final-STT timings as normal GPU performance. Per-sample stage clocks and raw power/clock samples are saved where available. GPU clocks normally also drop during idle/playback; the reviewer correlates clock state specifically with final STT.','','| Local candidate in isolation | Samples | Median decoding seconds | Peak sampled RAM MiB | Peak sampled whole-GPU VRAM MiB |','| --- | ---: | ---: | ---: | ---: |']
 for name,rows in configurations.items():
  if name.startswith('Live'):continue
  ram=[(r['resources'].get('ram_tree_rss_mb') or {}).get('peak') for r in rows];vram=[(r['resources'].get('gpu_vram_mb') or {}).get('peak') for r in rows]
  ram=[v for v in ram if v is not None];vram=[v for v in vram if v is not None]
  lines.append(f'| {name} | {len(rows)} | {statistics.median(r["decode_seconds"] for r in rows):.3f} | {max(ram) if ram else "unobserved"} | {max(vram) if vram else "unobserved"} |')
 lines+=['','Candidate decoding excludes microphone arrival, VAD, cloud generation and playback. Replay decoding includes all its preview computation. Live end-to-end timings include the recorded input duration and playback. Local pipeline RAM covers the server and speech workers; system RAM is the WSL guest rather than Windows host; cloud LLM RAM/VRAM are unavailable. GPU memory/utilization cover the whole device. Sampling can miss short peaks. RSS sums may count shared pages more than once. See each sample for CPU percentages, per-model RAM, GPU utilization, individual STT/LLM/TTS timings, and full completion time.','','## Findings requiring attention','']
 counts={}
 for folder,_ in reports:
  p=folder/'review.json'
  if p.exists():
   for k,n in json.loads(p.read_text())['findings_counts'].items():counts[k]=counts.get(k,0)+n
 for text,count in sorted(counts.items(),key=lambda x:-x[1]):lines.append(f'- {count} runs: {text}.')
 marker_results=[]
 for folder,_ in reports:
  path=folder/'marker-recovery-diagnostic.json'
  if path.exists():marker_results+=json.loads(path.read_text())
 if marker_results:
  times=[r['seconds'] for r in marker_results];valid=sum(bool(r.get('valid_schema') and r.get('nonwhitespace_preservation')) for r in marker_results)
  lines+=['','### Voice-marker recovery diagnostic','',f'A separate labeling-only LLM pass was tested on {len(marker_results)} actual unmarked bilingual replies. {valid}/{len(marker_results)} retained every non-whitespace character and returned valid English/German spans. Observed request time: {min(times):.3f}–{max(times):.3f} seconds. The inspected language boundaries matched the bilingual sentences. This prototype was not deployed and does not correct factual content.', '', '[Raw labeling outputs](story-extended-20261017/marker-recovery-diagnostic.json).']
 lines+=['','These findings separate speech loss/translation, canceled user segments, summary inferences and audio disagreement. Semantic inference flags are transparent review heuristics, not a certified semantic score. The same story is repeated while application memory persists, which can conceal missing current STT input; full LLM prompt history is retained in the traces.','','TTS WAV integrity is checked and actual WAV bytes are re-recognized; identical bytes may share an audit result. Automatic ASR agreement cannot replace human listening. Browser playback uses real-time WebAudio with a silent output sink, so physical speaker/microphone behavior is not verified.','','## Reproduction','','See [test instructions](../ADVANCED_STORY_TESTING.md). The normal conversation database remains separate from the test databases. No credentials or user conversation history are exported.','']
 dest=ROOT/'docs/advanced-story';dest.mkdir(exist_ok=True);(dest/'SUMMARY.md').write_text('\n'.join(lines)+'\n');(dest/'summary.json').write_text(json.dumps({'samples':len(samples),'recognition':aggregate,'complete_mixed_recognition':complete_scores,'findings':counts},ensure_ascii=False,indent=2)+'\n')
 print('\n'.join(lines[:25]));print('Summary:',dest/'SUMMARY.md')
if __name__=='__main__':main()
