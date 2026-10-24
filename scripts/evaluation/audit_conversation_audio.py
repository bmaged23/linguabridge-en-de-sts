"""Independent Whisper audit of native audio generated in a recorded browser run.

Known-language chunk scores measure agreement with the text sent to synthesis.
They cannot certify that the LLM chose the right text or language marker.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel
from voice_bridge.settings import ROOT
from config import EVALUATION_CONFIG as CONFIG
from scripts.evaluation.audio_tools import read_audio, signal_metrics
from scripts.evaluation.speech_metrics import speech_metrics, aggregate_metrics
from scripts.story_tests.resource_monitor import Resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / args.run).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'audio.jsonl').exists():
        raise RuntimeError('Preserve previous results; choose a new output folder')
    settings = CONFIG['independent_asr']
    monitor = Resources(os.getpid(), output / 'resources.jsonl', CONFIG['resource_sample_interval_seconds'])
    monitor.start()
    started = time.monotonic()
    rows = []
    try:
        model = WhisperModel(str(ROOT / settings['model_path']), device=settings['device'],
                             compute_type=settings['compute_type'], cpu_threads=settings['cpu_threads'],
                             local_files_only=True)
        loaded = time.monotonic() - started
        with (output / 'audio.jsonl').open('w') as stream:
            for trace in sorted((source / 'pipeline-traces').glob('*/events.jsonl')):
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                complete = {e['id'] for e in events if e['event'] == 'playback_ack' and e.get('complete')}
                for event in events:
                    if event['event'] != 'tts_audio':
                        continue
                    path = trace.parent / event['audio']
                    row = {'session': trace.parent.name, 'id': event['id'], 'turn': event['turn'],
                           'reference': event['text'], 'language': event['language'],
                           'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'completed_playback': event['id'] in complete}
                    monitor.label = trace.parent.name + '/' + event['id']
                    began = time.monotonic()
                    try:
                        audio = read_audio(path)
                        row['signal'] = signal_metrics(audio, 16000, CONFIG)
                        segments, info = model.transcribe(audio, language=event['language'],
                            beam_size=settings['beam_size'], task='transcribe',
                            condition_on_previous_text=False, vad_filter=False, multilingual=False)
                        row['segments'] = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
                        row['hypothesis'] = ' '.join(s['text'].strip() for s in row['segments'])
                        row['metrics'] = speech_metrics(row['reference'], row['hypothesis'])
                        row['decode_seconds'] = time.monotonic() - began
                        row['real_time_factor'] = row['decode_seconds'] / (len(audio) / 16000)
                    except Exception as error:
                        row['error'] = {'type': type(error).__name__, 'message': str(error)[:400]}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({'id': monitor.label, 'wer': row.get('metrics', {}).get('wer'),
                                      'error': row.get('error')}), flush=True)
    finally:
        monitor.stop()
    valid = [r for r in rows if 'metrics' in r]
    summary = {'generated_chunks': len(rows), 'scored': len(valid), 'failed': len(rows) - len(valid),
               'model_load_seconds': loaded, 'groups': {
                   language: aggregate_metrics([r for r in valid if r['language'] == language])
                   for language in ('en', 'de')},
               'resources': monitor.summarize(started, time.monotonic()),
               'scope': 'Known-language native generated chunks, including cancelled/unplayed chunks labelled separately. '
                        'Text/audio agreement proxy, not a human listening score or a check of LLM marker correctness.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
