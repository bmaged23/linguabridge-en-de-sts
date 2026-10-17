"""Synthetic complex/code-switch ASR comparison; no app state or cloud calls."""
import json
import re
import time
import wave
from pathlib import Path

import numpy as np

CASES = [
    ('english_complex', [('en', 'Although my appointment was originally scheduled for Thursday, I need to move it to Friday afternoon because my connecting train has been cancelled.')]),
    ('german_complex', [('de', 'Obwohl mein Termin ursprünglich für Donnerstag geplant war, muss ich ihn auf Freitagnachmittag verschieben, weil mein Anschlusszug ausgefallen ist.')]),
    ('english_numbers', [('en', 'Please reserve two tickets to Berlin for the twenty third of November, but do not confirm the booking until I have checked the total price.')]),
    ('german_numbers', [('de', 'Bitte reservieren Sie zwei Fahrkarten nach Berlin für den dreiundzwanzigsten November, aber bestätigen Sie die Buchung erst, nachdem ich den Gesamtpreis geprüft habe.')]),
    ('en_de_en_travel', [('en', 'I would like to change my reservation because'), ('de', 'mein Zug hat Verspätung und ich verpasse den Anschluss'), ('en', 'so could you find another connection that arrives before midnight?')]),
    ('de_en_de_travel', [('de', 'Könnten Sie bitte prüfen, ob'), ('en', 'the connecting flight has been cancelled or simply delayed'), ('de', 'damit ich entscheiden kann, ob ich heute noch ein Hotel buchen muss?')]),
    ('en_de_en_learning', [('en', 'Please explain the difference between'), ('de', 'obwohl ich müde bin und weil ich müde bin'), ('en', 'and then give me an example that I could use at work tomorrow.')]),
    ('de_en_de_work', [('de', 'Meine Kollegin hat gesagt'), ('en', 'we should postpone the meeting until everyone has reviewed the updated contract'), ('de', 'aber ich möchte zuerst wissen, ob der Kunde damit einverstanden ist.')]),
]


def normalized(text):
    return re.findall(r'\w+', text.casefold())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        following = [i]
        for j, y in enumerate(b, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = following
    return row[-1]


def metrics(reference, hypothesis):
    words, predicted = normalized(reference), normalized(hypothesis)
    chars, output = ' '.join(words), ' '.join(predicted)
    return {'wer': round(distance(words, predicted) / max(len(words), 1), 4),
            'cer': round(distance(chars, output) / max(len(chars), 1), 4)}


def main():
    from voice_bridge.settings import ROOT, load_settings
    from voice_bridge import workers
    config = load_settings()
    folder = ROOT / 'logs' / 'codeswitch-check'
    folder.mkdir(parents=True, exist_ok=True)
    workers.init_tts(config['tts'], ROOT)
    audio_cases = []
    for name, spans in CASES:
        pieces = []
        for language, text in spans:
            pcm, rate = workers.synthesize(text, language)
            samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
            samples = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype(np.float32)
            pieces.append(samples)
            # Small phrase boundary, well below the endpoint silence threshold.
            pieces.append(np.zeros(1920, dtype=np.float32))
        audio = np.concatenate(pieces)
        with wave.open(str(folder / f'{name}.wav'), 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
        audio_cases.append((name, spans, audio))
        print(f'Generated {name}: {len(audio)/16000:.1f}s', flush=True)
    # Keep the comparison baseline fixed after the app deploys another recognizer.
    settings = dict(config['stt'], provider='faster_whisper', model_path='models/whisper-large-v3-turbo')
    workers.init_stt(settings, ROOT)
    report = {'method': 'Synthetic Kokoro English / Piper German segments joined at 16 kHz with 120 ms pauses. Different voices at language switches. Not a human speech benchmark. Case/punctuation normalized; numbers are not expanded.', 'cases': []}
    for name, spans, audio in audio_cases:
        reference = ' '.join(text for _, text in spans)
        item = {'name': name, 'reference': reference, 'seconds': round(len(audio)/16000, 2), 'results': {}}
        for strategy in ('current_previous_en', 'current_previous_de', 'auto', 'auto_multilingual'):
            started = time.monotonic()
            if strategy.startswith('current'):
                result = workers.transcribe(audio, previous=strategy[-2:])
                text = result['text']
            else:
                segments, info = workers._model.transcribe(audio, language=None, task='transcribe',
                    multilingual=strategy == 'auto_multilingual', beam_size=settings['beam_size'],
                    vad_filter=False, condition_on_previous_text=False,
                    no_speech_threshold=settings['no_speech_threshold'], log_prob_threshold=settings['log_prob_threshold'])
                text = ' '.join(segment.text.strip() for segment in segments)
                result = {'text': text, 'language': info.language, 'probability': info.language_probability}
            result.update(metrics(reference, text), elapsed_seconds=round(time.monotonic()-started, 2))
            item['results'][strategy] = result
            print(f'{name} | {strategy} | WER {result["wer"]:.1%} CER {result["cer"]:.1%} | {text}', flush=True)
        report['cases'].append(item)
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Report: {folder / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
