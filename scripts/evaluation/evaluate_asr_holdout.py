"""Freeze four fresh synthetic sentences after selecting the bilingual ASR hint."""
import json
import wave
import numpy as np
from voice_bridge.settings import ROOT, load_settings
from voice_bridge import workers
from scripts.evaluation.evaluate_mixed_language_asr import metrics

CASES = [
    ('en_de_en_booking', [('en', 'Before we confirm the booking, please check whether'), ('de', 'die Rückfahrt am Freitag wirklich im Preis enthalten ist'), ('en', 'and tell me if I can cancel without paying an additional fee.')]),
    ('de_en_de_grammar', [('de', 'Ich möchte nicht, dass du automatisch übersetzt, sondern'), ('en', 'explain the grammar step by step and correct only the mistakes that change the meaning'), ('de', 'damit ich die Antwort selbst formulieren kann.')]),
    ('en_de_en_invoice', [('en', 'My colleague Anna said'), ('de', 'wir dürfen die Rechnung erst bezahlen, wenn die neue Kundennummer bestätigt wurde'), ('en', 'but the supplier is asking for a reply before tomorrow morning.')]),
    ('de_en_de_hotel', [('de', 'Bitte buche noch kein Zimmer, denn'), ('en', 'I may arrive on Tuesday instead of Monday and I need a quiet room near the station'), ('de', 'und ich möchte vorher den endgültigen Preis wissen.')]),
]


def main():
    config = load_settings()
    folder = ROOT / 'logs' / 'codeswitch-holdout'
    folder.mkdir(parents=True, exist_ok=True)
    workers.init_tts(config['tts'], ROOT)
    cases = []
    for name, spans in CASES:
        pieces = []
        for language, text in spans:
            pcm, rate = workers.synthesize(text, language)
            audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
            pieces.extend([np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32), np.zeros(1920, np.float32)])
        samples = np.concatenate(pieces)
        with wave.open(str(folder / f'{name}.wav'), 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
        cases.append((name, ' '.join(text for _, text in spans), samples))
        print(f'Generated {name}: {len(samples)/16000:.2f}s', flush=True)
    workers.init_stt(dict(config['stt'], provider='faster_whisper', model_path='models/whisper-large-v3-turbo'), ROOT)
    report = {'method': 'Four synthetic held-out utterances, frozen after ASR context selection. Kokoro English and Piper German joined at 16 kHz with 120 ms pauses. Voice changes at language switches; not a human speech benchmark.', 'cases': []}
    for name, reference, samples in cases:
        import time
        started = time.monotonic()
        segments, info = workers._model.transcribe(samples, language=None, task='transcribe',
            beam_size=config['stt']['beam_size'], vad_filter=False, condition_on_previous_text=False,
            no_speech_threshold=config['stt']['no_speech_threshold'], log_prob_threshold=config['stt']['log_prob_threshold'])
        text = ' '.join(segment.text.strip() for segment in segments)
        result = {'text': text, 'language': info.language, **metrics(reference, text), 'elapsed_seconds': round(time.monotonic()-started, 3)}
        report['cases'].append({'name': name, 'reference': reference, 'seconds': round(len(samples)/16000, 3), 'results': {'auto': result}})
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'{name} | WER {result["wer"]:.1%} CER {result["cer"]:.1%} | {text}', flush=True)


if __name__ == '__main__':
    main()
