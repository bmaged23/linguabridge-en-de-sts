"""Evaluate Qwen in the isolated .venv-asr against existing Whisper test WAVs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'asr-evaluation'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-ASR-0.6B')
    parser.add_argument('--suite', choices=('codeswitch-check', 'codeswitch-holdout'), default='codeswitch-check')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--progressive', action='store_true', help='Replay mixed WAVs in real time with replaceable partial transcripts; not native vLLM streaming')
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--context', default='', help='Fixed ASR context; never use a reference transcript')
    parser.add_argument('--rollback-tokens', type=int, default=None,
        help='Experimental Transformers prefix reuse: keep earlier decoded tokens, allowing this many trailing tokens to change')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if args.rollback_tokens is not None and (args.rollback_tokens < 1 or not args.progressive):
        parser.error('--rollback-tokens requires --progressive and a positive count')
    import torch
    from huggingface_hub import model_info, snapshot_download
    from qwen_asr import Qwen3ASRModel
    from scripts.evaluation.evaluate_mixed_language_asr import metrics

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local GPU comparison')
    # Pin each downloaded checkpoint to its resolved commit in the report.
    revision = model_info(args.model).sha
    path = snapshot_download(args.model, revision=revision)
    started = time.monotonic()
    model = Qwen3ASRModel.from_pretrained(
        path, dtype=getattr(torch, args.dtype), device_map='cuda:0',
        attn_implementation='eager', max_inference_batch_size=1, max_new_tokens=256,
    )
    folder = ROOT / 'logs' / args.suite
    baseline = json.loads((folder / 'report.json').read_text())
    report = {
        'model': args.model, 'revision': revision, 'gpu': torch.cuda.get_device_name(),
        'dtype': args.dtype, 'attention': 'eager', 'batch_size': 1,
        'load_seconds': round(time.monotonic() - started, 2),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'qwen-asr')},
        'method': baseline['method'], 'cases': [],
    }
    report['progressive'] = args.progressive
    report['partial_interval_seconds'] = args.interval if args.progressive else None
    report['rollback_tokens'] = args.rollback_tokens
    report['context'] = args.context
    suffix = '-progressive' if args.progressive else ''
    if args.dtype != 'float16':
        suffix += '-' + args.dtype
    if args.rollback_tokens is not None:
        suffix += f'-rollback{args.rollback_tokens}'
    if args.context:
        import hashlib
        suffix += '-context' + hashlib.sha256(args.context.encode()).hexdigest()[:8]
    destination = folder / (args.model.split('/')[-1] + suffix + '-report.json')
    # Warm up without touching the measured examples.
    import numpy as np
    model.transcribe(audio=(np.zeros(16000, dtype=np.float32), 16000), language=None)
    for case in baseline['cases']:
        if args.progressive and not case['name'].startswith(('en_de_en', 'de_en_de')):
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        updates = []
        if args.progressive:
            import soundfile as sf
            samples, rate = sf.read(folder / (case['name'] + '.wav'), dtype='float32')
            duration = len(samples) / rate
            last_end = 0
            raw_decoded = ''
            while True:
                # Audio continues to arrive during GPU work. Never queue stale snapshots.
                available = min(len(samples), int((time.monotonic() - started) * rate))
                if available < len(samples) and available - last_end < args.interval * rate:
                    time.sleep(0.02)
                    continue
                before = time.monotonic()
                with torch.inference_mode():
                    if args.rollback_tokens is None:
                        result = model.transcribe(audio=(samples[:available], rate), language=None, context=args.context)[0]
                    else:
                        from types import SimpleNamespace
                        from qwen_asr import parse_asr_output
                        prefix = ''
                        if len(updates) >= 2:
                            ids = model.processor.tokenizer.encode(raw_decoded)
                            end = max(0, len(ids) - args.rollback_tokens)
                            prefix = model.processor.tokenizer.decode(ids[:end])
                            while '\ufffd' in prefix and end:
                                end -= 1
                                prefix = model.processor.tokenizer.decode(ids[:end])
                        prompt = model._build_text_prompt(context=args.context, force_language=None) + prefix
                        inputs = model.processor(text=[prompt], audio=[samples[:available]], return_tensors='pt', padding=True)
                        inputs = inputs.to(model.model.device).to(model.model.dtype)
                        outputs = model.model.generate(**inputs, max_new_tokens=256)
                        continuation = model.processor.batch_decode(outputs.sequences[:, inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        raw_decoded = prefix + continuation
                        language, text = parse_asr_output(raw_decoded, user_language=None)
                        result = SimpleNamespace(language=language, text=text)
                torch.cuda.synchronize()
                updates.append({'audio_seconds': round(available / rate, 3),
                    'ready_at_seconds': round(time.monotonic() - started, 3),
                    'decode_seconds': round(time.monotonic() - before, 3), 'text': result.text})
                print(f'Partial {case["name"]} {available/rate:.1f}s: {result.text}', flush=True)
                last_end = available
                if available == len(samples):
                    break
        else:
            with torch.inference_mode():
                result = model.transcribe(audio=str(folder / (case['name'] + '.wav')), language=None, context=args.context)[0]
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        item = {
            'name': case['name'], 'reference': case['reference'], 'text': result.text,
            'language': result.language, **metrics(case['reference'], result.text),
            'elapsed_seconds': round(elapsed, 3), 'rtf': round(elapsed / case['seconds'], 3),
            'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 1024**2),
            'whisper': case['results']['auto'],
        }
        if args.progressive:
            item.update(updates=updates, first_partial_seconds=updates[0]['ready_at_seconds'],
                final_ready_after_audio_seconds=round(elapsed-duration, 3),
                replay_walltime_ratio=item['rtf'],
                rtf=round(sum(update['decode_seconds'] for update in updates)/duration, 3))
        report['cases'].append(item)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'{item["name"]} | WER {item["wer"]:.1%} CER {item["cer"]:.1%} | {elapsed:.2f}s | {result.text}', flush=True)
    print(f'Report: {destination}', flush=True)


if __name__ == '__main__':
    main()
