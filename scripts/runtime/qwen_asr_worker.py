"""Persistent isolated Qwen CUDA worker. Stdout is reserved for JSON replies."""
import base64
import contextlib
import json
from pathlib import Path
import sys


def main():
    protocol = sys.stdout
    # Third-party progress messages must never corrupt the stdio protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel, parse_asr_output
        model = None
        settings = None
        stream = None
        raw = ''
        updates = 0
        last_size = 0
        for line in sys.stdin:
            try:
                packet = json.loads(line)
                if packet['type'] == 'init':
                    settings = packet['settings']
                    if not torch.cuda.is_available():
                        raise RuntimeError('Qwen requires the configured CUDA GPU')
                    model = Qwen3ASRModel.from_pretrained(str(Path(packet['root']) / settings['model_path']),
                        dtype=getattr(torch, settings['dtype']), device_map=f"cuda:{settings['device_index']}",
                        attn_implementation=settings['attention'], max_inference_batch_size=1,
                        max_new_tokens=settings['max_new_tokens'], local_files_only=True)
                    # Warm kernels without generating a long silence hallucination.
                    result = {'ready': True, 'device': str(model.model.device)}
                elif packet['type'] == 'transcribe':
                    if model is None:
                        raise RuntimeError('Qwen is not initialized')
                    audio = np.frombuffer(base64.b64decode(packet['audio'], validate=True), dtype='<f4').copy()
                    if not np.isfinite(audio).all() or len(audio) > settings['window_seconds'] * 16000:
                        raise ValueError('Invalid ASR audio')
                    identity = packet.get('stream_id')
                    if identity is None or identity != stream or len(audio) < last_size:
                        stream = identity; raw = ''; updates = 0
                    last_size = len(audio)
                    prefix = ''
                    if updates >= settings['unfixed_chunk_num']:
                        ids = model.processor.tokenizer.encode(raw)
                        end = max(0, len(ids) - settings['rollback_tokens'])
                        prefix = model.processor.tokenizer.decode(ids[:end])
                        while '\ufffd' in prefix and end:
                            end -= 1; prefix = model.processor.tokenizer.decode(ids[:end])
                    if not len(audio) or np.max(np.abs(audio)) < settings['silence_peak_threshold']:
                        text = ''; language = ''
                    else:
                        prompt = model._build_text_prompt(context=settings['context'], force_language=None) + prefix
                        with torch.inference_mode():
                            inputs = model.processor(text=[prompt], audio=[audio], return_tensors='pt', padding=True)
                            inputs = inputs.to(model.model.device).to(model.model.dtype)
                            output = model.model.generate(**inputs, max_new_tokens=settings['max_new_tokens'])
                            generated = output.sequences[:, inputs['input_ids'].shape[1]:]
                            eos = model.model.generation_config.eos_token_id
                            eos = eos if isinstance(eos,list) else [eos]
                            if generated.shape[1]>=settings['max_new_tokens'] and int(generated[0,-1]) not in eos:
                                raise ValueError('STT token limit reached. Use a shorter turn or increase stt.max_new_tokens in config.py.')
                            continuation = model.processor.batch_decode(generated,
                                skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        raw = prefix + continuation
                        language, text = parse_asr_output(raw, user_language=None)
                        updates += 1
                    code = {'English': 'en', 'German': 'de'}.get(language, 'en')
                    supported = language in ('English', 'German')
                    result = {'text': text, 'language': code, 'detected_language': language,
                        'certain': supported, 'probability': None, 'used_context': False,
                        'can_route': supported and bool(text), 'supported_probability': None}
                    if settings.get('audit_raw_outputs',False):
                        result.update(decoder_raw=raw,reused_prefix=prefix,updates=updates)
                    if packet.get('final', True):
                        stream = None; raw = ''; updates = 0; last_size = 0
                else:
                    raise ValueError('Unknown Qwen worker request')
            except Exception as error:
                stream = None; raw = ''; updates = 0; last_size = 0
                result = {'error': f'{type(error).__name__}: {error}'}
            protocol.write(json.dumps(result, ensure_ascii=False) + '\n'); protocol.flush()


if __name__ == '__main__':
    main()
