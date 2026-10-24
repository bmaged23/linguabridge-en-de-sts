"""Persistent isolated Qwen CUDA worker. Stdout is reserved for JSON replies."""
import base64
import contextlib
import json
from pathlib import Path
import sys
import re
import hashlib


class SegmentCache:
    """Only independently decoded, complete segments may enter this bounded cache."""
    def __init__(self, maximum_entries=64):
        self.maximum_entries=maximum_entries
        self.entries={}

    def clear(self):self.entries.clear()

    def key(self,audio,left,right,context,beams,padding,batch_size):
        value=audio[left:right]
        if padding is not None:value=canonicalize_padding(value,padding)
        digest=hashlib.sha256(value.astype('<f4',copy=False).tobytes()).digest()
        return len(value),context,beams,padding,batch_size,digest

    def get(self,key):return self.entries.get(key)

    def put(self,key,value):
        self.entries[key]=value
        while len(self.entries)>self.maximum_entries:self.entries.pop(next(iter(self.entries)))


def canonicalize_padding(audio, padding_ms, sample_rate=16000):
    """Remove only exterior digital silence and restore a fixed acoustic margin."""
    import numpy as np
    nonzero = np.flatnonzero(audio)
    if not len(nonzero):
        return audio
    padding = np.zeros(round(sample_rate*padding_ms/1000), dtype=audio.dtype)
    return np.concatenate((padding, audio[nonzero[0]:nonzero[-1]+1], padding))


def pause_segments(audio, settings, sample_rate=16000, speech_regions=None):
    """Partition at sustained low-energy pauses using audio alone, without language labels."""
    import numpy as np
    offset=0
    if speech_regions:
        offset=max(0,speech_regions[0][0]-round(settings['speech_head_padding_seconds']*sample_rate))
        end=min(len(audio),speech_regions[-1][1]+round(settings['speech_tail_padding_seconds']*sample_rate))
        audio=audio[offset:end]
    frame = max(1, round(sample_rate*settings['frame_ms']/1000))
    rms = np.array([np.sqrt(np.mean(audio[i:i+frame]**2)) for i in range(0,len(audio),frame)])
    if not len(rms):return [(offset,offset+len(audio))]
    threshold = max(settings['minimum_rms'], min(float(np.percentile(rms,10))*settings['noise_multiplier'],
                    float(np.percentile(rms,90))*settings['maximum_relative_threshold']))
    quiet = rms <= threshold
    changes = np.diff(np.r_[False,quiet,False].astype(np.int8))
    starts,ends = np.flatnonzero(changes==1),np.flatnonzero(changes==-1)
    pauses=[]
    for start,end in zip(starts,ends):
        if start==0 or end==len(rms):continue
        if (end-start)*settings['frame_ms']>=settings['minimum_pause_ms']:
            pauses.append(min(len(audio),int((start+end)*frame//2)))
    minimum=round(settings['minimum_segment_seconds']*sample_rate)
    maximum=round(settings['preferred_maximum_segment_seconds']*sample_rate)
    tail=round(settings['minimum_tail_seconds']*sample_rate)
    result=[];left=0
    for _ in range(settings['maximum_segments']-1):
        candidate=next((p for p in pauses if left+minimum<=p<=min(left+maximum,len(audio)-tail)),None)
        if candidate is None:break
        result.append((left,candidate));left=candidate
    result.append((left,len(audio)))
    return [(a+offset,b+offset) for a,b in result]


def adaptive_transcription(audio, decode, segmentation, chunk_beams=3, padding_ms=200, whole_beams=1, sample_rate=16000, speech_regions=None, prompt_context='', segment_decoder=None, preview=False):
    """Use chunk output only when independent chunks identify both supported languages."""
    def whole(reason, pieces):
        raw,language,text=decode(audio,whole_beams,None)
        return {'raw':raw,'language':language,'text':text,'strategy':reason,'pieces':pieces}
    if len(audio)<segmentation['minimum_total_seconds']*sample_rate:
        return whole('whole_short',[])
    if speech_regions==[]:return whole('whole_no_speech_regions',[])
    bounds=pause_segments(audio,segmentation,sample_rate,speech_regions=speech_regions)
    if len(bounds)<2:return whole('whole_no_pause',[])
    pieces=[]
    decoded=segment_decoder(bounds,chunk_beams,padding_ms) if segment_decoder else [
        decode(audio[left:right],chunk_beams,padding_ms) for left,right in bounds]
    if len(decoded)!=len(bounds):raise ValueError('ASR batch returned the wrong number of segments')
    for (left,right),(raw,language,text) in zip(bounds,decoded):
        pieces.append({'start_seconds':left/sample_rate,'end_seconds':right/sample_rate,
                       'raw':raw,'language':language,'text':text})
    context_words=re.findall(r'\w+',prompt_context.casefold())
    for piece in pieces:
        words=re.findall(r'\w+',piece['text'].casefold())
        duration=piece['end_seconds']-piece['start_seconds']
        echo=(len(context_words)>=segmentation.get('minimum_context_echo_words',8) and words==context_words)
        impossible_rate=len(words)>duration*segmentation.get('maximum_chunk_words_per_second',8)
        if echo or impossible_rate:
            return whole('whole_unreliable_chunk',pieces)
    languages={p['language'] for p in pieces if p['text'].strip()}
    word_counts={language:sum(len(re.findall(r'\w+',p['text'])) for p in pieces if p['language']==language)
                 for language in languages}
    if preview:
        first=next((p for p in pieces if p['text'].strip()),{'language':''})
        return {'raw':'\n'.join(p['raw'] for p in pieces),'language':first['language'],
                'text':' '.join(p['text'].strip() for p in pieces if p['text'].strip()),
                'strategy':'pause_preview','pieces':pieces}
    if languages!={'English','German'} or any(
            count<segmentation.get('minimum_words_per_language',1) for count in word_counts.values()):
        return whole('whole_monolingual_or_uncertain',pieces)
    first=next(p for p in pieces if p['text'].strip())
    return {'raw':'\n'.join(p['raw'] for p in pieces),'language':first['language'],
            'text':' '.join(p['text'].strip() for p in pieces if p['text'].strip()),
            'strategy':'pause_chunks','pieces':pieces}


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
        decode_calls = 0
        cache=SegmentCache()
        cache_hits=0
        batch_fallbacks=0

        def decode(audio,context,beams,prefix='',padding_ms=None):
            nonlocal decode_calls
            if padding_ms is not None:audio=canonicalize_padding(audio,padding_ms)
            if not len(audio) or np.max(np.abs(audio))<settings['silence_peak_threshold']:
                return '','',''
            prompt=model._build_text_prompt(context=context,force_language=None)+prefix
            with torch.inference_mode():
                inputs=model.processor(text=[prompt],audio=[audio],return_tensors='pt',padding=True)
                inputs=inputs.to(model.model.device).to(model.model.dtype)
                output=model.model.generate(**inputs,max_new_tokens=settings['max_new_tokens'],do_sample=False,num_beams=beams)
                decode_calls+=1
                generated=output.sequences[:,inputs['input_ids'].shape[1]:]
                eos=model.model.generation_config.eos_token_id
                eos=eos if isinstance(eos,list) else [eos]
                if generated.shape[1]>=settings['max_new_tokens'] and int(generated[0,-1]) not in eos:
                    raise ValueError('STT token limit reached. Use a shorter turn or increase stt.max_new_tokens in config.py.')
                continuation=model.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            raw=prefix+continuation
            language,text=parse_asr_output(raw,user_language=None)
            return raw,language,text

        def decode_batch(values,context,beams,padding_ms):
            nonlocal decode_calls,batch_fallbacks
            if len(values)==1:return [decode(values[0],context,beams,padding_ms=padding_ms)]
            prepared=[canonicalize_padding(a,padding_ms) if padding_ms is not None else a for a in values]
            if any(not len(a) or np.max(np.abs(a))<settings['silence_peak_threshold'] for a in prepared):
                return [decode(a,context,beams,padding_ms=padding_ms) for a in values]
            failed=False
            inputs=output=generated=None
            try:
                prompt=model._build_text_prompt(context=context,force_language=None)
                with torch.inference_mode():
                    inputs=model.processor(text=[prompt]*len(prepared),audio=prepared,return_tensors='pt',padding=True)
                    inputs=inputs.to(model.model.device).to(model.model.dtype)
                    output=model.model.generate(**inputs,max_new_tokens=settings['max_new_tokens'],do_sample=False,num_beams=beams)
                    decode_calls+=1
                    generated=output.sequences[:,inputs['input_ids'].shape[1]:]
                    eos=model.model.generation_config.eos_token_id
                    eos=eos if isinstance(eos,list) else [eos]
                    if any(len(row)>=settings['max_new_tokens'] and int(row[-1]) not in eos for row in generated):
                        raise ValueError('STT token limit reached in batch')
                    texts=model.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                return [(text,*parse_asr_output(text,user_language=None)) for text in texts]
            except torch.cuda.OutOfMemoryError:
                failed=True
            if failed:
                batch_fallbacks+=1
                inputs=output=generated=None
                torch.cuda.empty_cache()
                return [decode(a,context,beams,padding_ms=padding_ms) for a in values]

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
                    cache=SegmentCache(settings.get('stream_cache_max_entries',64))
                elif packet['type'] == 'transcribe':
                    if model is None:
                        raise RuntimeError('Qwen is not initialized')
                    audio = np.frombuffer(base64.b64decode(packet['audio'], validate=True), dtype='<f4').copy()
                    if not np.isfinite(audio).all() or len(audio) > settings['window_seconds'] * 16000:
                        raise ValueError('Invalid ASR audio')
                    options = packet.get('decode_options') or {}
                    if set(options) - {'context', 'num_beams', 'padding_ms', 'strategy', 'speech_gate', 'stream_cache', 'batch_size'}:
                        raise ValueError('Unsupported ASR decode option')
                    context = options.get('context', settings['context'])
                    beams = options.get('num_beams', settings.get('num_beams', 1))
                    padding_ms = options.get('padding_ms', settings.get('final_padding_ms'))
                    identity = packet.get('stream_id')
                    caching=options.get('stream_cache',settings.get('stream_chunk_cache',False))
                    if type(caching) is not bool:raise ValueError('ASR stream cache must be a boolean')
                    cached=caching and identity is not None
                    adaptive_preview=cached and len(audio)>=settings.get('final_segmentation',{}).get('minimum_total_seconds',8)*16000
                    strategy = options.get('strategy',settings.get('final_strategy','whole')) if packet.get('final',True) or adaptive_preview else 'whole'
                    batch_size=options.get('batch_size',settings.get('final_batch_size',1))
                    if type(batch_size) is not int or not 1<=batch_size<=4:raise ValueError('Invalid ASR batch size')
                    if strategy not in ('whole','adaptive_pause_chunks'):raise ValueError('Invalid final ASR strategy')
                    if not isinstance(context, str) or len(context) > 2048 or type(beams) is not int or not 1 <= beams <= 5:
                        raise ValueError('Invalid ASR context or beam count')
                    if padding_ms is not None and (type(padding_ms) is not int or not 0 <= padding_ms <= 2000):
                        raise ValueError('Invalid final ASR padding')
                    if identity is None or identity != stream or len(audio) < last_size:
                        stream = identity; raw = ''; updates = 0
                        cache.clear()
                    last_size = len(audio)
                    prefix = ''
                    if updates >= settings['unfixed_chunk_num'] and strategy=='whole':
                        ids = model.processor.tokenizer.encode(raw)
                        end = max(0, len(ids) - settings['rollback_tokens'])
                        prefix = model.processor.tokenizer.decode(ids[:end])
                        while '\ufffd' in prefix and end:
                            end -= 1; prefix = model.processor.tokenizer.decode(ids[:end])
                    decode_calls=0
                    cache_hits=0
                    batch_fallbacks=0
                    pieces=[]
                    if strategy=='adaptive_pause_chunks':
                        def segments(bounds,nb,pad):
                            nonlocal cache_hits
                            values=[None]*len(bounds)
                            pending=[]
                            for index,(left,right) in enumerate(bounds):
                                key=cache.key(audio,left,right,context,nb,pad,batch_size)
                                found=cache.get(key) if cached else None
                                if found is not None:values[index]=found;cache_hits+=1
                                else:pending.append((index,left,right,key))
                            for start in range(0,len(pending),batch_size):
                                group=pending[start:start+batch_size]
                                result=decode_batch([audio[left:right] for _,left,right,_ in group],context,nb,pad)
                                for (index,left,right,key),value in zip(group,result):
                                    values[index]=value
                                    if cached and index<len(bounds)-1:cache.put(key,value)
                            return values
                        selected=adaptive_transcription(audio,lambda value,nb,pad:decode(value,context,nb,padding_ms=pad),
                            settings['final_segmentation'],chunk_beams=options.get('num_beams',settings.get('final_num_beams',3)),
                            padding_ms=padding_ms,whole_beams=settings.get('num_beams',1),speech_regions=packet.get('speech_regions'),
                            prompt_context=context,segment_decoder=segments,preview=not packet.get('final',True))
                        raw,language,text=selected['raw'],selected['language'],selected['text']
                        strategy=selected['strategy'];pieces=selected['pieces']
                    else:
                        raw,language,text=decode(audio,context,beams,prefix,
                            padding_ms=padding_ms if packet.get('final',True) else None)
                    updates += 1
                    code = {'English': 'en', 'German': 'de'}.get(language, 'en')
                    supported = language in ('English', 'German')
                    result = {'text': text, 'language': code, 'detected_language': language,
                        'certain': supported, 'probability': None, 'used_context': False,
                        'can_route': supported and bool(text), 'supported_probability': None,
                        'decode_strategy':strategy,'model_decode_calls':decode_calls,
                        'cached_segment_hits':cache_hits,'batch_oom_fallbacks':batch_fallbacks}
                    if settings.get('audit_raw_outputs',False):
                        result.update(decoder_raw=raw,reused_prefix=prefix,updates=updates,decoder_chunks=pieces)
                    if packet.get('final', True):
                        stream = None; raw = ''; updates = 0; last_size = 0
                        cache.clear()
                else:
                    raise ValueError('Unknown Qwen worker request')
            except Exception as error:
                stream = None; raw = ''; updates = 0; last_size = 0
                cache.clear()
                result = {'error': f'{type(error).__name__}: {error}'}
            protocol.write(json.dumps(result, ensure_ascii=False) + '\n'); protocol.flush()


if __name__ == '__main__':
    main()
