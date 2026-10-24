import numpy as np
from scripts.runtime import qwen_asr_worker as runtime


SEGMENTATION = {
    'minimum_total_seconds':8.0,'frame_ms':20,'minimum_pause_ms':80,
    'minimum_segment_seconds':2,'preferred_maximum_segment_seconds':8,
    'minimum_tail_seconds':.7,'maximum_segments':16,'minimum_rms':.003,
    'noise_multiplier':1.8,'maximum_relative_threshold':.3,
}


def test_partition_cuts_in_pause_and_preserves_every_sample():
    audio=np.r_[np.zeros(200),np.full(2400,.1),np.zeros(120),np.full(2500,.2)].astype(np.float32)
    parts=runtime.pause_segments(audio,SEGMENTATION,sample_rate=1000)
    assert len(parts)==2 and 2600<=parts[0][1]<=2720
    np.testing.assert_array_equal(np.concatenate([audio[a:b] for a,b in parts]),audio)


def test_trailing_silence_is_not_a_separate_noise_utterance():
    audio=np.r_[np.full(2500,.1),np.zeros(3000)].astype(np.float32)
    assert runtime.pause_segments(audio,SEGMENTATION,sample_rate=1000)==[(0,len(audio))]


def test_no_forced_word_cut_when_no_pause_is_found():
    audio=np.full(15000,.1,np.float32)
    assert runtime.pause_segments(audio,SEGMENTATION,sample_rate=1000)==[(0,len(audio))]


def test_supported_language_switch_uses_all_chunks(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,4000),(4000,8000)])
    audio=np.r_[np.full(4000,.1),np.full(4000,.2)].astype(np.float32)
    calls=[]
    def decode(part,beams,padding):
        calls.append((len(part),beams,padding))
        return ('raw','English','I will wait.') if part.mean()<.15 else ('raw','German','Der Zug fällt aus.')
    result=runtime.adaptive_transcription(audio,decode,SEGMENTATION,sample_rate=1000)
    assert result['strategy']=='pause_chunks'
    assert result['text']=='I will wait. Der Zug fällt aus.'
    assert result['language']=='English' and calls==[(4000,3,200),(4000,3,200)]


def test_monolingual_chunks_keep_original_whole_decoder(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,4000),(4000,8000)])
    audio=np.full(8000,.1,np.float32)
    calls=[]
    def decode(part,beams,padding):
        calls.append((len(part),beams,padding))
        return ('whole','English','Reliable full sentence.') if len(part)==len(audio) else ('chunk','English','Incomplete fragment.')
    result=runtime.adaptive_transcription(audio,decode,SEGMENTATION,sample_rate=1000)
    assert result['text']=='Reliable full sentence.'
    assert result['strategy']=='whole_monolingual_or_uncertain'
    assert calls[-1]==(8000,1,None)


def test_short_requests_need_only_original_decode():
    calls=[]
    def decode(part,beams,padding):
        calls.append((beams,padding));return ('raw','German','Hallo.')
    result=runtime.adaptive_transcription(np.ones(1000),decode,SEGMENTATION,sample_rate=1000)
    assert result['text']=='Hallo.' and calls==[(1,None)]


def test_speech_regions_exclude_noise_only_tail_from_chunking():
    audio=np.r_[np.full(2800,.1),np.zeros(200),np.full(2800,-.1),np.full(4000,.004)].astype(np.float32)
    settings={**SEGMENTATION,'speech_head_padding_seconds':.2,'speech_tail_padding_seconds':.32}
    bounds=runtime.pause_segments(audio,settings,sample_rate=1000,speech_regions=[[0,2800],[3000,5800]])
    assert len(bounds)==2 and bounds[-1][1]<=6120
    assert bounds[0][0]==0 and 2800<=bounds[0][1]<=3000


def test_instruction_echo_cannot_make_monolingual_speech_look_bilingual(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,4000),(4000,8000)])
    audio=np.r_[np.full(4000,.1),np.full(4000,.2)].astype(np.float32)
    context='Transcribe English and German speech verbatim without adding or translating any words.'
    def decode(part,beams,padding):
        if len(part)==8000:return ('raw','German','Der Damm wurde gebaut.')
        return ('raw','German','Der Damm wurde gebaut.') if part.mean()<.15 else ('raw','English',context)
    result=runtime.adaptive_transcription(audio,decode,SEGMENTATION,sample_rate=1000,prompt_context=context)
    assert result['strategy']=='whole_unreliable_chunk'
    assert result['text']=='Der Damm wurde gebaut.'


def test_implausible_chunk_word_rate_falls_back_without_deleting_speech(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,7000),(7000,8000)])
    audio=np.ones(8000,dtype=np.float32)
    def decode(part,beams,padding):
        if len(part)==8000:return ('raw','German','Vollständiger Originalsatz.')
        return ('raw','German','Ein Satz.') if len(part)>1000 else ('raw','English','word '*20)
    result=runtime.adaptive_transcription(audio,decode,SEGMENTATION,sample_rate=1000)
    assert result['strategy']=='whole_unreliable_chunk' and result['text']=='Vollständiger Originalsatz.'


def test_streaming_preview_does_not_add_a_monolingual_whole_decode(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,4000),(4000,8000)])
    calls=[]
    def decode(part,beams,padding):
        calls.append(len(part));return ('raw','English','A preview fragment.')
    result=runtime.adaptive_transcription(np.ones(8000),decode,SEGMENTATION,sample_rate=1000,preview=True)
    assert result['strategy']=='pause_preview' and calls==[4000,4000]


def test_batched_segments_keep_original_audio_order(monkeypatch):
    monkeypatch.setattr(runtime,'pause_segments',lambda *args,**kwargs:[(0,4000),(4000,8000)])
    def forbidden(*args):raise AssertionError('Unexpected whole or serial decode')
    def batch(bounds,beams,padding):
        assert bounds==[(0,4000),(4000,8000)]
        return [('a','English','I will wait.'),('b','German','Der Zug fällt aus.')]
    result=runtime.adaptive_transcription(np.ones(8000),forbidden,SEGMENTATION,sample_rate=1000,segment_decoder=batch)
    assert result['text']=='I will wait. Der Zug fällt aus.'
