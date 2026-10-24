import numpy as np
from scripts.runtime.qwen_asr_worker import SegmentCache


def test_cache_reuses_identical_model_input_but_not_revised_speech():
    cache=SegmentCache();core=np.array([.1,.2,0,-.1],dtype=np.float32)
    a=np.r_[np.zeros(8),core,np.zeros(10)].astype(np.float32)
    b=np.r_[np.zeros(12),core,np.zeros(4)].astype(np.float32)
    key=cache.key(a,0,len(a),'context',3,200,1)
    cache.put(key,('raw','English','hello'))
    assert cache.get(cache.key(b,0,len(b),'context',3,200,1))==('raw','English','hello')
    b[13]=.21
    assert cache.get(cache.key(b,0,len(b),'context',3,200,1)) is None


def test_cache_is_bounded_and_clearable_between_utterances():
    cache=SegmentCache(2)
    for key in range(3):cache.put(key,('raw','English',str(key)))
    assert len(cache.entries)==2 and cache.get(0) is None
    cache.clear()
    assert not cache.entries


def test_decoder_settings_are_part_of_cache_identity():
    cache=SegmentCache();audio=np.ones(100,dtype=np.float32)
    key=cache.key(audio,0,100,'context',3,200,1)
    assert key!=cache.key(audio,0,100,'other context',3,200,1)
    assert key!=cache.key(audio,0,100,'context',1,200,1)
    assert key!=cache.key(audio,0,100,'context',3,200,2)
