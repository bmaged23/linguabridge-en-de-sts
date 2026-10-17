import numpy as np
from voice_bridge.core import Endpoint
from voice_bridge.settings import load_config


def test_separated_noise_bursts_do_not_add_up_to_speech():
    config=load_config();settings={**config['vad'],**{key:config['interruptions'][key] for key in ('speech_threshold','minimum_rms','min_speech_ms')}}
    endpoint=Endpoint(settings);frame=np.full(512,.02,np.float32)
    events=[]
    for _ in range(8):
        for _ in range(2):events+=endpoint.feed(frame,.99)
        for _ in range(3):events+=endpoint.feed(frame,.1)
    assert not events and not endpoint.started
    for _ in range(10):events+=endpoint.feed(frame,.99)
    assert [kind for kind,_ in events]==['start']


def test_high_vad_probability_on_quiet_frames_cannot_interrupt():
    config=load_config();settings={**config['vad'],**{key:config['interruptions'][key] for key in ('speech_threshold','minimum_rms','min_speech_ms')}}
    endpoint=Endpoint(settings)
    quiet=np.full(512,.001,np.float32)
    assert not [event for _ in range(40) for event in endpoint.feed(quiet,.99)]
    assert not endpoint.started


def test_confirmed_speech_survives_a_brief_internal_pause():
    settings=load_config()['vad'];endpoint=Endpoint(settings)
    frame=np.full(512,.02,np.float32)
    events=[event for _ in range(8) for event in endpoint.feed(frame,.99)]
    assert [kind for kind,_ in events]==['start']
    for _ in range(5):assert not endpoint.feed(np.zeros(512,np.float32),0)
    assert endpoint.started
    for _ in range(8):assert not endpoint.feed(frame,.99)
