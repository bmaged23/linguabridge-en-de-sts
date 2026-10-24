"""Final decoding must be able to correct a mistaken multilingual preview."""
import numpy as np
from voice_bridge import workers


def test_final_redecodes_full_audio_without_committed_preview(monkeypatch):
    class Recognizer:
        def __init__(self):
            self.calls = []
        def transcribe(self, audio, identity, final):
            self.calls.append((identity, final))
            return {'text': 'incorrect preview' if identity else 'Unser Zug fiel aus.'}
    model = Recognizer()
    monkeypatch.setattr(workers, '_model', model)
    monkeypatch.setattr(workers, '_settings', {'provider': 'qwen_asr', 'fresh_final_decode': True})
    audio = np.ones(16000, dtype=np.float32)
    preview = workers.transcribe(audio, stream_id='turn-1', final=False)
    result = workers.transcribe(audio, stream_id='turn-1', final=True)
    assert preview['text'] == 'incorrect preview'
    assert result['text'] == 'Unser Zug fiel aus.'
    assert model.calls == [('turn-1', False), (None, True)]


def test_adaptive_cache_keeps_stream_identity_without_reusing_preview_text(monkeypatch):
    calls=[]
    class Recognizer:
        def transcribe(self,audio,identity,final):
            calls.append((identity,final));return {'text':'final independent segments'}
    monkeypatch.setattr(workers,'_model',Recognizer())
    monkeypatch.setattr(workers,'_settings',{'provider':'qwen_asr','fresh_final_decode':True,
        'final_strategy':'adaptive_pause_chunks','stream_chunk_cache':True})
    result=workers.transcribe(np.ones(16000,dtype=np.float32),stream_id='session:4',final=True)
    assert calls==[('session:4',True)] and result['text']=='final independent segments'
