"""Startup must not overlap GPU model initialization or mask initializer failures."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import BrokenThreadPool
import pytest
from voice_bridge import server
from voice_bridge.settings import load_config


def setup_runtime(tmp_path,monkeypatch):
    config=load_config();config['tts']['device']='cpu';config['memory']['database']=str(tmp_path/'memory.sqlite3')
    monkeypatch.setattr(server,'load_settings',lambda:config)
    monkeypatch.setattr(server.mp,'set_executable',lambda _:None)
    monkeypatch.setattr(server,'ProcessPoolExecutor',lambda workers,mp_context,initializer,initargs:ThreadPoolExecutor(workers,initializer=initializer,initargs=initargs))
    monkeypatch.setattr(server,'ready',lambda:True)
    monkeypatch.setattr(server,'warmup_stt',lambda:True)
    monkeypatch.setattr(server,'warmup_tts',lambda:True)
    return config


@pytest.mark.asyncio
async def test_model_initializers_do_not_overlap(tmp_path,monkeypatch):
    setup_runtime(tmp_path,monkeypatch);stt_ready=threading.Event();closed=[]
    def init_stt(*_):time.sleep(.04);stt_ready.set()
    def init_tts(*_):assert stt_ready.is_set(),'TTS allocated before recognizer finished loading'
    monkeypatch.setattr(server,'init_stt',init_stt);monkeypatch.setattr(server,'init_tts',init_tts)
    monkeypatch.setattr(server,'shutdown_stt',lambda:closed.append(True))
    async with server.lifespan(server.app):assert stt_ready.is_set()
    assert closed==[True]


@pytest.mark.asyncio
async def test_broken_initializer_is_not_submitted_another_cleanup_task(tmp_path,monkeypatch):
    setup_runtime(tmp_path,monkeypatch);submitted=[]
    class CountingPool(ThreadPoolExecutor):
        def submit(self,function,*args,**kwargs):submitted.append(function);return super().submit(function,*args,**kwargs)
    def fail(*_):raise RuntimeError('Synthetic GPU allocation failure')
    monkeypatch.setattr(server,'ThreadPoolExecutor',CountingPool)
    monkeypatch.setattr(server,'init_stt',fail);monkeypatch.setattr(server,'init_tts',lambda *_:None)
    monkeypatch.setattr(server,'shutdown_stt',lambda:None)
    with pytest.raises(BrokenThreadPool):
        async with server.lifespan(server.app):pass
    assert submitted==[server.ready]
