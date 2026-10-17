import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest
from voice_bridge import server
from voice_bridge.memory import Memory
from voice_bridge.settings import load_config


@pytest.fixture
def session(tmp_path,monkeypatch):
    config=load_config()
    config['llm'].update(base_url='https://ollama.com',api_key='test',model='test',
                         first_token_timeout_seconds=.03,stream_idle_timeout_seconds=.03)
    config['memory']['summary_timeout_seconds']=.03
    memory=Memory(tmp_path/'memory.sqlite3',config['memory'])
    monkeypatch.setattr(server.app.state,'settings',config,raising=False)
    monkeypatch.setattr(server.app.state,'memory',memory,raising=False)
    yield SimpleNamespace(config=config,memory=memory)
    memory.close()


@pytest.mark.asyncio
async def test_interrupt_stops_browser_before_slow_response_cleanup(session):
    packets=[]
    class Socket:
        async def send_json(self,packet):packets.append(packet)
    voice=server.Session(Socket())
    cleaning=asyncio.Event()
    release=asyncio.Event()
    async def response():
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
    voice.response_task=asyncio.create_task(response())
    await asyncio.sleep(0)
    semaphore=asyncio.Semaphore(0)
    voice.outstanding['old']=(voice.turn,'Unfinished speech',semaphore)
    interrupt=asyncio.create_task(voice.interrupt())
    try:
        await asyncio.wait_for(cleaning.wait(),1)
        assert packets==[{'type':'interrupt','turn':voice.turn}]
        assert not interrupt.done()
    finally:
        release.set()
        await asyncio.wait_for(interrupt,1)
    assert not voice.outstanding
    await asyncio.wait_for(semaphore.acquire(),1)


@pytest.mark.asyncio
@pytest.mark.parametrize('first_content',[False,True])
async def test_cloud_keepalives_cannot_extend_text_deadline(session,monkeypatch,first_content):
    class Response:
        def raise_for_status(self):pass
        async def aiter_lines(self):
            if first_content:yield json.dumps({'message':{'content':'Hello'}})
            while True:
                await asyncio.sleep(.005)
                yield json.dumps({'message':{'thinking':'still waiting'}})
    class Client:
        @asynccontextmanager
        async def stream(self,*args,**kwargs):yield Response()
    monkeypatch.setattr(server.app.state,'client',Client(),raising=False)
    voice=server.Session(None)
    tokens=asyncio.Queue()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(voice.stream_llm([],tokens),.5)
    assert tokens.qsize()==int(first_content)


@pytest.mark.asyncio
async def test_slow_summary_preserves_history_and_previous_summary(session,monkeypatch):
    for _ in range(22):session.memory.add_turn('conversation','en','en','Synthetic turn')
    session.memory.save_summary('conversation',1,'A saved fact.')
    class Client:
        async def post(self,*args,**kwargs):await asyncio.sleep(10)
    monkeypatch.setattr(server.app.state,'client',Client(),raising=False)
    voice=server.Session(None)
    await asyncio.wait_for(voice.compact('conversation'),.5)
    assert session.memory.summary('conversation')==(1,'A saved fact.')
    assert session.memory.db.execute('SELECT COUNT(*) FROM turns').fetchone()[0]==22


@pytest.mark.asyncio
async def test_reply_is_delivered_before_summary(session,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    packets=[]
    class Socket:
        async def send_json(self,packet):packets.append(packet)
    monkeypatch.setattr(server.app.state,'diagnostics',{},raising=False)
    monkeypatch.setattr(server.app.state,'controls',SimpleNamespace(voice_answer=lambda *args:None,pending=None),raising=False)
    monkeypatch.setattr(server,'transcribe',lambda *args:{'text':'Hello there','language':'en','certain':True},raising=False)
    executor=ThreadPoolExecutor(1)
    monkeypatch.setattr(server.app.state,'stt',executor,raising=False)
    voice=server.Session(Socket())
    async def stream(messages,tokens):
        await tokens.put('[en]Hello! How can I help you?[/en]');await tokens.put(None)
    async def speak(phrases,turn):
        while await phrases.get() is not None:pass
        packets.append({'type':'test_audio_complete'})
    async def compact(mode):
        assert packets[-1]=={'type':'status','text':'Listening','turn':0}
        assert any(p['type']=='test_audio_complete' for p in packets)
        packets.append({'type':'test_summary'})
    monkeypatch.setattr(voice,'stream_llm',stream)
    monkeypatch.setattr(voice,'speak',speak)
    monkeypatch.setattr(voice,'compact',compact)
    try:
        await asyncio.wait_for(voice.response([],0,0),1)
        assert packets[-1]['type']=='test_summary'
    finally:executor.shutdown()


@pytest.mark.asyncio
async def test_length_limit_does_not_cancel_already_generated_speech(session,monkeypatch):
    packets=[]
    class Socket:
        async def send_json(self,packet):packets.append(packet)
    class Response:
        def raise_for_status(self):pass
        async def aiter_lines(self):
            yield json.dumps({'message':{'content':'[en]Keep this spoken answer.'}})
            yield json.dumps({'done':True,'done_reason':'length'})
    class Client:
        @asynccontextmanager
        async def stream(self,*args,**kwargs):yield Response()
    monkeypatch.setattr(server.app.state,'client',Client(),raising=False)
    voice=server.Session(Socket());tokens=asyncio.Queue()
    await voice.stream_llm([],tokens)
    assert await tokens.get()=='[en]Keep this spoken answer.'
    assert await tokens.get() is None
    assert any(p['type']=='notice' and 'length limit' in p['text'] for p in packets)
