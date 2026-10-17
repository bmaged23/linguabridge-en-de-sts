import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest

from voice_bridge import server
from voice_bridge.memory import Memory
from voice_bridge.settings import load_config


@pytest.mark.asyncio
async def test_parallel_chunks_preserve_playback_order_and_history(tmp_path, monkeypatch):
    config=load_config()
    config['tts']['parallel_chunks_by_language']={'de':2}
    memory=Memory(tmp_path/'memory.sqlite3',config['memory'])
    monkeypatch.setattr(server.app.state,'settings',config,raising=False)
    monkeypatch.setattr(server.app.state,'memory',memory,raising=False)
    monkeypatch.setattr(server.app.state,'diagnostics',{},raising=False)
    packets=[];batches=[]
    class Socket:
        async def send_json(self,packet):
            packets.append(packet)
            if packet['type']=='audio':voice.acknowledge({'id':packet['id'],'complete':True})
    voice=server.Session(Socket());voice.output_language='de'
    row=memory.add_turn('conversation','de','de','Synthetic test')
    voice.answers[0]={'id':row,'heard':''}
    def single(text,language):return b'\0\0',22050
    def batch(phrases,language):
        batches.append(list(phrases));return [single(phrase,language) for phrase in phrases]
    monkeypatch.setattr(server,'synthesize',single)
    monkeypatch.setattr(server,'synthesize_batch',batch)
    executor=ThreadPoolExecutor(1)
    monkeypatch.setattr(server.app.state,'tts',executor,raising=False)
    phrases=asyncio.Queue()
    for phrase in ('Eins.','Zwei.','Drei.','Vier.',None):phrases.put_nowait(phrase)
    try:
        await asyncio.wait_for(voice.speak(phrases,0),5)
        assert batches==[['Zwei.','Drei.']]
        assert [packet['id'] for packet in packets if packet['type']=='audio']==['0:1','0:2','0:3','0:4']
        assert memory.recent('conversation')[0]['heard']=='Eins. Zwei. Drei. Vier. '
        assert not voice.outstanding
    finally:
        executor.shutdown();memory.close()
