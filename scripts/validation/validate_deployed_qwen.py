"""Exercise real VAD → deployed Qwen → cloud → TTS; remove only own test turns."""
import asyncio
import base64
import json
import sqlite3
import time
import wave
import numpy as np
import websockets
from voice_bridge.settings import ROOT, load_settings


async def main():
    config = load_settings()
    base = f"http://localhost:{config['server']['port']}"
    database = ROOT / config['memory']['database']
    results = []
    async with websockets.connect(base.replace('http', 'ws') + '/ws', origin=base, max_size=8*1024*1024) as ws:
        events = []
        async def receive():
            async for value in ws:
                packet = json.loads(value)
                if packet['type'] == 'audio':
                    assert base64.b64decode(packet.pop('pcm'))
                    await ws.send(json.dumps({'type':'played', 'id':packet['id'], 'complete':True}))
                # Do not export stored user history or memory notes from mode packets.
                if packet['type'] != 'mode':
                    events.append(packet)
        receiver = asyncio.create_task(receive())
        async def wait(predicate):
            async with asyncio.timeout(90):
                while not predicate():
                    errors = [e['text'] for e in events if e['type']=='error']
                    if errors:
                        raise AssertionError(errors)
                    await asyncio.sleep(.05)
        await wait(lambda:any(e['type']=='ready' for e in events))
        for name, expected_language, anchor in [('english_complex','en','appointment'), ('german_complex','de','termin'), ('en_de_en_travel','en','anschluss')]:
            begin = len(events)
            with sqlite3.connect(database) as db:
                before = db.execute('SELECT COALESCE(MAX(id),0) FROM turns').fetchone()[0]
            try:
                with wave.open(str(ROOT/'logs/codeswitch-check'/f'{name}.wav')) as handle:
                    assert handle.getframerate()==16000
                    samples = np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
                silence = max(profile['end_of_turn_silence_ms'] for key,profile in config['modes'].items() if isinstance(profile,dict))
                audio = np.concatenate([np.zeros(512*10,np.float32), samples, np.zeros(int(16000*(silence/1000+.4)),np.float32)])
                started = time.monotonic()
                for offset in range(0,len(audio),512):
                    frame = np.zeros(512,dtype='<f4'); chunk=audio[offset:offset+512];frame[:len(chunk)]=chunk
                    await ws.send(frame.tobytes())
                    await asyncio.sleep(max(0, started+(offset+512)/16000-time.monotonic()))
                await wait(lambda:any(e['type']=='audio' for e in events[begin:]) and any(e['type']=='status' and e.get('text')=='Listening' for e in events[begin:]))
                turn = events[begin:]
                final = next(e for e in turn if e['type']=='final')
                assert final['language']==expected_language, final
                assert anchor in final['text'].casefold(), final
                if name=='en_de_en_travel':
                    assert all(word in final['text'].casefold() for word in ('reservation','verspätung','midnight')),final
                assert any(e['type']=='partial' for e in turn)
                assert any(e['type']=='token' for e in turn)
                results.append({'case':name,'final':final,'partials':sum(e['type']=='partial' for e in turn),
                    'audio_chunks':sum(e['type']=='audio' for e in turn), 'route':next(e for e in turn if e['type']=='route')})
                print(f'{name}: partials, final, cloud tokens and TTS audio PASS',flush=True)
            finally:
                # Only delete newly created rows whose text is a final from this test.
                own_texts = [e['text'] for e in events[begin:] if e['type']=='final']
                with sqlite3.connect(database) as db:
                    for text in own_texts:
                        db.execute('DELETE FROM turns WHERE id>? AND user=?',(before,text))
        await ws.send(json.dumps({'type':'stop'})); await receiver
    (ROOT/'logs/qwen-deployment-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('Deployed Qwen GPU + streaming Ollama + chunked TTS: PASS')


if __name__=='__main__':
    asyncio.run(main())
