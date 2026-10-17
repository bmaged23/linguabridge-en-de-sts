"""Live VAD/STT/cloud/TTS transport checks using synthetic bilingual audio.

No assistant packets are mocked. Playback is decoded/acknowledged by the test,
not played to a physical speaker. Writes synthetic test turns into local translation history.
"""
import asyncio
import base64
import json
import time
import wave
from pathlib import Path
import numpy as np
import websockets
from voice_bridge.settings import ROOT,load_settings


def samples(language):
    path=ROOT/'logs/speech-check'/f'{language}.wav'
    with wave.open(str(path),'rb') as handle:
        rate=handle.getframerate();data=np.frombuffer(handle.readframes(handle.getnframes()),'<i2').astype(np.float32)/32768
    return np.interp(np.arange(0,len(data),rate/16000),np.arange(len(data)),data).astype('<f4')


async def main():
    config=load_settings();port=config['server']['port'];events=[];latest_mode=None
    async with websockets.connect(f'ws://localhost:{port}/ws',origin=f'http://localhost:{port}',max_size=8*1024*1024) as ws:
        async def receive():
            nonlocal latest_mode
            async for message in ws:
                packet=json.loads(message)
                if packet['type']=='mode':latest_mode=packet
                if packet['type']=='audio':
                    raw=base64.b64decode(packet.pop('pcm'));assert len(raw)>0 and len(raw)%2==0
                    await ws.send(json.dumps({'type':'played','id':packet['id'],'complete':True}))
                events.append(packet)
        receiver=asyncio.create_task(receive())
        async def wait(predicate,timeout=90):
            start=time.monotonic()
            while not predicate():
                errors=[e['text'] for e in events if e['type']=='error']
                if errors:raise AssertionError(errors)
                if time.monotonic()-start>timeout:raise TimeoutError('Session check timed out')
                await asyncio.sleep(.05)
        await wait(lambda:latest_mode is not None)
        original_mode=latest_mode['mode']
        await ws.send(json.dumps({'type':'set_mode','mode':'translation'}))
        await wait(lambda:latest_mode['mode']=='translation' or latest_mode.get('pending_confirmation'))
        if latest_mode.get('pending_confirmation'):
            await ws.send(json.dumps({'type':'confirm_change','id':latest_mode['pending_confirmation']['id']}))
        await wait(lambda:latest_mode['mode']=='translation')
        try:
            for index,language in enumerate(('en','en','de')):
                begin=len(events);audio=np.concatenate([np.zeros(512*10,np.float32),samples(language),np.zeros(512*44,np.float32)])
                for offset in range(0,len(audio),512):
                    frame=np.zeros(512,dtype='<f4');chunk=audio[offset:offset+512];frame[:len(chunk)]=chunk
                    await ws.send(frame.tobytes());await asyncio.sleep(.032)
                await wait(lambda:any(e['type']=='status' and e.get('text')=='Listening' for e in events[begin:]) and any(e['type']=='audio' for e in events[begin:]))
                route=next(e for e in events[begin:] if e['type']=='route');expected='de' if language=='en' else 'en'
                assert route['source']==language and route['target']==expected,route
                assert all(e['language']==expected for e in events[begin:] if e['type']=='audio')
                text=''.join(e['text'] for e in events[begin:] if e['type']=='token')
                print(f'Turn {index+1} {language} → {expected}: {text}',flush=True)
                if index==1:
                    for _ in range(188):await ws.send(np.zeros(512,dtype='<f4').tobytes());await asyncio.sleep(.032)
                    assert ws.state.name=='OPEN'
                    print('Six-second pause: connection remains ready',flush=True)
        finally:
            await ws.send(json.dumps({'type':'set_mode','mode':original_mode}))
            await ws.send(json.dumps({'type':'stop'}));await receiver
        out=ROOT/'logs/session-check.json';out.write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    print('Live VAD, EN→DE twice, DE→EN, TTS routing, and idle waiting: PASS')


if __name__=='__main__':asyncio.run(main())
