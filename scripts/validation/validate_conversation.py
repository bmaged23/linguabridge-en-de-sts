"""Check the core reported regression: English conversation must reply in English."""
import asyncio
import base64
import json
import time
import numpy as np
import httpx
import websockets
from scripts.validation.validate_voice_session import samples
from voice_bridge.settings import load_settings


async def main():
    config=load_settings();base=f"http://localhost:{config['server']['port']}"
    async with httpx.AsyncClient() as client:
        response=await client.post(base+'/control',headers={'Origin':base},json={'type':'set_mode','mode':'conversation'})
        assert response.is_success,response.status_code
        state=response.json();assert state['mode']=='conversation' and state['response_language']=='same'
    async with websockets.connect(base.replace('http','ws')+'/ws',origin=base,max_size=8*1024*1024) as ws:
        events=[]
        async def send_audio():
            audio=np.concatenate([np.zeros(512*10,np.float32),samples('en'),np.zeros(512*28,np.float32)])
            for offset in range(0,len(audio),512):
                frame=np.zeros(512,dtype='<f4');chunk=audio[offset:offset+512];frame[:len(chunk)]=chunk
                await ws.send(frame.tobytes());await asyncio.sleep(.032)
        sender=None
        async with asyncio.timeout(90):
            async for value in ws:
                packet=json.loads(value);events.append(packet)
                if packet['type']=='ready':sender=asyncio.create_task(send_audio())
                if packet['type']=='error':raise AssertionError(packet['text'])
                if packet['type']=='audio':
                    assert packet['language']=='en';assert base64.b64decode(packet['pcm'])
                    await ws.send(json.dumps({'type':'played','id':packet['id'],'complete':True}))
                if packet['type']=='status' and packet.get('text')=='Listening' and any(e['type']=='audio' for e in events):break
        if sender:await sender
        route=next(e for e in events if e['type']=='route')
        assert route['mode']=='conversation' and route['source']==route['target']=='en'
        print('English conversation reply: '+''.join(e['text'] for e in events if e['type']=='token'))
        await ws.send(json.dumps({'type':'stop'}))
    print('English conversation → English Kokoro speech, without enabling translation: PASS')


if __name__=='__main__':asyncio.run(main())
