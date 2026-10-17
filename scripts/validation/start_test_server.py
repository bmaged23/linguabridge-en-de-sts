"""Real GPU app with isolated synthetic memory and explicit QA fault injection."""
from contextlib import asynccontextmanager
from pathlib import Path
import json,os
import httpx
import uvicorn
from fastapi import Request,HTTPException
from voice_bridge import server
from voice_bridge.settings import ROOT,load_settings


class QAClient:
    def __init__(self,client):self.client=client;self.fault=None
    async def post(self,*args,**kwargs):return await self.client.post(*args,**kwargs)
    @asynccontextmanager
    async def stream(self,*args,**kwargs):
        fault=self.fault;self.fault=None
        if fault=='cloud_timeout':raise httpx.ReadTimeout('Synthetic QA timeout')
        if fault=='length':
            class Response:
                def raise_for_status(self):pass
                async def aiter_lines(self):
                    yield json.dumps({'message':{'content':'[en]This response was intentionally limited for the recovery test.'}})
                    yield json.dumps({'done':True,'done_reason':'length'})
            yield Response()
        else:
            async with self.client.stream(*args,**kwargs) as response:yield response


if __name__=='__main__':
    settings=load_settings();folder=ROOT/os.environ.get('VOICE_BRIDGE_QA_DIR','logs/full-qa');folder.mkdir(parents=True,exist_ok=True)
    settings['server']['port']=8767
    settings['stt']['audit_fresh_final']=os.environ.get('VOICE_BRIDGE_QA_AUDIT_FINAL','1')=='1'
    settings['memory']['database']=str(folder/'memory.sqlite3')
    settings['logging']['pipeline_trace']={'enabled':True,'directory':str((folder/'pipeline-traces').relative_to(ROOT))}
    server.load_settings=lambda:settings
    @server.app.post('/__qa__/fault')
    async def inject_fault(request:Request):
        kind=(await request.json()).get('kind')
        if kind not in ('cloud_timeout','length'):raise HTTPException(400,'Unknown QA fault')
        if not isinstance(server.app.state.client,QAClient):server.app.state.client=QAClient(server.app.state.client)
        server.app.state.client.fault=kind
        return {'armed':kind}
    @server.app.post('/__qa__/audit-audio')
    async def audit_audio(request:Request):
        import asyncio,re,wave,time
        import numpy as np
        from voice_bridge.workers import transcribe
        value=await request.json();identity=value.get('session','');name=value.get('audio','')
        if not re.fullmatch(r'[a-f0-9]{32}',identity) or not re.fullmatch(r'[0-9]+-[0-9]+\.wav',name):raise HTTPException(400,'Invalid synthetic audio identity')
        path=folder/'pipeline-traces'/identity/name
        if not path.is_file():raise HTTPException(404,'No synthetic audio')
        with wave.open(str(path)) as handle:
            rate=handle.getframerate();samples=np.frombuffer(handle.readframes(handle.getnframes()),'<i2').astype(np.float32)/32768
        audio=np.interp(np.arange(0,len(samples),rate/16000),np.arange(len(samples)),samples).astype(np.float32)
        started=time.perf_counter()
        result=await asyncio.get_running_loop().run_in_executor(server.app.state.stt,transcribe,audio,None,None,True)
        return {**result,'audit_seconds':round(time.perf_counter()-started,4)}
    uvicorn.run(server.app,host='127.0.0.1',port=8767,log_level='info',access_log=False)
