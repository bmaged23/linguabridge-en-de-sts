"""Measure local resource use, cloud latency, speech generation and live turn timing."""
import asyncio
import base64
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import wave

import httpx
import numpy as np
import psutil
import websockets
from voice_bridge.settings import ROOT, load_settings


class Monitor(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid=pid;self.phase='idle';self.samples=[];self.stop_event=threading.Event();self.processes={}

    def run(self):
        while not self.stop_event.is_set():
            try:
                root=psutil.Process(self.pid); processes=[root,*root.children(recursive=True)]
                cpu=0;rss=0;pss=0
                for process in processes:
                    try:
                        tracked=self.processes.setdefault(process.pid,process)
                        cpu+=tracked.cpu_percent();rss+=tracked.memory_info().rss
                        pss+=getattr(tracked.memory_full_info(),'pss',0)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0].split(',')
                memory=psutil.virtual_memory()
                self.samples.append({'phase':self.phase,'time':time.monotonic(),'process_cpu_percent':cpu,
                    'cpu_machine_percent':cpu/psutil.cpu_count(),'system_cpu_percent':psutil.cpu_percent(),
                    'rss_mib':rss/1024**2,'pss_mib':pss/1024**2,'vram_mib':float(gpu[0]),'gpu_percent':float(gpu[1]),
                    'system_ram_percent':memory.percent,'system_available_mib':memory.available/1024**2})
            except (psutil.NoSuchProcess,subprocess.SubprocessError):pass
            self.stop_event.wait(.25)

    def finish(self):
        self.stop_event.set();self.join()
        result={}
        for phase in sorted({sample['phase'] for sample in self.samples}):
            samples=[sample for sample in self.samples if sample['phase']==phase]
            result[phase]={'samples':len(samples)}
            for key in ('process_cpu_percent','cpu_machine_percent','system_cpu_percent','rss_mib','pss_mib','vram_mib','gpu_percent','system_ram_percent'):
                values=[sample[key] for sample in samples]
                result[phase][key]={'mean':round(sum(values)/len(values),2),'peak':round(max(values),2)}
        return result


def tts_worker(pipe, config):
    from voice_bridge.workers import init_tts, synthesize, warmup_tts
    from voice_bridge.core import PhraseBuffer
    init_tts(config['tts'],ROOT);warmup_tts();pipe.send({'ready':True})
    while (request:=pipe.recv()) is not None:
        language,text=request;settings=config['tts']['chunking']
        phrases=PhraseBuffer(settings['min_characters'],settings['max_characters']).push(text,flush=True)
        times=[];seconds=0
        for phrase in phrases:
            start=time.monotonic();pcm,rate=synthesize(phrase,language);times.append(time.monotonic()-start)
            seconds+=len(pcm)/2/rate
        pipe.send({'first_chunk_seconds':round(times[0],3),'total_generation_seconds':round(sum(times),3),
            'audio_seconds':round(seconds,3),'chunks':len(times),'words':len(text.split())})


async def live_turn(ws, name, monitor, database):
    with wave.open(str(ROOT/'logs/codeswitch-check'/f'{name}.wav')) as handle:
        samples=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
    with sqlite3.connect(database) as db:before=db.execute('SELECT COALESCE(MAX(id),0) FROM turns').fetchone()[0]
    times={};finals=[];audio_chunks=0;output_audio_seconds=0;tokens=0
    async def send():
        audio=np.concatenate([np.zeros(512*10,np.float32),samples,np.zeros(16000*2,np.float32)])
        times['input_audio_end']=time.monotonic()+.32+len(samples)/16000
        start=time.monotonic()
        for offset in range(0,len(audio),512):
            frame=np.zeros(512,dtype='<f4');chunk=audio[offset:offset+512];frame[:len(chunk)]=chunk
            await ws.send(frame.tobytes());await asyncio.sleep(max(0,start+(offset+512)/16000-time.monotonic()))
    sender=asyncio.create_task(send())
    monitor.phase=name+'_stt'
    try:
        async with asyncio.timeout(120):
            async for value in ws:
                packet=json.loads(value);kind=packet['type'];now=time.monotonic()
                if kind=='error':raise RuntimeError(packet['text'])
                if kind=='partial':times.setdefault('partial',now)
                if kind=='status' and packet.get('text')=='Transcribing':times['stt_start']=now
                if kind=='final':times['final']=now;finals.append(packet['text'])
                if kind=='status' and packet.get('text')=='Thinking':times['llm_start']=now;monitor.phase=name+'_llm_and_tts'
                if kind=='token':times.setdefault('token',now);times['last_token']=now;tokens+=1
                if kind=='audio':
                    times.setdefault('audio',now);audio_chunks+=1
                    output_audio_seconds+=len(base64.b64decode(packet['pcm']))/2/packet['sample_rate']
                    await ws.send(json.dumps({'type':'played','id':packet['id'],'complete':True}))
                if kind=='status' and packet.get('text')=='Listening' and audio_chunks:break
        await sender
        return {'input_words':len(finals[-1].split()),'input_audio_seconds':round(len(samples)/16000,2),
            'stt_final_seconds':round(times['final']-times['stt_start'],3),
            'vad_wait_from_wav_end_seconds':round(times['stt_start']-times['input_audio_end'],3),
            'llm_first_token_seconds':round(times['token']-times['llm_start'],3),
            'llm_last_token_seconds':round(times['last_token']-times['llm_start'],3),
            'first_audio_after_final_seconds':round(times['audio']-times['final'],3),
            'first_audio_after_wav_end_seconds':round(times['audio']-times['input_audio_end'],3),
            'audio_chunks':audio_chunks,'output_audio_seconds':round(output_audio_seconds,2),'token_packets':tokens}
    finally:
        sender.cancel();await asyncio.gather(sender,return_exceptions=True)
        with sqlite3.connect(database) as db:
            for text in finals:db.execute('DELETE FROM turns WHERE id>? AND user=?',(before,text))


async def main():
    config=load_settings();base=f"http://localhost:{config['server']['port']}"
    server=next(p for p in psutil.process_iter(['cmdline']) if 'voice_bridge.server:app' in (p.info['cmdline'] or []))
    monitor=Monitor(server.pid);monitor.start();await asyncio.sleep(2)
    report={'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),
        'system_ram_gib':round(psutil.virtual_memory().total/1024**3,2),'server_pid':server.pid,
        'scope':'App process plus descendants; CPU 100%=one logical core; machine CPU divides by logical core count. GPU metrics are whole-device. RAM PSS apportions shared pages; RSS sum double-counts shared memory.',
        'live':{}}
    try:
        async with websockets.connect(base.replace('http','ws')+'/ws',origin=base,max_size=8*1024*1024) as ws:
            for name in ('english_complex','german_complex'):
                report['live'][name]=await live_turn(ws,name,monitor,ROOT/config['memory']['database'])
                print(name,report['live'][name],flush=True)
            await ws.send(json.dumps({'type':'stop'}))
    finally:report['app_resources']=monitor.finish()
    if config['tts']['device']=='cuda':
        # Do not load a second pair of GPU voices alongside the resident app.
        destination=ROOT/'logs/runtime-profile-gpu.json';destination.write_text(json.dumps(report,indent=2)+'\n')
        print('Saved live GPU pipeline profile',destination,flush=True)
        return
    context=mp.get_context('spawn');parent,child=context.Pipe();process=context.Process(target=tts_worker,args=(child,config));process.start()
    try:
        assert await asyncio.to_thread(parent.recv)=={'ready':True}
        tts_monitor=Monitor(process.pid);tts_monitor.start();await asyncio.sleep(1)
        report['tts']={}
        texts={'en':'I can help you move your appointment to Friday afternoon. Please check whether the later train arrives before your meeting, and let me know if you would like another option.',
            'de':'Ich kann Ihnen helfen, den Termin auf Freitagnachmittag zu verschieben. Bitte prüfen Sie, ob der spätere Zug vor Ihrem Treffen ankommt, und sagen Sie mir, ob Sie eine andere Verbindung benötigen.'}
        try:
            for language,text in texts.items():
                runs=[];tts_monitor.phase=language+'_tts'
                for _ in range(3):
                    parent.send((language,text));runs.append(await asyncio.to_thread(parent.recv))
                report['tts'][language]=runs
                print(language,'TTS',runs,flush=True)
        finally:report['tts_resources']=tts_monitor.finish()
    finally:
        if process.is_alive():parent.send(None)
        await asyncio.to_thread(process.join,10)
        if process.is_alive():process.terminate();process.join()
    destination=ROOT/'logs/runtime-profile.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',destination,flush=True)


if __name__=='__main__':asyncio.run(main())
