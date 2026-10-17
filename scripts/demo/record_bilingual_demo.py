"""Record a real bilingual voice session using a synthetic browser microphone.

No mocked assistant replies. Uses isolated demo memory, restores the regular app,
records actual playback cuts, and publishes only synthetic demo material locally.
"""
import argparse, asyncio, base64, hashlib, json, os, signal, subprocess, sys, time, wave
from pathlib import Path
import httpx
import numpy as np
import psutil
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT, load_config

# A continuous weekend story, deliberately distinct from Voca's birthday picnic.
SCENES=[
 ('remember',[('en','Bridge, remember that my German level is A2 and our Berlin food market budget is sixty euros.')]),
 ('plan',[('en','Hi Bridge! Lena and I are exploring a Berlin food market this Saturday. We love vegetarian food. Suggest three fun things we could do, with a little excitement!')]),
 ('interrupt',[('en','Wait, one change!'),('de','Es wird regnen.'),('en','Give us one indoor idea instead. Just two short sentences, please.')]),
 ('mixed_teaching',[('en','I want to tell Lena'),('de','Ich freue mich auf Samstag.'),('en','Does that sound natural? Explain briefly, then say the German sentence once.')]),
 ('translation_request',[('en','Bridge, translator mode.')]),
 ('translation_confirm',[('en','Yes, confirm.')]),
 ('english_guest',[('en','Hi Lena! I am excited about Saturday. If it rains, shall we meet inside the market at eleven?')]),
 ('german_host',[('de','Ja, sehr gern! Wir treffen uns am Haupteingang. Ich kenne dort einen Stand mit vegetarischen Gerichten.')]),
 ('english_guest_again',[('en','That sounds wonderful! We have sixty euros for both of us. Is there also somewhere to get hot chocolate?')]),
 ('german_host_again',[('de','Natürlich! Neben dem Blumenladen gibt es ein kleines Café. Danach können wir zusammen nach einem Geschenk suchen.')]),
 ('conversation_return',[('en','Bridge, conversation mode.')]),
 ('memory_recall',[('en','Before we go, what German level and market budget did I ask you to remember? Keep it to one sentence.')]),
 ('bilingual_finish',[('de','Das klingt super!'),('en','Teach me how to say I am looking forward to it in German, then wish us a great Saturday in English. Keep it short.')]),
]

BOOTSTRAP=r'''
window.__serverEvents=[];window.__playedSources=[];window.__spokenInputs=[];
const RealSocket=window.WebSocket;
window.WebSocket=class extends RealSocket {
 constructor(...args){super(...args);this.addEventListener('message',event=>{
 const {pcm,...safe}=JSON.parse(event.data);window.__serverEvents.push({time:performance.now(),...safe});});}
};
const realStart=AudioBufferSourceNode.prototype.start;
const realStop=AudioBufferSourceNode.prototype.stop;
AudioBufferSourceNode.prototype.start=function(...args){
 if(this.buffer){const item={start:performance.now(),contextStart:this.context.currentTime,role:this.__demoRole||'assistant',
 rate:this.buffer.sampleRate,samples:this.buffer.getChannelData(0).slice(),duration:this.buffer.duration};
 window.__playedSources.push(item);this.__recordingItem=item;this.addEventListener('ended',()=>{item.contextEnd=this.context.currentTime;});}
 return realStart.apply(this,args);
};
AudioBufferSourceNode.prototype.stop=function(...args){if(this.__recordingItem)this.__recordingItem.contextEnd=this.context.currentTime;return realStop.apply(this,args);};
const RealContext=window.AudioContext;
window.AudioContext=class extends RealContext{
 constructor(...args){super({sinkId:{type:'none'}});window.__demoContext=this;window.__microphoneSink=this.createMediaStreamDestination();
 const quiet=this.createConstantSource();quiet.offset.value=.000001;quiet.connect(window.__microphoneSink);quiet.start();}
};
navigator.mediaDevices.getUserMedia=async()=>window.__microphoneSink.stream;
window.__say=async ({url,index})=>{const ctx=window.__demoContext;const buffer=await ctx.decodeAudioData(await(await fetch(url)).arrayBuffer());
 const source=ctx.createBufferSource();source.buffer=buffer;source.__demoRole='user';source.connect(window.__microphoneSink);
 window.__spokenInputs.push({index,start:performance.now(),duration:buffer.duration});
 await new Promise(resolve=>{source.onended=resolve;source.start();});};
window.__exportAudio=()=>window.__playedSources.map(item=>{const bytes=new Uint8Array(item.samples.buffer);let binary='';
 for(let i=0;i<bytes.length;i+=16384)binary+=String.fromCharCode(...bytes.subarray(i,i+16384));
 return {start:item.start,role:item.role,rate:item.rate,samples:btoa(binary),duration:Math.min(item.duration,Math.max(0,(item.contextEnd??window.__demoContext.currentTime)-item.contextStart))};});
'''


def prepare(work):
    from voice_bridge.workers import init_tts,synthesize
    config=load_config();config['tts']['device']='cpu';config['tts']['voices']['en']['name']='am_michael'
    init_tts(config['tts'],ROOT)
    rows=[]
    for index,(name,spans) in enumerate(SCENES):
        parts=[]
        for lang,text in spans:
            pcm,rate=synthesize(text,lang);audio=np.frombuffer(pcm,'<i2').astype(np.float32)/32768
            audio=np.interp(np.arange(0,len(audio),rate/16000),np.arange(len(audio)),audio).astype(np.float32)
            parts.extend([audio,np.zeros(1920,np.float32)])
        audio=np.concatenate(parts)
        with wave.open(str(work/f'input-{index}.wav'),'wb') as h:
            h.setnchannels(1);h.setsampwidth(2);h.setframerate(16000);h.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
        rows.append({'index':index,'name':name,'spans':spans,'seconds':len(audio)/16000})
        print('Prepared',name,round(len(audio)/16000,1),'s',flush=True)
    (work/'inputs.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')


async def record(work):
    media=ROOT/'docs/demo';media.mkdir(parents=True,exist_ok=True)
    frames=work/'frames';frames.mkdir(exist_ok=True)
    frame_times=[];capture=True
    async with async_playwright() as p:
        browser=await p.chromium.launch(args=['--no-sandbox','--autoplay-policy=no-user-gesture-required','--disable-gpu'])
        context=await browser.new_context(viewport={'width':1440,'height':1050},device_scale_factor=1)
        await context.add_init_script(BOOTSTRAP)
        page=await context.new_page();errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        async def route(request):
            name=request.request.url.rsplit('/',1)[-1]
            if name not in {f'input-{i}.wav' for i in range(len(SCENES))}:await request.abort();return
            await request.fulfill(path=str(work/name),content_type='audio/wav')
        await page.route('**/__demo_input__/*',route)
        await page.goto('http://127.0.0.1:8767',wait_until='networkidle')
        before=time.monotonic();browser_time=await page.evaluate('performance.now()');clock_offset=(before+time.monotonic())/2-browser_time/1000
        async def record_frames():
            while capture:
                began=time.monotonic()
                await page.screenshot(path=str(frames/f'{len(frame_times):06}.jpg'),type='jpeg',quality=85)
                frame_times.append(began)
                await asyncio.sleep(max(0,.2-(time.monotonic()-began)))
        frame_task=asyncio.create_task(record_frames())
        async def say(index):
            count=await page.evaluate('__serverEvents.length')
            print('Speaking',SCENES[index][0],flush=True)
            await page.evaluate('args=>__say(args)',{'url':f'/__demo_input__/input-{index}.wav','index':index})
            await page.wait_for_function("n=>__serverEvents.slice(n).some(e=>e.type==='final')",arg=count,timeout=90000)
            return count
        async def finish(count):
            await page.wait_for_function("n=>__serverEvents.slice(n).some(e=>e.type==='audio') && !playing && !queue.length && !responsePending && document.body.dataset.state==='listening'",arg=count,timeout=120000)
            batch=await page.evaluate('n=>__serverEvents.slice(n)',count)
            failures=[e['text'] for e in batch if e['type']=='error']
            if failures:raise RuntimeError(str(failures))
            print('Reply:', ''.join(e['text'] for e in batch if e['type']=='token'),flush=True)
            await page.wait_for_timeout(650)
        try:
            await page.wait_for_timeout(800)
            await page.get_by_role('button',name='Start conversation',exact=True).click()
            await page.wait_for_function("active && document.body.dataset.state==='listening'",timeout=30000)
            for index in range(len(SCENES)):
                if index==2:continue
                count=await say(index)
                if index==1:
                    await page.wait_for_function("playing!==null",timeout=90000)
                    await page.wait_for_timeout(1000)
                    count=await say(2)
                    await finish(count)
                    assert await page.locator('.message.interrupted').count()>0,'No actual voice interruption'
                    await page.screenshot(path=str(media/'poster.png'))
                else:await finish(count)
                if index==4:
                    assert await page.locator('#confirmation-card').is_visible(),'Translation did not request confirmation'
                if index==5:
                    assert await page.evaluate("bridgeMode==='translation'"),'Translation confirmation failed'
                if index==10:
                    assert await page.evaluate("bridgeMode==='conversation'"),'Conversation mode did not return'
            await page.wait_for_timeout(900)
            await page.screenshot(path=str(media/'conversation.png'))
            events=await page.evaluate('__serverEvents')
            inputs=await page.evaluate('__spokenInputs')
            transcript=await page.evaluate("""()=>[...document.querySelectorAll('#messages .message')].map(n=>({role:n.classList.contains('user')?'user':'assistant',text:n.querySelector('.message-body').textContent,interrupted:n.classList.contains('interrupted')}))""")
            (work/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
            (work/'spoken-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
            (media/'transcript.json').write_text(json.dumps(transcript,ensure_ascii=False,indent=2)+'\n')
            (media/'transcript.txt').write_text('\n\n'.join(f"[{r['role']}{' — interrupted; displayed text may not have played' if r['interrupted'] else ''}]\n{r['text']}" for r in transcript)+'\n')
            capture=False;await frame_task
            # Stop capture/capture watchdogs before serializing the large PCM mix.
            await page.evaluate("active ? stopSession('Demo complete') : undefined")
            (work/'browser-errors.json').write_text(json.dumps(errors))
            audio=await page.evaluate('__exportAudio()')
            (work/'audio-segments.json').write_text(json.dumps(audio))
            (work/'frame-timeline.json').write_text(json.dumps({'clock_offset':clock_offset,'frames':frame_times}))
        finally:
            capture=False;await frame_task
            await context.close();await browser.close()
        if errors:raise RuntimeError('Browser errors: '+str(errors))


def encode(work):
    media=ROOT/'docs/demo';frames=work/'frames'
    timeline=json.loads((work/'frame-timeline.json').read_text());times=timeline['frames'];offset=timeline['clock_offset']
    audio=json.loads((work/'audio-segments.json').read_text())
    cut_chunks=[item for item in audio if item['role']=='assistant' and
                item['duration']<len(base64.b64decode(item['samples']))/4/item['rate']-.15]
    if not cut_chunks:raise RuntimeError('No observed interruption cut in browser audio')
    export_transcript(work,audio)
    duration=times[-1]-times[0]+.2;entries=['ffconcat version 1.0']
    for i,stamp in enumerate(times):entries.extend([f"file '{i:06}.jpg'",f"duration {(times[i+1]-stamp if i+1<len(times) else .2):.6f}"])
    entries.append(f"file '{len(times)-1:06}.jpg'");concat=frames/'frames.ffconcat';concat.write_text('\n'.join(entries)+'\n')
    rate=48000;mix=np.zeros(round(duration*rate),np.float32)
    for segment in audio:
        a=np.frombuffer(base64.b64decode(segment['samples']),'<f4')[:round(segment['duration']*segment['rate'])]
        if not len(a):continue
        a=np.interp(np.arange(0,len(a),segment['rate']/rate),np.arange(len(a)),a)
        start=round((offset+segment['start']/1000-times[0])*rate)
        if start<0:a=a[-start:];start=0
        count=min(len(a),max(0,len(mix)-start))
        if count:mix[start:start+count]+=a[:count]
    with wave.open(str(work/'soundtrack.wav'),'wb') as h:
        h.setnchannels(1);h.setsampwidth(2);h.setframerate(rate);h.writeframes((np.clip(mix,-1,1)*32767).astype('<i2').tobytes())
    print('Encoding actual session',round(duration,1),'seconds',flush=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-i',str(work/'soundtrack.wav'),'-map','0:v:0','-map','1:a:0','-vf','scale=1280:-2','-r','24','-c:v','libx264','-profile:v','baseline','-level:v','4.0','-preset','fast','-crf','23','-pix_fmt','yuv420p','-threads','4','-c:a','aac','-profile:a','aac_low','-ar','48000','-b:a','128k','-t',f'{duration:.3f}','-movflags','+faststart',str(media/'bridge-demo.mp4')],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(media/'bridge-demo.mp4'),'-c:v','libvpx-vp9','-crf','38','-b:v','0','-deadline','realtime','-cpu-used','6','-threads','4','-c:a','libvorbis','-q:a','4',str(media/'bridge-demo.webm')],check=True)
    report={'recorded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'duration_seconds':round(duration,3),'scene_count':len(SCENES),'scenario':'A rainy Saturday at a Berlin food market','synthetic_user_voices':{'en':'Kokoro am_michael','de':'Piper Thorsten High'},'assistant_voices':{'en':'Kokoro af_heart','de':'Piper Thorsten High'},'stt':'Qwen3-ASR-0.6B FP32 CUDA','llm':'gemma4:31b on Ollama Cloud','mocked_assistant_responses':False,'physical_microphone_or_speaker_test':False,'real_voice_interruption_verified':True,'translation_confirmation_verified':True,'soundtrack':'Actual browser audio buffers, scheduled starts and observed stops; original pauses retained','captions':'Source synthesis text for user; actual synthesized phrase text for assistant; interrupted phrase caption omitted','browser_errors':json.loads((work/'browser-errors.json').read_text()) if (work/'browser-errors.json').exists() else None,'video_bytes':(media/'bridge-demo.mp4').stat().st_size}
    report['observed_stopped_assistant_chunks']=len(cut_chunks)
    report['recorded_user_utterances']=sum(item['role']=='user' for item in audio)
    (media/'recording.json').write_text(json.dumps(report,indent=2)+'\n')
    build_captions(work,times[0]-offset)
    print('Saved',media/'bridge-demo.mp4',flush=True)


def stamp(seconds):
    ms=round(max(0,seconds)*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'


def export_transcript(work,audio):
    # Mode changes rebuild the visible history. Preserve the complete recorded
    # stream, not just whichever mode happens to be visible at the final frame.
    cuts=set()
    for item in audio:
        if item['role']=='assistant' and item['duration']<len(base64.b64decode(item['samples']))/4/item['rate']-.15:
            pcm=(np.frombuffer(base64.b64decode(item['samples']),'<f4')*32768).astype('<i2').tobytes()
            cuts.add((item['rate'],hashlib.sha256(pcm).hexdigest()))
    cut_turns=set()
    for path in (work/'pipeline-traces').glob('*/events.jsonl'):
        for line in path.read_text().splitlines():
            phrase=json.loads(line)
            if phrase['event']!='tts_audio':continue
            with wave.open(str(path.parent/phrase['audio'])) as h:
                if (h.getframerate(),hashlib.sha256(h.readframes(h.getnframes())).hexdigest()) in cuts:cut_turns.add(phrase['turn'])
    rows=[];answers={};current=0
    for event in json.loads((work/'events.json').read_text()):
        if event['type']=='interrupt':current=event['turn']
        elif event['type']=='final':rows.append({'role':'user','turn':current,'text':event['text'],'interrupted':False})
        elif event['type']=='token':
            turn=event.get('turn',current)
            if turn not in answers:
                answers[turn]={'role':'assistant','turn':turn,'text':'','interrupted':turn in cut_turns};rows.append(answers[turn])
            answers[turn]['text']+=event['text']
    media=ROOT/'docs/demo'
    (media/'transcript.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (media/'transcript.txt').write_text('\n\n'.join(f"[{r['role']}{' — interrupted; displayed text may not have played' if r['interrupted'] else ''}]\n{r['text']}" for r in rows)+'\n')


def build_captions(work,origin):
    events=json.loads((work/'events.json').read_text());inputs=json.loads((work/'spoken-inputs.json').read_text())
    audio=json.loads((work/'audio-segments.json').read_text());trace=[]
    for file in (work/'pipeline-traces').glob('*/events.jsonl'):trace.extend(json.loads(line) for line in file.read_text().splitlines())
    native={}
    for path in (work/'pipeline-traces').glob('*/events.jsonl'):
        for line in path.read_text().splitlines():
            phrase=json.loads(line)
            if phrase['event']!='tts_audio':continue
            with wave.open(str(path.parent/phrase['audio'])) as handle:
                key=(handle.getframerate(),hashlib.sha256(handle.readframes(handle.getnframes())).hexdigest())
            native.setdefault(key,[]).append(phrase)
    cues=[]
    for item in inputs:
        start=item['start']/1000-origin
        cues.append((start,start+item['duration'],'Speaker: '+' '.join(text for lang,text in SCENES[item['index']][1])))
    for item in audio:
        if item['role']!='assistant':continue
        # Match exact PCM, rather than packet timing: queued chunks may arrive early.
        pcm=(np.frombuffer(base64.b64decode(item['samples']),'<f4')*32768).astype('<i2').tobytes()
        candidates=native.get((item['rate'],hashlib.sha256(pcm).hexdigest()),[])
        if not candidates:raise RuntimeError('Played assistant audio has no exact trace match')
        phrase=candidates.pop(0)
        if item['duration'] < phrase['audio_seconds']-.15:continue
        start=item['start']/1000-origin;cues.append((start,start+item['duration'],'Bridge: '+phrase['text']))
    lines=['WEBVTT','','NOTE Synthesis-text captions aligned to actual playback; interrupted phrases omitted.','']
    for i,(start,end,text) in enumerate(sorted(cues),1):lines.extend([str(i),f'{stamp(start)} --> {stamp(end)}',text,''])
    (ROOT/'docs/demo/captions.vtt').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path);parser.add_argument('--encode',type=Path);args=parser.parse_args()
    if args.prepare:prepare(args.prepare);return
    if args.encode:encode(args.encode);return
    work=ROOT/'logs'/('demo-'+time.strftime('%Y%m%d-%H%M%S'));work.mkdir(parents=True)
    subprocess.run([sys.executable,__file__,'--prepare',str(work)],check=True)
    for p in psutil.process_iter(['pid','cmdline','cwd']):
        try:
            command=p.info['cmdline'] or []
            if p.info['cwd']==str(ROOT) and 'voice_bridge.server:app' in command and '8766' in command:
                p.send_signal(signal.SIGINT);p.wait(timeout=40)
        except psutil.NoSuchProcess:pass
    env=os.environ.copy();env['VOICE_BRIDGE_DEMO_DIR']=str(work.relative_to(ROOT));
    if Path('/mnt/d/voice-bridge-tools/playwright').is_dir():env.setdefault('PLAYWRIGHT_BROWSERS_PATH','/mnt/d/voice-bridge-tools/playwright')
    os.environ.update({k:v for k,v in env.items() if k=='PLAYWRIGHT_BROWSERS_PATH'})
    log=(work/'server.log').open('w');process=subprocess.Popen([sys.executable,'-m', 'scripts.application.start_voice_app','-m', 'scripts.demo.start_demo_server'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+180
        while True:
            if process.poll() is not None:raise RuntimeError('Demo server stopped; inspect '+str(work/'server.log'))
            try:
                if httpx.get('http://127.0.0.1:8767/health',timeout=1).status_code==200:break
            except httpx.HTTPError:pass
            if time.monotonic()>deadline:raise TimeoutError('Demo server startup')
            time.sleep(.5)
        asyncio.run(record(work))
    finally:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        log.close()
        with (ROOT/'logs/normal-after-demo.log').open('a') as normal:
            subprocess.Popen([sys.executable,'-m', 'scripts.application.start_voice_app'],cwd=ROOT,stdout=normal,stderr=subprocess.STDOUT,start_new_session=True)
    encode(work)


if __name__=='__main__':main()
