"""Browser QA with real local speech, live cloud replies, and virtual microphone."""
import asyncio
import json
from pathlib import Path
import wave
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.workers import init_tts,synthesize

BOOTSTRAP=r'''
window.__events=[];
const RealSocket=window.WebSocket;
window.WebSocket=class extends RealSocket{
 constructor(...args){super(...args);this.addEventListener('message',event=>{const {pcm,...safe}=JSON.parse(event.data);window.__events.push(safe);});}
};
const RealContext=window.AudioContext;
window.AudioContext=class extends RealContext{
 constructor(...args){super(...args);window.__ctx=this;window.__mic=this.createMediaStreamDestination();const quiet=this.createConstantSource();quiet.offset.value=.000001;quiet.connect(window.__mic);quiet.start();}
};
navigator.mediaDevices.getUserMedia=async()=>window.__mic.stream;
window.__say=async url=>{const ctx=window.__ctx;const buffer=await ctx.decodeAudioData(await(await fetch(url)).arrayBuffer());const source=ctx.createBufferSource();source.buffer=buffer;source.connect(window.__mic);await new Promise(resolve=>{source.onended=resolve;source.start();});};
'''
PROMPTS=[('en','Bridge, teaching mode.'),('en','What is my German level? Please give me one short practice exercise.'),
         ('en','Where is the train station?'),('de','Der Bahnhof ist neben dem Supermarkt.'),
         ('en','Bridge, translator mode.'),('en','No, cancel.'),
         ('en','Bridge, translator mode.'),('en','Yes, confirm.')]


def prepare(directory,config):
    init_tts(config['tts'],ROOT)
    for i,(lang,text) in enumerate(PROMPTS):
        pcm,rate=synthesize(text,lang)
        with wave.open(str(directory/f'input-{i}.wav'),'wb') as handle:
            handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate);handle.writeframes(pcm)


async def main():
    config=load_settings();base=f"http://localhost:{config['server']['port']}";out=ROOT/'logs/browser-check';out.mkdir(parents=True,exist_ok=True)
    await asyncio.to_thread(prepare,out,config)
    async with async_playwright() as p:
        installed=Path.home()/'.cache/ms-playwright/chromium-1194/chrome-linux/chrome'
        browser=await p.chromium.launch(executable_path=str(installed) if installed.exists() else p.chromium.executable_path,
            args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        context=await browser.new_context(viewport={'width':1440,'height':1150})
        await context.add_init_script(BOOTSTRAP);page=await context.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        async def route(request):
            name=request.request.url.rsplit('/',1)[-1]
            if name not in {f'input-{i}.wav' for i in range(len(PROMPTS))}:await request.abort();return
            await request.fulfill(path=str(out/name),content_type='audio/wav')
        await page.route('**/__qa__/*',route)
        await page.goto(base,wait_until='networkidle')
        await page.get_by_role('button',name='Conversation',exact=True).click()
        await page.wait_for_function("document.querySelector('[data-mode=conversation]').getAttribute('aria-pressed')==='true'")
        await page.locator('#memory-card summary').click()
        await page.locator('#memory-note').fill('QA test preference: my German level is A2.')
        await page.get_by_role('button',name='Remember',exact=True).click()
        await page.get_by_text('QA test preference: my German level is A2.',exact=True).wait_for()
        await page.reload(wait_until='networkidle')
        assert await page.locator('[data-mode=conversation]').get_attribute('aria-pressed')=='true'
        assert 'QA test preference' in await page.locator('#memory-list').text_content()
        await page.get_by_role('button',name='Start conversation',exact=True).click()
        await page.wait_for_function("document.body.dataset.state==='listening'",timeout=30000)
        async def say(index):
            before=await page.evaluate('window.__events.length')
            await page.evaluate('(url)=>window.__say(url)',base+f'/__qa__/input-{index}.wav')
            await page.wait_for_function('''(before)=>window.__events.slice(before).some(e=>e.type==='token')
                &&window.__events.slice(before).some(e=>e.type==='status'&&e.text==='Listening')
                &&document.body.dataset.state==='listening' ''',arg=before,timeout=90000)
            batch=await page.evaluate('(before)=>window.__events.slice(before)',before)
            assert not [e for e in batch if e['type']=='error'],batch
            print(f'Browser turn {index+1}: '+''.join(e['text'] for e in batch if e['type']=='token'),flush=True)
            return batch
        await say(0)
        assert await page.locator('[data-mode=teaching]').get_attribute('aria-pressed')=='true'
        batch=await say(1)
        assert 'a2' in ''.join(e['text'] for e in batch if e['type']=='token').lower()
        await page.get_by_role('button',name='Conversation',exact=True).click()
        await page.wait_for_function("document.querySelector('[data-mode=conversation]').getAttribute('aria-pressed')==='true'")
        await page.locator('#reply-language').select_option('de')
        await page.locator('#confirmation-card').wait_for(state='visible')
        assert await page.locator('#reply-language').input_value()=='same'
        await page.get_by_role('button',name='Cancel',exact=True).click()
        await page.locator('#confirmation-card').wait_for(state='hidden')
        await say(4)
        assert await page.locator('[data-mode=conversation]').get_attribute('aria-pressed')=='true'
        assert await page.locator('#confirmation-card').is_visible()
        await say(5)
        assert await page.locator('[data-mode=conversation]').get_attribute('aria-pressed')=='true'
        assert not await page.locator('#confirmation-card').is_visible()
        await say(6)
        assert await page.locator('#confirmation-card').is_visible()
        await say(7)
        await page.wait_for_function("document.querySelector('[data-mode=translation]').getAttribute('aria-pressed')==='true'")
        for index,source,target in [(2,'en','de'),(3,'de','en')]:
            batch=await say(index)
            route=next(e for e in batch if e['type']=='route')
            assert (route['source'],route['target'])==(source,target),route
            assert all(e['language']==target for e in batch if e['type']=='audio')
        await page.screenshot(path=str(out/'desktop.png'),full_page=True)
        await page.get_by_role('button',name='End conversation',exact=True).click()
        await page.locator('#memory-card summary').click()
        note=page.locator('#memory-list li').filter(has_text='QA test preference')
        await note.get_by_role('button',name='Forget',exact=False).click()
        await page.wait_for_function("!document.querySelector('#memory-list').textContent.includes('QA test preference')")
        await page.set_viewport_size({'width':390,'height':844})
        await page.screenshot(path=str(out/'mobile.png'),full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        events=await page.evaluate('window.__events')
        await page.goto(base+'/docs/architecture.html',wait_until='networkidle')
        await page.get_by_role('button',name='Pause animation',exact=True).click()
        assert await page.locator('svg').evaluate('(s)=>s.animationsPaused()')
        assert not errors,errors
        (out/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
        await browser.close()
    print('Browser voice commands, saved memory recall, real bilingual translation/playback, responsive UI, spoken yes/no and language-change confirmation: PASS')


if __name__=='__main__':asyncio.run(main())
