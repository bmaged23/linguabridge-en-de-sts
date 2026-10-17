"""Real browser microphone/WebAudio/ACK lifecycle audit on isolated QA port 8767."""
import argparse,asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT
from scripts.validation.validate_browser_interface import BOOTSTRAP

EXTRA=r'''
window.__out=[];window.__errors=[];
const CaptureSocket=window.WebSocket;
window.WebSocket=class extends CaptureSocket{
 send(data){if(typeof data==='string')window.__out.push(JSON.parse(data));return super.send(data);}
};
'''

async def main():
 parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');parser.add_argument('--mixed',action='store_true');parser.add_argument('--targeted',action='store_true');args=parser.parse_args()
 base='http://127.0.0.1:8767';out=ROOT/'logs/full-qa';prompts=json.loads((out/'prompts.json').read_text())
 if args.mixed:
  from scripts.evaluation.evaluate_mixed_language_asr import CASES
  import shutil
  for name,spans in CASES:
   original=ROOT/'logs/codeswitch-check'/f'{name}.wav'
   if original.exists():shutil.copyfile(original,out/f'{name}.wav')
   prompts.append((name,'mixed',' '.join(t for lang,t in spans)))
 failures=[];results=[];browser_errors=[];semantic_failures=[];ui_confirmation_recovery=None
 async with async_playwright() as p:
  browser=await p.chromium.launch(args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required','--remote-debugging-port=9223','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--audio-buffer-size=512'])
  context=await browser.new_context(viewport={'width':1366,'height':1000})
  await context.add_init_script(BOOTSTRAP.replace('super(...args);window.__ctx=this;',"super({sinkId:{type:'none'}});window.__ctx=this;").replace('window.__events.push(safe)','window.__events.push({...safe,at_ms:performance.now()})')+EXTRA)
  page=await context.new_page()
  def browser_error(error):browser_errors.append(str(error));print('BROWSER ERROR:',error,flush=True)
  page.on('pageerror',browser_error)
  async def route(request):
   name=request.request.url.rsplit('/',1)[-1]
   path=out/name
   if not path.is_file() or path.suffix!='.wav':await request.abort();return
   await request.fulfill(path=str(path),content_type='audio/wav')
  await page.route('**/__qa__/*',route)
  await page.goto(base,wait_until='networkidle')
  assert await page.locator('#status').text_content()=='Microphone off'
  await page.get_by_role('button',name='Start conversation',exact=True).click()
  await page.wait_for_function("active && window.__ctx.state==='running'",timeout=30000)
  assert await page.locator('#status').text_content()=='Waiting for speech'
  async def watch_clock():
   samples=[]
   while True:
    try:
     values=await page.evaluate('''()=>({clock:window.__ctx?.currentTime,now:performance.now(),state:window.__ctx?.state,rate:window.__ctx?.sampleRate,ui:document.body.dataset.state,events:window.__events?.length})''')
     samples.append(values);(out/'audio-clock.json').write_text(json.dumps(samples,indent=2))
     await asyncio.sleep(1)
    except Exception:return
  watcher=asyncio.create_task(watch_clock())
  async def turn(name):
   await page.wait_for_function('!playing && !queue.length && !responsePending',timeout=120000)
   await page.wait_for_timeout(350)
   before=await page.evaluate('window.__events.length');sent=await page.evaluate('window.__out.length');start=time.monotonic()
   await asyncio.wait_for(page.evaluate('(url)=>window.__say(url)',base+f'/__qa__/{name}.wav'),30)
   timed_out=False
   try:
    await page.wait_for_function('''before=>window.__events.slice(before).some(e=>e.type==='error'||e.type==='audio')
      && !playing && !queue.length && !responsePending''',arg=before,timeout=120000)
   except Exception:timed_out=True
   events=await page.evaluate('(n)=>window.__events.slice(n)',before)
   sent_events=await page.evaluate('(n)=>window.__out.slice(n)',sent)
   from scripts.evaluation.evaluate_mixed_language_asr import metrics
   expected=next((text for key,lang,text in prompts if key==name),'')
   finals=[e for e in events if e['type']=='final']
   stt=finals[-1]['text'] if finals else ''
   errors=[e['text'] for e in events if e['type']=='error'];audio=[e for e in events if e['type']=='audio']
   acks=[e for e in sent_events if e['type']=='played' and e.get('complete')]
   row={'case':name,'seconds':round(time.monotonic()-start,2),'timed_out':timed_out,'errors':errors,
        'audio_chunks':len(audio),'played_chunks':len(acks),'languages':[e['language'] for e in audio],
        'reply':''.join(e['text'] for e in events if e['type']=='token'),
        'ui_state':await page.evaluate('document.body.dataset.state'),
        'context_state':await page.evaluate('window.__ctx.state'),'input':expected,'stt_final':stt,'stt_metrics':metrics(expected,stt),'events':events,'playback_acks':sent_events}
   results.append(row)
   if args.targeted and name=='isolated_vocabulary':
    assert all(e.get('language')=='de' for e in events if e['type']=='token' and any(w in e.get('text','').casefold() for w in ('obwohl','weil')))
   if args.targeted and name=='confirm_translation' and await page.evaluate("bridgeMode!=='translation'"):
    semantic_failures.append('Spoken confirmation did not activate translation. STT: '+stt)
   if timed_out or errors or not audio or len(acks)!=len(audio):failures.append(row)
   print(name,'FAIL' if row in failures else 'PASS',row['seconds'],'seconds',f'{len(acks)}/{len(audio)} played',errors,flush=True)
   (out/('targeted-browser.json' if args.targeted else 'mixed-browser.json' if args.mixed else ('baseline-browser.json' if args.baseline else 'browser-report.json'))).write_text(json.dumps({'turns':results,'failures':failures,'browser_errors':browser_errors},ensure_ascii=False,indent=2)+'\n')
   return row
  for name in (['isolated_vocabulary','enable_translation','confirm_translation'] if args.targeted else [name for name,spans in CASES] if args.mixed else ['hello','coffee_lesson','five_pairs'] if args.baseline else ['hello','coffee_lesson','mixed_repeat','german_lesson','five_pairs','long_lesson','continue','recovery']):
   await turn(name)
   if await page.evaluate('responsePending'):
    await page.get_by_role('button',name='End conversation',exact=True).click();await page.get_by_role('button',name='Start conversation',exact=True).click()
    await page.wait_for_function('active',timeout=15000)
  if args.targeted and semantic_failures:
   await page.locator('[data-mode=translation]').click();await page.locator('#confirmation-card').wait_for(state='visible');await page.locator('#confirm-change').click();await page.wait_for_function("bridgeMode==='translation'")
   ui_confirmation_recovery='PASS: UI confirmation activates translation.'
   await page.locator('[data-mode=conversation]').click();await page.wait_for_function("bridgeMode==='conversation'")
  if not args.baseline and not args.mixed and not args.targeted:
   # A real silent interval must not produce an unsolicited response or cancellation.
   before=await page.evaluate('window.__events.length');await page.wait_for_timeout(8000)
   idle=await page.evaluate('(n)=>window.__events.slice(n)',before)
   assert not any(e['type'] in ('interrupt','final','audio','error') for e in idle),idle
   print('Eight-second idle interval PASS',flush=True)
   for name in ['enable_translation','confirm_translation','translate_english','translate_german']:await turn(name)
   # Stop/restart/reconnect must release the old session and resume the microphone.
   await page.get_by_role('button',name='End conversation',exact=True).click()
   await page.wait_for_function('!active && !ws',timeout=15000)
   await page.reload(wait_until='networkidle')
   await page.get_by_role('button',name='Start conversation',exact=True).click()
   await page.wait_for_function('active',timeout=15000)
   await turn('translate_english')
   await page.get_by_role('button',name='Conversation',exact=True).click()
   await page.wait_for_function("bridgeMode==='conversation'",timeout=10000)
   await turn('short_response')
   await page.set_viewport_size({'width':390,'height':844})
   assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  await page.screenshot(path=str(out/('targeted-desktop.png' if args.targeted else 'mixed-desktop.png' if args.mixed else 'baseline-desktop.png' if args.baseline else 'mobile-final.png')),full_page=True)
  await page.get_by_role('button',name='End conversation',exact=True).click()
  await browser.close()
 report={'turns':results,'failures':failures,'browser_errors':browser_errors,'semantic_failures':semantic_failures,'ui_confirmation_recovery':ui_confirmation_recovery}
 (out/('targeted-browser.json' if args.targeted else 'mixed-browser.json' if args.mixed else ('baseline-browser.json' if args.baseline else 'browser-report.json'))).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('Browser audit complete:',len(results),'turns,',len(failures),'failed turns;',len(browser_errors),'JS errors',flush=True)
 if failures or browser_errors:raise SystemExit(1)
if __name__=='__main__':asyncio.run(main())
