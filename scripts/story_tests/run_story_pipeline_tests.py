"""Continuous-story browser test with raw pipeline artifacts and sampled resources.

Uses only synthetic inputs, a separate QA DB, and a silent real-time WebAudio sink.
Starts the isolated QA server; restores the normal app after GPU tests finish.
"""
import argparse,asyncio,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import httpx,psutil
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT
from scripts.validation.validate_browser_interface import BOOTSTRAP
from scripts.validation.run_browser_pipeline_tests import EXTRA
from scripts.evaluation.evaluate_mixed_language_asr import metrics
from scripts.story_tests.resource_monitor import Resources


def trace_rows(folder):
 return [json.loads(line) for p in sorted((folder/'pipeline-traces').glob('*/events.jsonl')) for line in p.read_text().splitlines()]


def analysis(case,events,acks,trace,start,end,monitor):
 finals=[e for e in events if e['type']=='final'];text=' '.join(e['text'] for e in finals)
 audio=[e for e in events if e['type']=='audio'];complete=[e for e in acks if e['type']=='played' and e.get('complete')]
 flags=[]
 if len(finals)>1:flags.append('Story split into multiple turns before input ended')
 if not finals:flags.append('No final transcription')
 if len(complete)!=len(audio):flags.append('Some generated audio did not finish playback')
 errors=[e['text'] for e in events if e['type']=='error']
 if errors:flags.extend(errors)
 m=metrics(case['reference'],text)
 if m['wer']>.15:flags.append('WER exceeds 15 percent')
 native=[e for e in trace if e['event']=='tts_audio']
 for e in native:
  if not any(c.isalnum() for c in e['text']):flags.append('Punctuation-only TTS chunk')
 raw=''.join(e['text'] for e in trace if e['event']=='llm_raw')
 timings={'whole_sample_seconds':round(end-start,3),'input_seconds':case['seconds'],'stages':[]}
 for index,e in enumerate(trace):
  stage=None;finish=None
  if e['event']=='stt_input':stage='final_stt';finish=next((x for x in trace[index+1:] if x['event']=='stt_final' and x.get('turn')==e.get('turn')),None)
  elif e['event']=='llm_messages':stage='llm_to_first_text';finish=next((x for x in trace[index+1:] if x['event']=='llm_raw' and x.get('text')),None)
  elif e['event']=='tts_input':stage='tts_inference';finish=next((x for x in trace[index+1:] if x['event']=='tts_complete'),None)
  if stage and finish:
   a=e['monotonic_seconds'];b=finish['monotonic_seconds'];timings['stages'].append({'stage':stage,'seconds':round(b-a,4),'turn':e.get('turn'),'resources':monitor.summarize(a,b),'decode_seconds':finish.get('decode_seconds')})
 first_audio=next((e for e in events if e['type']=='audio'),None)
 if first_audio:timings['first_audio_from_input_start_seconds']=round(first_audio['at_ms']/1000-start,3)
 return {'case':case,'stt_final':text,'stt_metrics':m,'stt_partials':[e for e in events if e['type']=='partial'],'raw_llm':raw,'reply':''.join(e['text'] for e in events if e['type']=='token'),'audio_chunks':len(audio),'completed_playback_chunks':len(complete),'native_audio':native,'errors':errors,'review_flags':flags,'timings':timings,'resources':monitor.summarize(start,end),'browser_events':events,'playback_acks':acks,'pipeline_trace':trace}


async def run(folder,monitor,limit,resume=False):
 cases=json.loads((folder/'cases.json').read_text());results=[];js_errors=[]
 if resume and (folder/'results.json').exists():
  saved=json.loads((folder/'results.json').read_text());results=saved['samples'];js_errors=saved.get('javascript_errors',[]);done={r['case']['id'] for r in results};cases=[c for c in cases if c['id'] not in done]
 async with async_playwright() as p:
  browser=await p.chromium.launch(args=['--no-sandbox','--disable-gpu','--autoplay-policy=no-user-gesture-required'])
  context=await browser.new_context(viewport={'width':1400,'height':1050})
  bootstrap=BOOTSTRAP.replace('super(...args);window.__ctx=this;',"super({sinkId:{type:'none'}});window.__ctx=this;").replace('window.__events.push(safe)','window.__events.push({...safe,at_ms:performance.now()})')
  await context.add_init_script(bootstrap+EXTRA.replace('window.__out.push(JSON.parse(data))','window.__out.push({...JSON.parse(data),at_ms:performance.now()})'))
  page=await context.new_page();page.on('pageerror',lambda e:js_errors.append(str(e)))
  async def route(request):
   name=request.request.url.rsplit('/',1)[-1];path=folder/'inputs'/name
   if path.is_file() and path.suffix=='.wav':await request.fulfill(path=str(path),content_type='audio/wav')
   else:await request.abort()
  await page.route('**/__story__/*',route)
  await page.goto('http://127.0.0.1:8767',wait_until='networkidle')
  # Explicit synthetic memory, persisted across the connected story runs.
  await page.locator('#memory-card summary').click();await page.locator('#memory-note').fill('Synthetic QA: Anna and Bola are travelling to Berlin. My German level is A2.');await page.get_by_role('button',name='Remember',exact=True).click()
  async def start():
   await page.get_by_role('button',name='Start conversation',exact=True).click();await page.wait_for_function("active && context.state==='running'",timeout=30000)
  await start()
  # Browser event time is relative to navigation; convert to the host monotonic clock.
  offset=time.monotonic()-await page.evaluate('performance.now()/1000')
  for case in cases[:limit or None]:
   monitor.label=case['id'];await page.wait_for_function('!playing && !queue.length && !responsePending',timeout=120000);await page.wait_for_timeout(500)
   before=await page.evaluate('__events.length');sent=await page.evaluate('__out.length');trace_before=len(trace_rows(folder));started=time.monotonic();failure=None
   try:
    await asyncio.wait_for(page.evaluate('url=>__say(url)',f'http://127.0.0.1:8767/__story__/{case["id"]}.wav'),case['seconds']+20)
    # Earlier story fragments may already have produced audio. Let the final
    # microphone silence reach VAD before accepting an idle browser as done.
    await page.wait_for_timeout(1700)
    await page.wait_for_function("n=>__events.slice(n).some(e=>e.type==='audio'||e.type==='error') && !playing && !queue.length && !responsePending && document.body.dataset.state!=='hearing'",arg=before,timeout=120000)
   except Exception as error:failure=f'{type(error).__name__}: {str(error)[:300]}'
   await page.wait_for_timeout(300);ended=time.monotonic()
   events=await page.evaluate('n=>__events.slice(n)',before);acks=await page.evaluate('n=>__out.slice(n)',sent)
   # Keep raw browser timestamps; use translated timestamps for latency calculations.
   for e in events:
    if 'at_ms' in e:e['at_ms']+=offset*1000
   trace=[e for e in trace_rows(folder) if started<=e['monotonic_seconds']<=ended];row=analysis(case,events,acks,trace,started,ended,monitor)
   if failure:row['review_flags'].append(failure)
   results.append(row);(folder/'samples').mkdir(exist_ok=True)
   (folder/'samples'/f'{case["id"]}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
   (folder/'results.json').write_text(json.dumps({'method':'Continuous synthetic story WAV per run; native voices change at language boundaries; silent real-time WebAudio output; not human microphone testing.','resource_scope':'Process-tree RSS sums include shared pages. CPU one-core percent can exceed 100. VRAM and utilization cover the whole GPU, including Windows/other apps. Sampled peaks are lower bounds. Stage peaks may have no samples for very short inference.','samples':results,'javascript_errors':js_errors},ensure_ascii=False,indent=2)+'\n')
   print(case['id'],f'WER={row["stt_metrics"]["wer"]:.1%}',f'{row["completed_playback_chunks"]}/{row["audio_chunks"]} audio',f'{ended-started:.1f}s',row['review_flags'],flush=True)
   if failure or not await page.evaluate('active'):
    if await page.evaluate('active'):await page.get_by_role('button',name='End conversation',exact=True).click()
    await start()
  await page.screenshot(path=str(folder/'final-screen.png'),full_page=True)
  await page.get_by_role('button',name='End conversation',exact=True).click();await browser.close()
 # Re-recognize each distinct generated phrase after browser timing has ended.
 # This is a model agreement diagnostic, not a human pronunciation score.
 cache={};owners={}
 for path in (folder/'pipeline-traces').glob('*/events.jsonl'):
  for line in path.read_text().splitlines():
   event=json.loads(line)
   if event['event']=='tts_audio':owners[event['monotonic_seconds']]=path.parent.name
 async with httpx.AsyncClient(timeout=120) as client:
  for row in results:
   monitor.label=row['case']['id']+':tts-audit';audits=[]
   for event in row['native_audio']:
    key=(event['language'],event['text'],hashlib.sha256((folder/'pipeline-traces'/owners[event['monotonic_seconds']]/event['audio']).read_bytes()).hexdigest())
    if key not in cache:
     try:
      response=await client.post('http://127.0.0.1:8767/__qa__/audit-audio',json={'session':owners[event['monotonic_seconds']],'audio':event['audio']});response.raise_for_status()
      result=response.json();cache[key]={'expected_language':event['language'],'reference':event['text'],'recognized':result['text'],'detected_language':result['language'],'metrics':metrics(event['text'],result['text']),'seconds':result['audit_seconds']}
     except Exception as error:cache[key]={'error':str(error)}
    audits.append({'id':event['id'],**cache[key]})
   row['tts_audio_audit']=audits
   (folder/'samples'/f'{row["case"]["id"]}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
   print('TTS audit',row['case']['id'],len(audits),'chunks',flush=True)
 data=json.loads((folder/'results.json').read_text());data['samples']=results
 (folder/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 return results


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--folder',default='logs/advanced-story/inputs-run');parser.add_argument('--limit',type=int,default=0);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 folder=ROOT/args.folder;folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'cases.json').exists():
  from scripts.story_tests.prepare_story_inputs import prepare
  prepare(folder)
 # Stop only this project's normal uvicorn, if it is running; preserve its data.
 for process in psutil.process_iter(['pid','cmdline','cwd']):
  try:
   command=' '.join(process.info['cmdline'] or [])
   if process.info['cwd']==str(ROOT) and 'uvicorn' in (process.info['cmdline'] or []) and 'voice_bridge.server:app' in (process.info['cmdline'] or []) and '8766' in (process.info['cmdline'] or []):
    process.send_signal(signal.SIGINT);process.wait(timeout=30)
  except (psutil.Error,OSError):pass
 env=os.environ.copy();env['VOICE_BRIDGE_QA_DIR']=str(folder.relative_to(ROOT));env['VOICE_BRIDGE_QA_AUDIT_FINAL']='0'
 log=(folder/'server.log').open('a');server=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m', 'scripts.application.start_voice_app','-m', 'scripts.validation.start_test_server'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 monitor=Resources(server.pid,folder/'resources.jsonl');monitor.start()
 try:
  deadline=time.monotonic()+180
  while True:
   if server.poll() is not None:raise RuntimeError('QA startup failed; inspect server.log')
   try:
    if httpx.get('http://127.0.0.1:8767/health',timeout=1).status_code==200:break
   except httpx.HTTPError:pass
   if time.monotonic()>deadline:raise TimeoutError('QA startup timeout')
   time.sleep(.5)
  asyncio.run(run(folder,monitor,args.limit,args.resume))
 finally:
  monitor.stop();os.killpg(server.pid,signal.SIGINT)
  try:server.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=10)
  log.close()
  normal=(ROOT/'logs/normal-after-story.log').open('a')
  subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m', 'scripts.application.start_voice_app'],cwd=ROOT,stdout=normal,stderr=subprocess.STDOUT,start_new_session=True)
  normal.close()
 print('Saved all sample artifacts:',folder,flush=True)
if __name__=='__main__':main()
