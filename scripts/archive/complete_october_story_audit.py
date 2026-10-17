"""Complete the extended story suite and same-audio recognizer comparisons."""
import json,os,signal,subprocess,time
from pathlib import Path
import httpx,psutil
from voice_bridge.settings import ROOT

BASE=ROOT/'logs/advanced-story/story-20261017'
EXTENDED=ROOT/'logs/advanced-story/story-extended-20261017'


def ready(port):
 try:return httpx.get(f'http://127.0.0.1:{port}/health',timeout=1).status_code==200
 except httpx.HTTPError:return False


def command(args):
 print('RUN',' '.join(args),flush=True)
 subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,check=True)


def stop_normal():
 for p in psutil.process_iter(['cmdline','cwd']):
  try:
   text=' '.join(p.info['cmdline'] or [])
   if p.info['cwd']==str(ROOT) and 'uvicorn' in (p.info['cmdline'] or []) and 'voice_bridge.server:app' in (p.info['cmdline'] or []) and '8766' in (p.info['cmdline'] or []):p.send_signal(signal.SIGINT);p.wait(timeout=30)
  except (psutil.Error,OSError):pass


if __name__=='__main__':
 print('Waiting for the 40-run base story audit and its generated-audio checks.',flush=True)
 while True:
  try:
   data=json.loads((BASE/'results.json').read_text())
   if len(data['samples'])==40 and all('tts_audio_audit' in s for s in data['samples']) and ready(8766) and not ready(8767):break
  except (OSError,ValueError):pass
  time.sleep(3)
 command(['-m', 'scripts.reports.export_story_report',str(BASE.relative_to(ROOT))])
 command(['-m', 'scripts.story_tests.run_story_pipeline_tests','--folder',str(EXTENDED.relative_to(ROOT))])
 command(['-m', 'scripts.reports.export_story_report',str(EXTENDED.relative_to(ROOT))])
 deadline=time.monotonic()+180
 while not ready(8766):
  if time.monotonic()>deadline:raise TimeoutError('Normal server restart before comparison timed out')
  time.sleep(.5)
 stop_normal()
 try:
  for folder in (BASE,EXTENDED):
   for model in ('qwen-fresh','large-v3','turbo','qwen-replay'):
    args=['-m', 'scripts.application.start_voice_app','-m', 'scripts.story_tests.compare_story_recognizers',str(folder.relative_to(ROOT)),'--model',model]
    if model=='qwen-replay':args.append('--only-failures')
    command(args)
   command(['-m', 'scripts.reports.export_story_report',str(folder.relative_to(ROOT))])
 finally:
  log=(ROOT/'logs/normal-after-story.log').open('a')
  subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m', 'scripts.application.start_voice_app'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  log.close()
 print('Completed both story suites, audio checks and recognizer comparisons.',flush=True)
