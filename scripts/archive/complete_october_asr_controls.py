"""Finish standard-window controls after the ongoing GPU audit releases the device."""
import json,signal,subprocess,time
import httpx,psutil
from voice_bridge.settings import ROOT


def ready():
 try:return httpx.get('http://127.0.0.1:8766/health',timeout=1).status_code==200
 except httpx.HTTPError:return False


if __name__=='__main__':
 jobs=[p for p in psutil.process_iter(['cmdline']) if 'scripts.archive.complete_october_story_audit' in (p.info['cmdline'] or [])]
 print('Waiting for the current recognizer comparisons before standard-window controls.',flush=True)
 while any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in jobs):time.sleep(3)
 deadline=time.monotonic()+180
 while not ready():
  if time.monotonic()>deadline:raise TimeoutError('Normal app was not restored by the preceding audit')
  time.sleep(.5)
 for p in psutil.process_iter(['cmdline','cwd']):
  try:
   args=p.info['cmdline'] or []
   if p.info['cwd']==str(ROOT) and 'uvicorn' in args and 'voice_bridge.server:app' in args and '8766' in args:p.send_signal(signal.SIGINT);p.wait(timeout=30)
  except (psutil.Error,OSError):pass
 try:
  for model in ('large-v3','turbo'):
   command=[str(ROOT/'.venv/bin/python'),'-m', 'scripts.application.start_voice_app','-m', 'scripts.story_tests.compare_story_recognizers','logs/advanced-story/story-20261017','--model',model,'--whisper-chunk','30']
   print('RUN standard 30s',model,flush=True);subprocess.run(command,cwd=ROOT,check=True)
  for name in ('story-20261017','story-extended-20261017'):subprocess.run([str(ROOT/'.venv/bin/python'),'-m', 'scripts.reports.export_story_report','logs/advanced-story/'+name],cwd=ROOT,check=True)
  subprocess.run([str(ROOT/'.venv/bin/python'),'-m', 'scripts.reports.summarize_story_results'],cwd=ROOT,check=True)
 finally:
  log=(ROOT/'logs/normal-after-story.log').open('a');subprocess.Popen([str(ROOT/'.venv/bin/python'),'-m', 'scripts.application.start_voice_app'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
 print('Final controls completed; normal app restarted.',flush=True)
