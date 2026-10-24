"""Sample the complete local server process tree and the whole GPU, explicitly scoped."""
import json,shutil,subprocess,threading,time
import psutil

class Resources:
 def __init__(self,pid,path,interval=.5):
  self.pid=pid;self.path=path;self.interval=interval;self.label='startup';self.rows=[];self.stop_event=threading.Event();self.processes={}
  self.gpu=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
  self.thread=threading.Thread(target=self.run,daemon=True)
 def start(self):self.thread.start()
 def stop(self):self.stop_event.set();self.thread.join(timeout=5)
 def run(self):
  with self.path.open('a') as out:
   while not self.stop_event.is_set():
    started=time.monotonic();row={'monotonic_seconds':started,'case':self.label,'processes':[]}
    try:
     root=psutil.Process(self.pid)
     for process in [root,*root.children(recursive=True)]:
      try:
       if process.pid not in self.processes:self.processes[process.pid]=process;process.cpu_percent()
       p=self.processes[process.pid];row['processes'].append({'pid':p.pid,'rss_mb':round(p.memory_info().rss/1048576,2),'cpu_pct_one_core':p.cpu_percent(),'role':'stt' if any(name in ' '.join(p.cmdline()) for name in ('qwen_worker.py','qwen_asr_worker.py')) else 'tts' if 'spawn_main' in ' '.join(p.cmdline()) else 'server'})
      except (psutil.Error,OSError):pass
    except psutil.Error:pass
    row['ram_tree_rss_mb']=round(sum(p['rss_mb'] for p in row['processes']),2)
    row['cpu_pct_one_core']=sum(p['cpu_pct_one_core'] for p in row['processes'])
    row['cpu_pct_machine']=round(row['cpu_pct_one_core']/max(psutil.cpu_count(),1),2)
    ram=psutil.virtual_memory();row['system_ram_used_mb']=round(ram.used/1048576,2)
    try:
     result=subprocess.run([self.gpu,'--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
     fields=result.stdout.strip().splitlines()[0].split(',')
     for key,value in zip(('gpu_vram_mb','gpu_total_mb','gpu_util_pct','gpu_memory_util_pct','gpu_power_w','gpu_temperature_c','gpu_sm_clock_mhz','gpu_memory_clock_mhz','gpu_power_limit_w'),fields):
      try:row[key]=float(value.strip())
      except ValueError:row[key]=None
    except (OSError,subprocess.SubprocessError,IndexError) as error:row['gpu_error']=type(error).__name__
    self.rows.append(row);out.write(json.dumps(row)+'\n');out.flush()
    self.stop_event.wait(max(.01,self.interval-(time.monotonic()-started)))
 def summarize(self,start,end):
  rows=[r for r in self.rows if start<=r['monotonic_seconds']<=end];summary={'sample_count':len(rows)}
  for key in ('ram_tree_rss_mb','cpu_pct_one_core','cpu_pct_machine','gpu_vram_mb','gpu_util_pct','system_ram_used_mb'):
   values=[r[key] for r in rows if isinstance(r.get(key),(int,float))]
   summary[key]={'peak':round(max(values),2),'mean':round(sum(values)/len(values),2)} if values else None
  return summary
