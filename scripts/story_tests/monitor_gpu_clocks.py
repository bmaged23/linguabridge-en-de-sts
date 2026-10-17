"""Supplement an already-running audit with GPU clocks and power-state samples."""
import argparse,json,subprocess,time
from pathlib import Path
import psutil

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--pid',type=int,required=True);a=p.parse_args();folder=Path(a.folder)
 with (folder/'gpu-clocks-continuous.jsonl').open('a') as out:
  while psutil.pid_exists(a.pid):
   start=time.monotonic();case='unknown'
   try:
    with (folder/'resources.jsonl').open('rb') as resource:
     resource.seek(0,2);resource.seek(max(0,resource.tell()-8192));case=json.loads(resource.read().splitlines()[-1])['case']
   except (OSError,ValueError,IndexError):pass
   result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
   values=result.stdout.strip().split(',');row={'monotonic_seconds':start,'case':case,'raw':result.stdout.strip()}
   if len(values)==5:
    row['pstate']=values[0].strip()
    for key,value in zip(('gpu_sm_clock_mhz','gpu_memory_clock_mhz','gpu_util_pct','gpu_power_w'),values[1:]):
     try:row[key]=float(value)
     except ValueError:row[key]=None
   out.write(json.dumps(row)+'\n');out.flush();time.sleep(max(.1,1-(time.monotonic()-start)))
