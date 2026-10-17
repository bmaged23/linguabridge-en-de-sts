"""Run the real app with an isolated synthetic demo memory and local trace."""
import os
import uvicorn
from voice_bridge import server
from voice_bridge.settings import ROOT, load_settings

if __name__ == '__main__':
    settings=load_settings()
    folder=ROOT/os.environ['VOICE_BRIDGE_DEMO_DIR']
    settings['server']['port']=8767
    settings['memory']['database']=str(folder/'memory.sqlite3')
    settings['logging']['pipeline_trace']={'enabled':True,'directory':str((folder/'pipeline-traces').relative_to(ROOT))}
    server.load_settings=lambda:settings
    uvicorn.run(server.app,host='127.0.0.1',port=8767,log_level='info',access_log=False)
