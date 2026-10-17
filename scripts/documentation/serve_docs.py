"""Preview documentation with HTTP range support for video seeking; no models loaded."""
import argparse
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from voice_bridge.settings import ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8768)
    args=parser.parse_args()
    app=Starlette(routes=[Mount('/',app=StaticFiles(directory=ROOT/'docs',html=True))])
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
