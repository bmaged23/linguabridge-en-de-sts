"""Small real cloud translation check. Does not expose credentials."""
import asyncio
import httpx
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.memory import Memory
from voice_bridge.routing import build_messages


async def main():
    import tempfile
    config=load_settings();settings=config['llm']
    with tempfile.TemporaryDirectory() as directory:
        memory=Memory(directory+'/memory.sqlite3',config['memory'])
        async with httpx.AsyncClient(timeout=90) as client:
            for source,target,text,expected in [('en','de','Where is the train station?','bahnhof'),('de','en','Das Treffen beginnt um drei Uhr.','three')]:
                messages=build_messages(config,memory,'translation',source,target,text)
                response=await client.post(settings['base_url']+'/api/chat',headers={'Authorization':'Bearer '+settings['api_key']},
                    json={'model':settings['model'],'messages':messages,'stream':False,'think':False,'options':{'temperature':0,'num_predict':128}})
                assert response.is_success, f'Cloud HTTP {response.status_code}'
                result=response.json()['message']['content'];print(source+' → '+target+': '+result)
                assert expected in result.lower() or (expected=='three' and '3' in result),result
        memory.close()
    print('Cloud bidirectional translation: PASS')


if __name__=='__main__':asyncio.run(main())
