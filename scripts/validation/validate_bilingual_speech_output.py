"""Real cloud → tagged stream → GPU voices, using a temporary memory database.

Input text is injected in place of STT to isolate output pronunciation routing.
Decoded audio is saved, not physically played. Existing user history is untouched.
"""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import wave
import numpy as np
from voice_bridge import server
from voice_bridge.settings import ROOT,load_settings

CASES=[
 ('english_only','en','Repeat exactly: I would like a cup of coffee, please.',['en']),
 ('german_only','de','Wiederhole genau: Ich hätte gern einen Kaffee, bitte.',['de']),
 ('english_german_english','en','Repeat exactly these three parts in order, keeping every word in its original language, no introduction: You can say / Guten Morgen / to greet someone.',['en','de','en']),
 ('german_english_german','de','Wiederhole genau diese drei Teile in dieser Reihenfolge, ohne Einleitung und ohne Übersetzung: Du kannst / Good morning / zur Begrüßung sagen.',['de','en','de']),
 ('complex_english_german_english','en','Repeat exactly these three parts, preserving their languages and order, without an introduction: Tell the receptionist / Mein Zug hat Verspätung und ich komme später an / then ask whether reception stays open.',['en','de','en']),
 ('complex_german_english_german','de','Wiederhole genau diese drei Teile, ohne Übersetzung oder Einleitung: Meine Kollegin hat gesagt / we should postpone the meeting until tomorrow / aber ich möchte zuerst den Kunden fragen.',['de','en','de']),
]


async def main():
    settings=load_settings();folder=ROOT/'docs/mixed-output-check';folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        settings['memory']['database']=str(Path(directory)/'memory.sqlite3')
        server.load_settings=lambda:settings
        results=[]
        async with server.lifespan(server.app):
            for name,source,text,expected in CASES:
                packets=[]
                class Socket:
                    async def send_json(self,packet):
                        packets.append(packet)
                        if packet['type']=='audio':voice.acknowledge({'id':packet['id'],'complete':True})
                voice=server.Session(Socket())
                server.transcribe=lambda *args:{'text':text,'language':source,'certain':True}
                await asyncio.wait_for(voice.response(np.zeros(16000,np.float32),0,0),60)
                errors=[p['text'] for p in packets if p['type']=='error'];assert not errors,errors
                audio=[p for p in packets if p['type']=='audio']
                languages=[p['language'] for p in audio]
                runs=[language for i,language in enumerate(languages) if i==0 or language!=languages[i-1]]
                plain=''.join(p['text'] for p in packets if p['type']=='token')
                assert runs==expected,(name,runs,plain)
                assert '[' not in plain and ']' not in plain,plain
                pieces=[]
                for packet in audio:
                    a=np.frombuffer(base64.b64decode(packet['pcm']),dtype='<i2').astype(np.float32)/32768
                    rate=packet['sample_rate'];a=np.interp(np.arange(0,len(a),rate/24000),np.arange(len(a)),a)
                    pieces.append(a)
                combined=np.concatenate(pieces)
                with wave.open(str(folder/f'{name}.wav'),'wb') as handle:
                    handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(24000)
                    handle.writeframes((np.clip(combined,-1,1)*32767).astype('<i2').tobytes())
                results.append({'case':name,'input':text,'reply':plain,'languages':languages,'language_runs':runs,
                    'audio_seconds':round(len(combined)/24000,2),'audio':name+'.wav'})
                print(name,'PASS',runs,flush=True)
                await voice.close()
        (folder/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        import html
        cards=''.join(f'<section><h2>{r["case"]}</h2><p>{html.escape(r["reply"])}</p><p>{" → ".join(r["language_runs"])}</p><audio controls preload="none" src="{r["audio"]}"></audio></section>' for r in results)
        (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bridge bilingual output check</title><style>body{font:16px system-ui;background:#f4f6f1;color:#20382c;max-width:900px;margin:40px auto;padding:20px}section{background:white;border-radius:16px;padding:24px;margin:20px 0}audio{width:100%}</style><h1>English ⇄ German output</h1><p>Real cloud replies and GPU voices. Each segment uses its own language voice. Synthetic tests, decoded audio; physical listening remains manual.</p>'+cards)


if __name__=='__main__':asyncio.run(main())
