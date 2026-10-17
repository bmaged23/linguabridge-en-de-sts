"""Reproduce language-format and length failures using synthetic dialogue only."""
import asyncio,json,tempfile
from pathlib import Path
import httpx
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.memory import Memory
from voice_bridge.routing import build_messages
from voice_bridge.speech_segments import LanguageSegments

CASES=[
 ('hello','en','Hello. Answer in one short sentence.'),
 ('coffee_lesson','en','Teach me how to order coffee in German, with two German examples and an English explanation.'),
 ('mixed_repeat','en','Repeat exactly: You can say Guten Morgen to greet someone.'),
 ('german_lesson','de','Erkläre mir bitte den Unterschied zwischen borrow und lend, mit einem englischen Beispiel.'),
 ('five_pairs','en','Give me five pairs of German phrases and their English translations, no introduction.'),
 ('long_lesson','en','Teach me ten useful German sentences for the train station and explain each one in English.'),
 ('isolated_vocabulary','en','Explain the German words obwohl and weil in English, and give one German example for each.'),
 ('continue','en','Keep going with the next example.'),
 ('recovery','en','I am still here. Please reply briefly.'),
]

async def main():
 c=load_settings();llm=c['llm'];report=[]
 with tempfile.TemporaryDirectory() as directory:
  m=Memory(Path(directory)/'memory.sqlite3',c['memory'])
  async with httpx.AsyncClient(timeout=45) as client:
   for name,source,text in CASES:
    parser=LanguageSegments();raw='';plain='';error=None;reason=None
    messages=build_messages(c,m,'conversation',source,source,text)
    async with client.stream('POST',llm['base_url']+'/api/chat',headers={'Authorization':'Bearer '+llm['api_key']},json={'model':llm['model'],'messages':messages,'stream':True,'think':False,'options':{'temperature':llm['temperature'],'num_predict':llm['max_response_tokens']}}) as response:
     response.raise_for_status()
     async for line in response.aiter_lines():
      if not line:continue
      packet=json.loads(line);part=packet.get('message',{}).get('content','');raw+=part
      if error is None:
       try:plain+=''.join(value for _,value,_ in parser.push(part))
       except ValueError as e:error=str(e)
      if packet.get('done'):reason=packet.get('done_reason')
    if error is None:
     try:parser.push(final=True)
     except ValueError as e:error=str(e)
    row={'case':name,'source':source,'prompt':text,'raw':raw,'parser_error':error,'done_reason':reason}
    report.append(row);print(name,'parser:',error or 'PASS','finish:',reason,flush=True)
    turn=m.add_turn('conversation',source,source,text)
    if plain:m.append_heard(turn,plain)
    (ROOT/'logs/cloud-protocol-diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  m.close()
if __name__=='__main__':asyncio.run(main())
