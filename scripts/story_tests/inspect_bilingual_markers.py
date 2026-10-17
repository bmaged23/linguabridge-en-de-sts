"""Test output-format recovery on the actual unmarked synthetic story replies."""
import asyncio,hashlib,json,re,time
import httpx
from voice_bridge.settings import ROOT,load_settings


def normalized_space(text):return re.sub(r'\s+','',text)


async def main():
 folder=ROOT/'logs/advanced-story/story-extended-20261017';data=json.loads((folder/'results.json').read_text());settings=load_settings()['llm'];results=[]
 async with httpx.AsyncClient(timeout=60) as client:
  for row in data['samples']:
   raw=row['raw_llm']
   if not raw.strip() or re.search(r'\[(en|de)\]',raw):continue
   # A separate task tags the existing answer; it does not regenerate the answer.
   messages=[{'role':'system','content':'Split this English/German text into ordered monolingual spans. Return only a JSON object with a segments array. Each item has language (en or de) and text. Preserve every original word and punctuation in order. Do not translate, rewrite, summarize or add words. Keep whitespace where possible. Change span language whenever the spoken language changes, including foreign quotations. Punctuation belongs to its neighboring words. Example: {"segments":[{"language":"en","text":"You can say "},{"language":"de","text":"Guten Morgen"},{"language":"en","text":" to greet someone."}]}'},{'role':'user','content':raw}]
   started=time.monotonic();response=await client.post(settings['base_url']+'/api/chat',headers={'Authorization':'Bearer '+settings['api_key']},json={'model':settings['model'],'messages':messages,'stream':False,'think':False,'options':{'temperature':0,'num_predict':1024}});response.raise_for_status();packet=response.json();answer=packet['message']['content'];seconds=time.monotonic()-started
   try:
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',answer.strip());segments=json.loads(text)['segments'];joined=''.join(s['text'] for s in segments)
    valid=all(s['language'] in ('en','de') and isinstance(s['text'],str) for s in segments)
    result={'case':row['case']['id'],'method':'Separate labeling-only request on an actual unmarked answer; not deployed in production.','source':raw,'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'seconds':round(seconds,3),'raw_labeling_output':answer,'segments':segments,'valid_schema':valid,'exact_character_preservation':joined==raw,'nonwhitespace_preservation':normalized_space(joined)==normalized_space(raw),'languages':[s['language'] for s in segments],'done_reason':packet.get('done_reason')}
   except (ValueError,KeyError,TypeError) as error:result={'case':row['case']['id'],'seconds':seconds,'raw_labeling_output':answer,'error':str(error)}
   results.append(result);(folder/'marker-recovery-diagnostic.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(row['case']['id'],'labeling seconds',round(seconds,3),'preserved',result.get('nonwhitespace_preservation'),flush=True)
if __name__=='__main__':asyncio.run(main())
