"""Build the HTML pipeline and deterministic animated GIF used directly in README."""
import argparse,asyncio,html,os,subprocess
from pathlib import Path
from voice_bridge.settings import ROOT,load_config
if Path('/mnt/d/voice-bridge-tools/playwright').is_dir():os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/mnt/d/voice-bridge-tools/playwright')


def build_html():
 c=load_config();esc=html.escape
 def node(identity,x,y,title,tag,lines,width=260,height=145):
  text=''.join(f'<text x="18" y="{82+i*23}" class="sub">{esc(line)}</text>' for i,line in enumerate(lines))
  return f'<g id="{identity}" class="node" transform="translate({x} {y})" tabindex="0" role="button"><rect width="{width}" height="{height}" rx="17"/><text x="18" y="25" class="tag">{esc(tag)}</text><text x="18" y="55" class="title">{esc(title)}</text>{text}<circle cx="0" cy="{height/2}" r="5" class="port"/><circle cx="{width}" cy="{height/2}" r="5" class="port"/><text x="12" y="{height-10}" class="port-label">IN</text><text x="{width-35}" y="{height-10}" class="port-label">OUT</text></g>'
 def edge(identity,path,label,x,y,kind='data'):
  return f'<path id="{identity}" d="{path}" class="edge {kind}"/><text x="{x}" y="{y}" text-anchor="middle" class="label">{esc(label)}</text><circle data-edge="{identity}" r="6" class="packet {kind}"><animateMotion dur="1.4s" repeatCount="indefinite"><mpath href="#{identity}"/></animateMotion></circle>'
 nodes=''.join([
  node('mic',55,140,'Microphone','BROWSER · AUDIOWORKLET',['IN: English / German / mixed speech','OUT: 16 kHz mono · 512 samples']),
  node('vad',425,140,'Silero VAD','LOCAL CPU · ENDPOINT',['IN: 32 ms audio frames','OUT: speech start + completed turn']),
  node('stt',795,140,'Qwen3-ASR 0.6B','LOCAL GPU · FP32 · EAGER',['IN: cumulative utterance audio','OUT: preview / final text + EN/DE']),
  node('memory',1165,140,'Mode + memory','LOCAL SQLITE · ROUTING',['IN: final text + saved settings','OUT: prompt + target + context']),
  node('llm',55,530,'Gemma 4 31B','OLLAMA CLOUD · THINKING OFF',['IN: final text + selected memory','OUT: streamed [en] / [de] text']),
  node('parser',425,530,'Parse + phrase buffer','LOCAL · BOUNDED QUEUES',['IN: language-tagged text stream','OUT: ordered phrases + language']),
  node('en',795,510,'Kokoro · English','LOCAL GPU · AF_HEART',['IN: English phrase','OUT: 24 kHz PCM16 audio'],280,125),
  node('de',795,675,'Piper · German','LOCAL GPU · THORSTEN HIGH',['IN: German phrase','OUT: 22.05 kHz PCM16 audio'],280,125),
  node('speaker',1190,570,'Ordered playback','BROWSER · WEBAUDIO',['IN: native-rate audio chunks','OUT: speech + played ACK'],280,145),
 ])
 edges=''.join([
  edge('audio','M315 212.5H425','PCM frames',370,190),
  edge('speech','M685 212.5H795','speech audio',740,190),
  edge('final','M1055 212.5H1165','final text',1110,190),
  edge('request','M1295 285V405H185V530','HTTPS: finalized text + mode + selected context (12 recent turns)',755,391),
  edge('tokens','M315 602.5H425','text stream',370,581),
  edge('english','M685 602.5H740V572.5H795','EN',758,548),
  edge('german','M685 602.5H740V737.5H795','DE',758,720),
  edge('en_audio','M1075 572.5H1130V642.5H1190','audio',1134,552),
  edge('de_audio','M1075 737.5H1130V642.5H1190','audio',1134,767),
  edge('preview','M925 285V325','previews + final → transcript',1055,310,'feedback'),
  edge('text_ui','M555 675V850H1525V350H1105','clean reply text → UI (markers removed)',1000,871,'feedback'),
  edge('ack','M1470 642.5H1550V80H1295V140','completed playback → save only heard phrases',1060,58,'feedback'),
  edge('cancel','M555 285V310H375V950H1330V715','New speech: stop browser playback first · cancel response · clear queues',930,981,'control'),
  edge('cancel_cloud','M375 950H185V675','cancel cloud request',200,918,'control'),
  edge('cancel_queue','M555 950V675','cancel phrase queue',554,918,'control'),
  edge('cancel_tts','M935 950V813','discard stale inference',943,918,'control'),
 ])
 style='''*{box-sizing:border-box}body{margin:0;padding:24px;background:#f3f6ef;color:#234534;font-family:system-ui,sans-serif}main{max-width:1600px;margin:auto}header{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:29px;letter-spacing:-1px}p{color:#65765e;line-height:1.65}button{background:white;border:1px solid #b7cba8;border-radius:10px;padding:12px 18px;color:#31563c;cursor:pointer}svg{display:block;width:100%;height:auto;background:#fcfdf9;border:1px solid #d4dfce;border-radius:20px}.node rect{fill:white;stroke:#c4d6ba;stroke-width:1.5}.node.active rect{fill:#e9f5dc;stroke:#4f8551;stroke-width:3}.node:focus{outline:none}.node:focus rect{stroke:#385a8c;stroke-width:3}.title{font-size:22px;font-weight:700;fill:#244e34}.tag{font-size:12px;font-weight:700;letter-spacing:.8px;fill:#667f57}.sub{font-size:13px;fill:#53694b}.port{fill:#659853}.port-label{font-size:10px;fill:#638156;font-weight:700}.edge{fill:none;stroke:#a0b68e;stroke-width:2.5;marker-end:url(#arrow)}.edge.feedback{stroke:#7ca3b6;marker-end:url(#feedbackArrow)}.edge.control{stroke:#c18754;stroke-dasharray:7 5;marker-end:url(#controlArrow)}.edge.hot{stroke-width:4;stroke:#397d4b}.edge.feedback.hot{stroke:#3c87a5}.edge.control.hot{stroke:#bd7137}.label{font-size:14px;fill:#4e6945;paint-order:stroke;stroke:#fcfdf9;stroke-width:7px;stroke-linejoin:round}.packet{fill:#44884e;opacity:0}.packet.feedback{fill:#3485a3}.packet.control{fill:#c07b43}.packet.hot{opacity:1}.section{font-size:15px;font-weight:700;letter-spacing:1.2px;fill:#6c805d}.caption{font-size:22px;font-weight:600;fill:#30583b}.note{font-size:15px;fill:#687b5c}.legend{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px;font-size:13px}.legend span:before{content:'●';color:#44884e;margin-right:8px}.legend .feedback:before{color:#3485a3}.legend .control:before{color:#c07b43}@media(max-width:600px){body{padding:12px}header{flex-wrap:wrap}h1{font-size:23px}}'''
 defs=''.join(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L9 5L1 9" fill="none" stroke="{color}" stroke-width="1.7"/></marker>' for name,color in [('arrow','#6b965c'),('feedbackArrow','#428da8'),('controlArrow','#bd7a43')])
 script='''const steps=[
 [['mic'],['audio'],'1 · Capture English, German or mixed speech in clocked 32 ms frames'],
 [['vad'],['speech'],'2 · Silero detects speech and waits for the mode-specific silence threshold'],
 [['stt'],['final','preview'],'3 · Qwen streams replaceable previews; only the final transcript enters the LLM'],
 [['memory'],['request'],'4 · Confirmed mode, target language and persistent memory shape the request'],
 [['llm','parser'],['tokens','text_ui'],'5 · Cloud tokens stream through the language parser and bounded phrase buffer'],
 [['parser','en'],['english','en_audio'],'6 · English phrases use Kokoro; synthesis overlaps earlier playback'],
 [['parser','de','speaker'],['german','de_audio'],'7 · German phrases use Piper; language switches preserve audio order'],
 [['speaker','memory'],['ack'],'8 · Completed-playback acknowledgments save what was actually heard'],
 [['vad','llm','parser','en','de','speaker'],['cancel','cancel_cloud','cancel_queue','cancel_tts'],'9 · Confirmed new speech interrupts playback and invalidates old results']
 ];let current=0,paused=false;const svg=document.querySelector('svg');
 function step(index){current=index;document.querySelectorAll('.node').forEach(n=>n.classList.toggle('active',steps[index][0].includes(n.id)));document.querySelectorAll('.edge').forEach(e=>e.classList.toggle('hot',steps[index][1].includes(e.id)));document.querySelectorAll('.packet').forEach(p=>p.classList.toggle('hot',steps[index][1].includes(p.dataset.edge)));document.getElementById('caption').textContent=steps[index][2];}
 setInterval(()=>{if(!paused)step((current+1)%steps.length)},1400);step(0);
 document.getElementById('toggle').onclick=()=>{paused=!paused;paused?svg.pauseAnimations():svg.unpauseAnimations();document.getElementById('toggle').textContent=paused?'Resume animation':'Pause animation';};
 document.querySelectorAll('.node').forEach(node=>{const index=steps.findIndex(s=>s[0].includes(node.id));node.style.cursor='pointer';node.setAttribute('aria-label',steps[index][2]);node.onclick=()=>step(index);node.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();step(index);}};});
 if(matchMedia('(prefers-reduced-motion: reduce)').matches)document.getElementById('toggle').click();'''
 content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LinguaBridge STS · Animated speech-to-speech pipeline</title><style>{style}</style></head><body><main><header><div><h1>Speech in. Understanding. Speech out.</h1><p>LinguaBridge STS · English–German · real inputs, outputs, streaming and interruption control</p></div><button id="toggle">Pause animation</button></header>
 <svg viewBox="0 0 1600 1160" role="img" aria-labelledby="pipeline-title pipeline-desc"><title id="pipeline-title">Animated English–German speech-to-speech pipeline</title><desc id="pipeline-desc">Browser microphone to local Silero VAD and GPU Qwen recognition, persistent mode and memory, cloud Gemma, language-tagged phrases, separate English and German GPU voices, and ordered browser playback. Playback acknowledgments update history; new speech cancels replies.</desc><defs>{defs}</defs>
 <text x="55" y="115" class="section">01 / LISTEN + UNDERSTAND · LOCAL COMPUTER</text><text x="425" y="470" class="section">02 / STREAM + SPEAK</text>
 <rect x="35" y="487" width="300" height="210" rx="19" fill="#eff1fb" stroke="#b3bfdb" stroke-dasharray="6 5"/><text x="55" y="510" class="tag">REMOTE · TEXT ONLY</text>
 <rect x="775" y="475" width="320" height="340" rx="19" fill="#f0f5e9" stroke="#bacfa9"/><text x="795" y="496" class="tag">ONE SHARED GPU TTS PROCESS</text>
 {edges}{nodes}<rect x="805" y="325" width="300" height="52" rx="12" fill="#edf4f7" stroke="#b2cbd5"/><text x="955" y="348" text-anchor="middle" class="tag">LIVE BROWSER TRANSCRIPT</text><text x="955" y="366" text-anchor="middle" class="sub">Previews + final text + clean replies</text>
 <text x="55" y="1035" class="note">Conversation: {c['modes']['conversation']['end_of_turn_silence_ms']} ms silence · Translation: {c['modes']['translation']['end_of_turn_silence_ms']} ms · Teaching: {c['modes']['teaching']['end_of_turn_silence_ms']} ms · no idle timeout between speakers</text>
 <text x="55" y="1063" class="note">STT previews ≥2 s · phrases 40–180 characters · interruption gate 320 ms · one browser session · one STT + one TTS process</text>
 <text id="caption" x="800" y="1115" text-anchor="middle" class="caption"></text></svg>
 <div class="legend"><span>Forward data</span><span class="feedback">Transcript + memory feedback</span><span class="control">Interruption control</span></div>
 <p>The animation shows data dependencies, not measured latency. Audio stays local; finalized text and selected context go to Ollama Cloud. STT is progressive cumulative decoding; TTS emits completed phrase chunks. Explicit app commands may update mode or notes locally without an LLM request.</p></main><script>{script}</script></body></html>'''
 (ROOT/'docs/architecture.html').write_text(content+'\n')


async def render():
 from playwright.async_api import async_playwright
 folder=ROOT/'logs/pipeline-animation-frames';folder.mkdir(parents=True,exist_ok=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1648,'height':1450},device_scale_factor=1)
  await page.goto((ROOT/'docs/architecture.html').as_uri())
  for index in range(76):
   await page.evaluate('''i=>{const s=document.querySelector('svg');s.pauseAnimations();s.setCurrentTime(i/6);paused=true;step(Math.floor(i/6/1.4)%steps.length)}''',index)
   await page.locator('svg').screenshot(path=str(folder/f'{index:03}.png'))
  await browser.close()
 target=ROOT/'docs/assets/pipeline-animation.gif'
 subprocess.run(['ffmpeg','-v','error','-y','-framerate','6','-i',str(folder/'%03d.png'),'-filter_complex','scale=1440:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer','-loop','0',str(target)],check=True)
 print('Animated README pipeline:',target,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--html-only',action='store_true');args=parser.parse_args();build_html()
 if not args.html_only:asyncio.run(render())
