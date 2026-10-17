"""Create the clickable demo cover, including the play button inside the image."""
import asyncio,base64,json,os
from pathlib import Path
if Path('/mnt/d/voice-bridge-tools/playwright').is_dir():os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/mnt/d/voice-bridge-tools/playwright')
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT

async def main():
    media=ROOT/'docs/demo';image=base64.b64encode((media/'poster.png').read_bytes()).decode()
    duration=round(json.loads((media/'recording.json').read_text())['duration_seconds']);label=f'{duration//60}:{duration%60:02}'
    html='''<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;width:1600px;height:900px;background:#11251e;color:#f5f8ef;font-family:Arial,sans-serif}.screen{position:relative;width:1600px;height:900px;overflow:hidden;background:radial-gradient(ellipse at 16% 40%,#2b4b36,#102219 65%)}.left{position:absolute;left:58px;top:65px;width:480px}.brand{font-size:62px;font-weight:800;letter-spacing:-4px}.tag{font-size:16px;letter-spacing:3px;color:#bbdda1;margin-top:16px}h1{font-size:63px;line-height:1.06;letter-spacing:-3px;margin:82px 0 28px}p{font-size:25px;line-height:1.55;color:#d3ddcc;margin:0}.play{margin-top:45px;width:377px;padding:24px 25px;background:#cfedac;color:#163322;border-radius:15px;display:flex;align-items:center;gap:18px;font-size:23px;font-weight:800;box-shadow:0 12px 40px #0003}.triangle{width:0;height:0;border-top:12px solid transparent;border-bottom:12px solid transparent;border-left:20px solid #163322}.meta{font-size:15px;color:#a8bda3;letter-spacing:1px;margin-top:20px}.preview{position:absolute;left:565px;top:90px;width:980px;height:716px;border-radius:20px;overflow:hidden;border:1px solid #ffffff25;box-shadow:0 25px 80px #0007}.preview img{width:100%;height:100%;object-fit:cover;object-position:top}.badge{position:absolute;right:40px;bottom:25px;font-size:14px;letter-spacing:2px;color:#bcd0b2}
</style></head><body><div class="screen"><div class="left"><div class="brand">LinguaBridge</div><div class="tag">SPEECH-TO-SPEECH · EN ⇄ DE</div><h1>One Saturday.<br>Two languages.</h1><p>A shared plan.<br>A conversation that remembers.</p><div class="play"><span class="triangle"></span>WATCH THE DEMO</div><div class="meta">DURATION · REAL PIPELINE · SYNTHETIC SPEAKERS</div></div><div class="preview"><img src="data:image/png;base64,IMAGE"></div><div class="badge">LISTEN · TRANSLATE · REMEMBER</div></div></body></html>'''.replace('IMAGE',image).replace('DURATION',label)
    async with async_playwright() as p:
        browser=await p.chromium.launch(args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1600,'height':900},device_scale_factor=1)
        await page.set_content(html,wait_until='load');await page.screenshot(path=str(media/'demo-cover.png'))
        await browser.close()
    print('Rendered',media/'demo-cover.png')

if __name__=='__main__':asyncio.run(main())
