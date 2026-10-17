"""Validate the local recorded media and real browser video playback, without cloud calls."""
import asyncio,json,os,subprocess,wave
from pathlib import Path
import httpx,numpy as np
if Path('/mnt/d/voice-bridge-tools/playwright').is_dir():os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/mnt/d/voice-bridge-tools/playwright')
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT

async def main():
    media=ROOT/'docs/demo';recording=json.loads((media/'recording.json').read_text());results={}
    for suffix in ('mp4','webm'):
        path=media/f'bridge-demo.{suffix}'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]))
        assert {'video','audio'}<={s['codec_type'] for s in probe['streams']}
        duration=float(probe['format']['duration']);assert abs(duration-recording['duration_seconds'])<.3
        assert path.stat().st_size<100*1024*1024,'File exceeds GitHub regular-file limit'
        # Decode the whole audio stream, not just media metadata.
        decoded=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-vn','-f','s16le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
        assert not decoded.stderr,decoded.stderr.decode(errors='replace')
        pcm=decoded.stdout
        audio=np.frombuffer(pcm,'<i2').astype(np.float32)/32768
        rms=float(np.sqrt(np.mean(audio**2)));assert rms>.005,'Soundtrack is silent'
        blocks=[float(np.sqrt(np.mean(a*a))) for a in np.array_split(audio,max(1,int(duration)))]
        results[suffix]={'seconds':duration,'bytes':path.stat().st_size,'codecs':[s['codec_name'] for s in probe['streams']],'audio_rms':rms,'audible_one_second_blocks':sum(b>.005 for b in blocks)}
    assert (media/'captions.vtt').read_text().startswith('WEBVTT')
    assert len(json.loads((media/'transcript.json').read_text()))>=20
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(args=['--no-sandbox','--autoplay-policy=no-user-gesture-required'])
        page=await browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.goto('http://127.0.0.1:8766/docs/demo.html',wait_until='networkidle')
        await page.evaluate("document.querySelector('video').muted=true")
        await page.locator('#play').click()
        await page.wait_for_function("document.querySelector('video').currentTime>1 && !document.querySelector('video').paused",timeout=30000)
        assert await page.locator('#play').is_hidden()
        results['browser']={'source':await page.evaluate("document.querySelector('video').currentSrc.split('/').pop()"),'duration':await page.evaluate("document.querySelector('video').duration"),'playing_confirmed':True}
        await page.evaluate("document.querySelector('video').currentTime=document.querySelector('video').duration-1")
        await page.wait_for_function("document.querySelector('video').ended",timeout=15000)
        assert await page.locator('#play').is_visible()
        results['browser']['ended_confirmed']=True
        for width in (390,1440):
            await page.set_viewport_size({'width':width,'height':900})
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        await browser.close()
    assert not errors,errors
    results['browser_errors']=errors
    (media/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':asyncio.run(main())
