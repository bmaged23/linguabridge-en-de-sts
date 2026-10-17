"""Validate decoded media and the real player lifecycle, including a failed MP4 source."""
import argparse,asyncio,json,os,subprocess
from pathlib import Path
import numpy as np
if Path('/mnt/d/voice-bridge-tools/playwright').is_dir():os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/mnt/d/voice-bridge-tools/playwright')
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8766/docs/')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/demo/validation.json')
    args=parser.parse_args();base=args.base_url.rstrip('/')+'/'
    media=ROOT/'docs/demo';recording=json.loads((media/'recording.json').read_text());results={}
    for suffix in ('mp4','webm'):
        path=media/f'bridge-demo.{suffix}'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]))
        assert {'video','audio'}<={s['codec_type'] for s in probe['streams']}
        duration=float(probe['format']['duration']);assert abs(duration-recording['duration_seconds'])<.3
        assert path.stat().st_size<100*1024*1024
        decoded=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-vn','-f','s16le','-ac','1','-ar','16000','pipe:1'],capture_output=True,check=True)
        assert not decoded.stderr,decoded.stderr.decode(errors='replace')
        audio=np.frombuffer(decoded.stdout,'<i2').astype(np.float32)/32768
        rms=float(np.sqrt(np.mean(audio**2)));assert rms>.005
        print('Decoded media:',suffix,flush=True)
        results[suffix]={'seconds':duration,'bytes':path.stat().st_size,'codecs':[s['codec_name'] for s in probe['streams']],'audio_rms':rms}
    assert (media/'captions.vtt').read_text().startswith('WEBVTT')
    assert len(json.loads((media/'transcript.json').read_text()))>=20
    errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        await page.goto(base+'demo.html',wait_until='networkidle')
        await page.locator('video').evaluate('(v)=>v.muted=true')
        await page.locator('#play').click()
        await page.wait_for_function("demo.currentTime>1 && !demo.paused && demo.getVideoPlaybackQuality().totalVideoFrames>5",timeout=30000)
        assert await page.locator('#play').is_hidden()
        source=await page.locator('video').evaluate('(v)=>v.currentSrc')
        assert 'bridge-demo.mp4' in source,source
        print('Primary MP4 plays with decoded frames',flush=True)
        await page.locator('video').evaluate('(v)=>v.pause()')
        await page.locator('#play').wait_for(state='visible')
        assert await page.locator('#play-label').inner_text()=='Resume demo'
        previous=await page.locator('video').evaluate('(v)=>v.currentTime')
        await page.locator('#play').click()
        await page.wait_for_function('t=>demo.currentTime>t+.3 && !demo.paused',arg=previous)
        print('Pause / resume passed',flush=True)
        await page.locator('video').evaluate('(v)=>v.currentTime=30')
        await page.wait_for_function('!demo.seeking && demo.currentTime>=30')
        print('Mid-video seeking passed',flush=True)
        await page.locator('video').evaluate('(v)=>v.currentTime=v.duration-1')
        await page.wait_for_function('demo.ended',timeout=15000)
        await page.locator('#play').wait_for(state='visible')
        results['browser']={'base_url':base,'primary_source':source.split('/')[-1],'duration':await page.locator('video').evaluate('(v)=>v.duration'),'decoded_frames':await page.locator('video').evaluate('(v)=>v.getVideoPlaybackQuality().totalVideoFrames'),'start_pause_resume_seek_end':True}
        for width in (390,1440):
            await page.set_viewport_size({'width':width,'height':900})
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        # Force a media-element network error rather than relying only on native
        # <source> selection. The player must recover with the real WebM file.
        fallback=await browser.new_page()
        fallback.on('pageerror',lambda e:errors.append(str(e)))
        await fallback.route('**/bridge-demo.mp4*',lambda route:route.abort('failed'))
        await fallback.goto(base+'demo.html',wait_until='networkidle')
        await fallback.locator('video').evaluate('(v)=>{v.muted=true;v.src=new URL("demo/bridge-demo.mp4?forced-failure=1",location.href).href;v.load()}')
        await fallback.wait_for_function('demo.currentSrc.includes("webm") && demo.readyState>=2',timeout=30000)
        await fallback.locator('#play').click()
        await fallback.wait_for_function('demo.currentTime>.4 && !demo.paused && demo.getVideoPlaybackQuality().totalVideoFrames>0',timeout=30000)
        results['browser']['failed_mp4_recovers_with_webm']=True
        await browser.close()
    assert not errors,errors
    results['browser_errors']=errors
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':asyncio.run(main())
