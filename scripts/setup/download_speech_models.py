"""Download pinned bilingual speech assets, model cards, and SHA256 manifest."""
import hashlib
import json
from pathlib import Path
import runpy
import urllib.request
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    config=runpy.run_path(str(ROOT/'config.py'))['CONFIG'];models=ROOT/'models';models.mkdir(exist_ok=True)
    reference=ROOT/'docs/reference/model-manifest.json';manifest=models/'assets.json'
    previous=json.loads((manifest if manifest.exists() else reference).read_text()) if (manifest.exists() or reference.exists()) else {}
    api=HfApi();repo=config['stt']['model_repo']
    qwen=config['stt']['provider']=='qwen_asr'
    revision=config['stt']['revision'] if qwen else previous.get('whisper_revision') or api.model_info(repo).sha
    print('Downloading '+config['stt']['model']+'...',flush=True)
    snapshot_download(repo,revision=revision,local_dir=ROOT/config['paths']['stt_model'],
        allow_patterns=None if qwen else ['model.bin','config.json','tokenizer.json','vocabulary.*','preprocessor_config.json','README.md','LICENSE*'])
    piper_repo='rhasspy/piper-voices';piper_rev=previous.get('piper_revision') or api.model_info(piper_repo).sha
    for language,voice in config['tts']['voices'].items():
        if voice.get('provider')!='piper':continue
        name=voice['name'];locale,speaker,quality=name.split('-');lang=locale.split('_')[0]
        prefix=f'{lang}/{locale}/{speaker}/{quality}'
        target=ROOT/voice['model'];target.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ('.onnx','.onnx.json'):
            source=Path(hf_hub_download(piper_repo,f'{prefix}/{name}{suffix}',revision=piper_rev))
            import shutil
            shutil.copy2(source,target if suffix=='.onnx' else Path(str(target)+'.json'))
        card=Path(hf_hub_download(piper_repo,f'{prefix}/MODEL_CARD',revision=piper_rev))
        (target.parent/(name+'.MODEL_CARD')).write_bytes(card.read_bytes())
        print('Voice ready:',name,flush=True)
    kokoro_release=previous.get('kokoro_release','model-files-v1.1')
    request=urllib.request.Request('https://api.github.com/repos/thewh1teagle/kokoro-onnx/releases/tags/'+kokoro_release,headers={'User-Agent':'voice-bridge'})
    with urllib.request.urlopen(request,timeout=60) as response:release=json.load(response)
    for name in ('kokoro-v1.0.onnx','voices-v1.0.bin'):
        target=models/name;expected=previous.get('files',{}).get(name,{}).get('sha256')
        if target.exists() and expected and digest(target)==expected:continue
        asset=next(item for item in release['assets'] if item['name']==name)
        temporary=target.with_suffix('.part')
        with urllib.request.urlopen(asset['browser_download_url'],timeout=120) as response,temporary.open('wb') as output:
            while block:=response.read(8*1024*1024):output.write(block)
        if expected and digest(temporary)!=expected:raise ValueError('Kokoro checksum mismatch')
        temporary.replace(target)
    silero_rev=previous.get('silero_revision')
    if not silero_rev:
        request=urllib.request.Request('https://api.github.com/repos/snakers4/silero-vad/commits/master',headers={'User-Agent':'voice-bridge'})
        with urllib.request.urlopen(request,timeout=60) as response:silero_rev=json.load(response)['sha']
    url=f'https://raw.githubusercontent.com/snakers4/silero-vad/{silero_rev}/src/silero_vad/data/silero_vad.onnx'
    destination=ROOT/config['paths']['vad_model'];temp=destination.with_suffix('.part')
    with urllib.request.urlopen(url,timeout=90) as response:temp.write_bytes(response.read())
    temp.replace(destination)
    data={('qwen_repo' if qwen else 'whisper_repo'):repo,('qwen_revision' if qwen else 'whisper_revision'):revision,'piper_repo':piper_repo,'piper_revision':piper_rev,
          'silero_revision':silero_rev,'kokoro_release':kokoro_release,'files':{str(p.relative_to(models)):{'bytes':p.stat().st_size,'sha256':digest(p)}
             for p in models.rglob('*') if p.is_file() and '.cache' not in p.parts and p!=manifest}}
    for name,entry in previous.get('files',{}).items():
        if name in data['files'] and data['files'][name]['sha256']!=entry['sha256']:
            raise ValueError('Asset checksum differs from pinned manifest: '+name)
    manifest.write_text(json.dumps(data,indent=2)+'\n');reference.write_text(manifest.read_text())
    print('Assets verified. Manifest:',reference,flush=True)


if __name__=='__main__':main()
