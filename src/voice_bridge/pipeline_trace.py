"""Opt-in local QA traces. Disabled in normal sessions; never logs credentials."""
import json
from pathlib import Path
import time
import wave


class PipelineTrace:
    def __init__(self,settings,root,identity):
        self.enabled=settings['enabled'];self.started=time.monotonic()
        if self.enabled:
            self.folder=Path(root)/settings['directory']/identity
            self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
            self.path=self.folder/'events.jsonl'
            self.path.touch(mode=0o600,exist_ok=True)

    def event(self,kind,**payload):
        if self.enabled:
            with self.path.open('a') as handle:
                handle.write(json.dumps({'seconds':round(time.monotonic()-self.started,4),'monotonic_seconds':round(time.monotonic(),6),'event':kind,**payload},ensure_ascii=False)+'\n')

    def audio(self,identity,pcm,rate,event='tts_audio',**payload):
        if not self.enabled:return
        path=self.folder/f'{identity.replace(":","-")}.wav'
        with wave.open(str(path),'wb') as handle:
            handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate);handle.writeframes(pcm)
        path.chmod(0o600)
        self.event(event,id=identity,audio=path.name,sample_rate=rate,audio_seconds=round(len(pcm)/2/rate,4),**payload)
