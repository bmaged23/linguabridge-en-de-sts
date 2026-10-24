"""Local stdio transport to the isolated, persistent CUDA ASR interpreter."""
import base64
import json
import os
from pathlib import Path
import selectors
import subprocess


class QwenBridge:
    def __init__(self, settings, root):
        self.settings = settings
        root = Path(root)
        self.root=root
        self.speech_detector=None
        interpreter = root / settings['runtime_python']
        if not interpreter.is_file():
            raise RuntimeError('Qwen runtime missing. See docs/guides/qwen-stt-deployment.md.')
        env = {k: v for k, v in os.environ.items() if not k.startswith('OLLAMA_')}
        libraries = list((interpreter.parent.parent / 'lib').glob('python*/site-packages/nvidia/*/lib'))
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))
        env['HF_HUB_OFFLINE'] = '1'
        self.process = subprocess.Popen([str(interpreter), str(root / 'scripts/runtime/qwen_asr_worker.py')],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self._exchange({'type': 'init', 'settings': settings, 'root': str(root)}, settings['startup_timeout_seconds'])
        except BaseException:
            self.close()
            raise

    def _exchange(self, packet, timeout):
        self.process.stdin.write(json.dumps(packet) + '\n')
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.close()
                raise TimeoutError('Qwen GPU worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Qwen GPU worker exited; check server logs')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def transcribe(self, audio, stream_id=None, final=True, decode_options=None):
        packet={'type': 'transcribe', 'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode(),
                'stream_id': stream_id, 'final': final}
        if decode_options is not None:packet['decode_options']=decode_options
        options=decode_options or {}
        segmentation=self.settings.get('final_segmentation',{})
        strategy=options.get('strategy',self.settings.get('final_strategy','whole'))
        caching=options.get('stream_cache',self.settings.get('stream_chunk_cache',False)) and stream_id is not None
        if ((final or caching) and strategy=='adaptive_pause_chunks' and
                len(audio)>=segmentation.get('minimum_total_seconds',8)*16000 and
                options.get('speech_gate',segmentation.get('use_speech_gate',False))):
            packet['speech_regions']=self.speech_regions(audio)
        return self._exchange(packet, self.settings['request_timeout_seconds'])

    def speech_regions(self,audio):
        import numpy as np
        from .workers import Silero
        settings=self.settings['final_segmentation']
        if self.speech_detector is None:
            self.speech_detector=Silero(self.root/settings['vad_model_path'],{
                'intra_op_threads':settings['vad_intra_op_threads'],
                'inter_op_threads':settings['vad_inter_op_threads']})
        self.speech_detector.reset()
        regions=[]
        for left in range(0,len(audio),512):
            frame=audio[left:left+512]
            if len(frame)<512:frame=np.pad(frame,(0,512-len(frame)))
            if self.speech_detector(frame)>=settings['vad_speech_threshold']:
                right=min(left+512,len(audio))
                if regions and regions[-1][1]==left:regions[-1][1]=right
                else:regions.append([left,right])
        return regions

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        self.process.stdout.close()
