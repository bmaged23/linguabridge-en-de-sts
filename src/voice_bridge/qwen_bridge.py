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

    def transcribe(self, audio, stream_id=None, final=True):
        return self._exchange({'type': 'transcribe', 'audio': base64.b64encode(audio.astype('<f4').tobytes()).decode(),
            'stream_id': stream_id, 'final': final}, self.settings['request_timeout_seconds'])

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
