"""Pinned download inputs only; runtime settings remain in root config.py."""
CONFIG = {'stt': {'provider': 'qwen_asr',
         'model': 'Qwen3-ASR-0.6B',
         'model_repo': 'Qwen/Qwen3-ASR-0.6B',
         'revision': '5eb144179a02acc5e5ba31e748d22b0cf3e303b0'},
 'paths': {'stt_model': 'models/qwen3-asr-0.6b',
           'vad_model': 'models/silero_vad.onnx'},
 'tts': {'voices': {'en': {'provider': 'kokoro',
                           'name': 'af_heart',
                           'model': 'models/kokoro-v1.0.onnx',
                           'voices': 'models/voices-v1.0.bin'},
                    'de': {'provider': 'piper',
                           'name': 'de_DE-thorsten-high',
                           'model': 'models/piper/de_DE-thorsten-high.onnx'}}}}
