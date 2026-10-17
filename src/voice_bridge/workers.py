"""One GPU recognizer and one shared GPU process holding both language voices."""
from pathlib import Path
import time
import numpy as np
from .routing import choose_language

_model=None
_settings=None


def _locked_phonemizer(function, lock):
    def call(*args, **kwargs):
        with lock:return function(*args, **kwargs)
    return call


def init_stt(settings, root):
    global _model, _settings
    import multiprocessing, signal
    if multiprocessing.current_process().name != 'MainProcess':signal.signal(signal.SIGINT,signal.SIG_IGN)
    _settings=settings
    if settings['provider']=='qwen_asr':
        from .qwen_bridge import QwenBridge
        _model=QwenBridge(settings,root)
        return
    from faster_whisper import WhisperModel
    _model=WhisperModel(str(Path(root)/settings['model_path']),device=settings['device'],
        device_index=settings['device_index'],compute_type=settings['compute_type'],
        cpu_threads=settings['cpu_threads'],local_files_only=True)


def transcribe(audio, previous=None, stream_id=None, final=True):
    if _settings['provider']=='qwen_asr':
        started=time.perf_counter()
        result=_model.transcribe(audio,stream_id,final)
        result['decode_seconds']=round(time.perf_counter()-started,4)
        if final and _settings.get('audit_fresh_final',False):
            fresh=_model.transcribe(audio,str(stream_id)+':audit',True)
            result['audit_fresh_text']=fresh['text']
        return result
    _,_,scores=_model.detect_language(audio=audio,language_detection_segments=1)
    language,certain,probability=choose_language(scores,_settings,previous)
    supported={key:float(dict(scores).get(key,0)) for key in _settings['allowed_languages']}
    candidate=max(supported,key=supported.get)
    mass=sum(supported.values())
    def decode(language):
        segments,_=_model.transcribe(audio,language=language,task='transcribe',
            beam_size=_settings['beam_size'],vad_filter=False,condition_on_previous_text=False,
            no_speech_threshold=_settings['no_speech_threshold'],log_prob_threshold=_settings['log_prob_threshold'])
        return ' '.join(s.text.strip() for s in segments).strip()
    text=decode(language)
    used_context=not certain and previous in ('en','de')
    if used_context and len(text.split())>=_settings['minimum_language_words']:
        if language!=candidate:language=candidate;text=decode(language)
        used_context=False
    relative=supported[language]/max(mass,1e-10)
    can_route=certain or (mass>=_settings['minimum_supported_language_probability'] and
        (relative>=_settings['relative_language_confidence_threshold'] or used_context))
    greetings={'hello':'en','hi':'en','hey':'en','good morning':'en','good evening':'en',
               'hallo':'de','guten tag':'de','guten morgen':'de','guten abend':'de'}
    import re
    greeting=greetings.get(re.sub(r'[^\w\s]','',text.lower()).strip())
    if greeting:
        language=greeting;can_route=True
    return {'text':text,'language':language,'certain':certain,'probability':supported.get(language,0),
            'used_context':used_context,'can_route':can_route,'supported_probability':mass}


class KokoroSession:
    """kokoro-onnx 0.4.9 assumes integer speed for input_ids exports.

    The pinned v1.1 graph instead declares float speed. Respect the graph's
    schema without changing third-party code or truncating fractional speeds.
    """
    def __init__(self, session):
        self.session = session
        speed = next(item for item in session.get_inputs() if item.name == "speed")
        self.speed_dtype = {"tensor(float)": np.float32, "tensor(int32)": np.int32}[speed.type]

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, inputs):
        inputs = dict(inputs)
        # Pass through the caller's intended speed; see synthesize().
        inputs["speed"] = np.asarray([self.speed], dtype=self.speed_dtype)
        return self.session.run(outputs, inputs)



def init_tts(settings, root):
    global _model, _settings
    import multiprocessing, signal
    if settings['device']=='cuda':
        # multiprocessing restores the parent's sys.path even when a different
        # executable is selected. Activate the GPU runtime before importing ORT.
        import sys
        runtime=Path(root)/settings['runtime_python']
        sys.path[:0]=[str(path) for path in (runtime.parent.parent/'lib').glob('python*/site-packages')]
    import onnxruntime as ort
    import threading
    if multiprocessing.current_process().name != 'MainProcess':signal.signal(signal.SIGINT,signal.SIG_IGN)
    _settings=settings;_model={};root=Path(root)
    options=ort.SessionOptions();options.intra_op_num_threads=settings['intra_op_threads'];options.inter_op_num_threads=settings['inter_op_threads']
    providers=['CPUExecutionProvider']
    if settings['device']=='cuda':
        if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('GPU ONNX Runtime is required for CUDA TTS')
        providers=[('CUDAExecutionProvider',settings['cuda']), 'CPUExecutionProvider']
    phoneme_lock=threading.RLock()
    for language,voice in settings['voices'].items():
        session=ort.InferenceSession(str(root/voice['model']),sess_options=options,providers=providers)
        if settings['device']=='cuda' and session.get_providers()[0]!='CUDAExecutionProvider':
            raise RuntimeError('CUDA TTS initialization failed; refusing silent CPU fallback')
        if voice['provider']=='kokoro':
            from kokoro_onnx import Kokoro
            model=Kokoro.from_session(session,str(root/voice['voices']))
            model.sess=KokoroSession(session);model.sess.speed=voice['speed'];_model[language]=model
            model.tokenizer.phonemize=_locked_phonemizer(model.tokenizer.phonemize,phoneme_lock)
        else:
            from piper import PiperVoice
            model=PiperVoice.load(str(root/voice['model']),use_cuda=False)
            model.session=session;_model[language]=model
            model.phonemize=_locked_phonemizer(model.phonemize,phoneme_lock)


def synthesize(text, language='en'):
    if language not in _model:raise ValueError('Unsupported TTS language')
    voice=_settings['voices'][language]
    if voice['provider']=='kokoro':
        audio,rate=_model[language].create(text,voice=voice['name'],speed=voice['speed'],lang='en-us')
        return (np.clip(audio,-1,1)*32767).astype('<i2').tobytes(),rate
    from piper import SynthesisConfig
    settings=SynthesisConfig(length_scale=_settings['length_scale'],noise_scale=_settings['noise_scale'],
        noise_w_scale=_settings['noise_w_scale'],normalize_audio=_settings['normalize_audio'])
    chunks=list(_model[language].synthesize(text,syn_config=settings))
    if not chunks:raise ValueError('Piper returned no audio')
    return b''.join(chunk.audio_int16_bytes for chunk in chunks),chunks[0].sample_rate


def ready():return _model is not None


def synthesize_batch(phrases, language='en'):
    """Bounded concurrent inference shares weights; output stays in input order."""
    from concurrent.futures import ThreadPoolExecutor
    count=_settings.get('parallel_chunks_by_language',{}).get(language,_settings.get('parallel_chunks',1))
    with ThreadPoolExecutor(max_workers=count) as executor:
        return list(executor.map(lambda phrase:synthesize(phrase,language),phrases))


def warmup_stt():
    if _settings['provider']=='qwen_asr':return True
    transcribe(np.zeros(16000,np.float32),'en')
    return True


def shutdown_stt():
    if _settings and _settings['provider']=='qwen_asr':_model.close()


def warmup_tts():
    synthesize('Ready.','en');synthesize('Bereit.','de')
    return True


class Silero:
    """ONNX recurrent state for one 16 kHz microphone stream (512 samples/call)."""
    def __init__(self, path, settings=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = (settings or {}).get("intra_op_threads", 1)
        options.inter_op_num_threads = (settings or {}).get("inter_op_threads", 1)
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame):
        audio = np.concatenate((self.context, frame.reshape(1, 512)), axis=1)
        output, self.state = self.session.run(None, {"input": audio,
                    "state": self.state, "sr": np.array(16000, dtype=np.int64)})
        self.context = audio[:, -64:]
        return float(output[0][0])
