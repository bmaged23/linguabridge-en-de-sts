import copy
import importlib.util
import os
from pathlib import Path
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[2]


def load_config():
    spec=importlib.util.spec_from_file_location('bridge_user_config',ROOT/'config.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return copy.deepcopy(module.CONFIG)


def load_settings():
    config=load_config();secrets={**dotenv_values(ROOT/'.env'),**os.environ}
    for key in ('api_key','model','base_url'):
        value=secrets.get(config['llm'][key+'_env'])
        if not value:raise ValueError('Missing '+config['llm'][key+'_env'])
        config['llm'][key]=value
    if config['llm']['base_url'].rstrip('/')!='https://ollama.com':
        raise ValueError('Cloud credentials may only be sent to https://ollama.com')
    if (config['audio']['input_sample_rate'],config['audio']['frame_samples'],config['audio']['input_channels'])!=(16000,512,1):
        raise ValueError('Microphone capture must be 16 kHz, 512 samples, mono')
    if config['server']['web_workers']!=1 or config['workers']['max_sessions']!=1:
        raise ValueError('Only one local session/web worker is supported')
    if config['workers']['start_method']!='spawn':raise ValueError('Use spawn for speech workers')
    if config['stt']['provider']=='qwen_asr':
        stt=config['stt']
        if stt['device']!='cuda' or stt['dtype']!='float32':raise ValueError('Use the validated Qwen CUDA FP32 configuration')
        if config['workers']['stt_processes']!=1:raise ValueError('Qwen supports one GPU worker')
        if stt['window_seconds']<max(config['modes'][name]['max_utterance_seconds'] for name in ('translation','conversation','teaching')):
            raise ValueError('Qwen requires cumulative whole-utterance previews')
        if min(stt['rollback_tokens'],stt['max_new_tokens'],stt['partial_interval_ms'])<=0:raise ValueError('Invalid Qwen stream limits')
    if config['controls']['confirmation_timeout_seconds']<=0:raise ValueError('Confirmation timeout must be positive')
    if config['tts']['device'] not in ('cpu','cuda'):raise ValueError('Unsupported TTS device')
    if config['tts']['device']=='cuda':
        if config['workers']['tts_processes']!=1:raise ValueError('Use one GPU TTS process sharing both voices')
        if config['tts']['parallel_chunks']<1:raise ValueError('Invalid TTS concurrency')
        if any(not 1<=count<=config['tts']['parallel_chunks'] for count in config['tts']['parallel_chunks_by_language'].values()):
            raise ValueError('Invalid per-language TTS concurrency')
    if set(config['tts']['voices'])!={'en','de'}:raise ValueError('Provide both English and German voices')
    if set(config['modes'])!={'default','translation','conversation','teaching'}:raise ValueError('Invalid modes')
    if config['modes']['default'] not in ('translation','conversation','teaching'):raise ValueError('Invalid default mode')
    for name in ('translation','conversation','teaching'):
        profile=config['modes'][name]
        if profile['end_of_turn_silence_ms']<=0 or profile['max_utterance_seconds']<=0:
            raise ValueError('Mode timing must be positive')
    for name in ('stt_processes','tts_processes','tts_queue_max_phrases','playback_queue_max_chunks','llm_token_queue_size'):
        if not isinstance(config['workers'][name],int) or config['workers'][name]<1:raise ValueError('Invalid worker count/queue size')
    for gate in (config['vad'],config['interruptions']):
        if not 0<gate['speech_threshold']<1:raise ValueError('Invalid VAD threshold')
        if not 0<=gate['minimum_rms']<1 or gate['min_speech_ms']<=0:raise ValueError('Invalid speech gate')
    if config['audio']['playback_echo_tail_ms']<0:raise ValueError('Invalid echo tail')
    chunk=config['tts']['chunking']
    if not 0<chunk['min_characters']<=chunk['max_characters'] or chunk['flush_timeout_ms']<=0:
        raise ValueError('Invalid phrase limits')
    mem=config['memory']
    if mem['summary_trigger_turns']<=mem['recent_turns'] or mem['recent_turns']<1:
        raise ValueError('Summary trigger must exceed recent-turn count')
    if not mem['command_prefixes']:raise ValueError('Explicit command prefix required')
    if config['logging']['record_audio'] or config['logging']['log_transcripts']:
        raise ValueError('Raw audio/transcript logging is disabled; SQLite memory is configured separately')
    return config
