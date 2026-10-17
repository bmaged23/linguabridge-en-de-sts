"""Offline validation of config.py and credential presence; no API requests."""
from voice_bridge.settings import load_settings

if __name__ == '__main__':
    config = load_settings()
    print('config.py: valid; required credentials present (not printed)')
    print('Speech workers:', config['workers']['stt_processes'], 'STT /', config['workers']['tts_processes'], 'TTS')
    print('API connectivity is tested separately with scripts/validation/validate_cloud_connection.py.')
