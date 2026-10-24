"""Container entry point: retain native settings except the bind address."""
from voice_bridge import settings

original = settings.load_config


def container_config():
    config = original()
    config['server']['host'] = '0.0.0.0'
    return config


if __name__ == '__main__':
    # Uvicorn runs in this interpreter so the container-only override is inherited.
    settings.load_config = container_config
    import os
    from pathlib import Path
    root = Path('/app')
    libraries = list((root / '.venv-asr/lib').glob('python*/site-packages/nvidia/*/lib'))
    os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries)) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    # Dynamic-loader search paths must exist when the process starts.
    if os.environ.get('BRIDGE_CONTAINER_LIBRARIES_READY') != '1':
        import sys
        os.environ['BRIDGE_CONTAINER_LIBRARIES_READY'] = '1'
        os.execv(sys.executable, [sys.executable, '-m', 'deployment.docker.start'])
    import uvicorn
    uvicorn.run('voice_bridge.server:app', host='0.0.0.0', port=8766,
                workers=1, log_level='info', ws_max_size=65536, access_log=False)
