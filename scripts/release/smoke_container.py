"""Verify the normal container entrypoint warms real models and serves the UI."""
import argparse
import json
import subprocess
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--name', default='linguabridge-release-smoke-' + uuid.uuid4().hex[:10])
    args = parser.parse_args()
    ownership = uuid.uuid4().hex
    try:
        subprocess.run(['docker', 'run', '-d', '--name', args.name,
                        '--label', 'io.linguabridge.test-run=' + ownership, '--gpus', 'all',
                        '-p', '127.0.0.1:8768:8766', '-e', 'OLLAMA_API_KEY',
                        '-e', 'OLLAMA_BASE_URL', '-e', 'OLLAMA_MODEL', args.image], check=True)
        deadline = time.monotonic() + 180
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8768/health', timeout=3) as response:
                    health = json.load(response)
                assert health['ready'] is True
                with urllib.request.urlopen('http://127.0.0.1:8768/', timeout=3) as response:
                    assert response.status == 200 and b'<html' in response.read().lower()
                print('Normal GPU container entrypoint, model warmup, health and UI passed.')
                return
            except (OSError, ValueError, AssertionError):
                if time.monotonic() > deadline:
                    subprocess.run(['docker', 'logs', args.name], check=False)
                    raise RuntimeError('Normal container startup smoke failed')
                time.sleep(1)
    finally:
        existing = subprocess.run(['docker', 'inspect', args.name, '--format',
                                   '{{index .Config.Labels "io.linguabridge.test-run"}}'],
                                  capture_output=True, text=True, check=False)
        if existing.returncode == 0 and existing.stdout.strip() == ownership:
            subprocess.run(['docker', 'rm', '-f', args.name], stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
