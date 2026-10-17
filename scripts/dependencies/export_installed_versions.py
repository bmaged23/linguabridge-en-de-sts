"""Snapshot installed environments without importing models or changing packages."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = {
    '.venv': ('requirements.txt', 'Web app, CPU VAD, tests and browser QA', True),
    '.venv-asr': ('requirements/stt-cuda.txt', 'Production Qwen STT; PyTorch CUDA 12.4', False),
    '.venv-tts-gpu': ('requirements/tts-cuda.txt', 'Production GPU Kokoro/Piper; separate from CPU ONNX Runtime', True),
    '.venv-qwen-tts': ('requirements/experimental-qwen-tts.txt', 'Optional experimental Qwen TTS, unused by the app', False),
}
PROBE = '''
import json, platform, sys
from importlib.metadata import distributions
packages = {}
for dist in distributions():
    name = dist.metadata.get('Name', '').lower().replace('_', '-').replace('.', '-')
    if name and name not in ('voice-bridge', 'linguabridge-sts'):
        packages[name] = dist.version
print(json.dumps({'python': platform.python_version(), 'platform': sys.platform,
                  'machine': platform.machine(), 'packages': packages}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify snapshots without writing files')
    args = parser.parse_args()
    mismatches = []
    for environment, (filename, purpose, editable) in ENVIRONMENTS.items():
        interpreter = ROOT / environment / 'bin/python'
        if not interpreter.exists():
            print(f'Skipping absent local environment: {environment}')
            continue
        result = subprocess.run([str(interpreter), '-c', PROBE], check=True,
                                capture_output=True, text=True)
        snapshot = json.loads(result.stdout)
        header = (
            f'# Exact installed snapshot: {environment}; Python {snapshot["python"]}; '
            f'{snapshot["platform"]} {snapshot["machine"]}.\n'
            f'# {purpose}.\n'
            '# Regenerate all snapshots: .venv/bin/python -m scripts.dependencies.export_installed_versions\n'
            '# Install instructions and environment caveats: README.md\n'
        )
        if any(version.endswith('+cu124') for version in snapshot['packages'].values()):
            header += '--extra-index-url https://download.pytorch.org/whl/cu124\n'
        pins = [f'{name}=={version}' for name, version in sorted(snapshot['packages'].items())]
        content = header + '\n'.join(pins) + '\n' + ('-e .\n' if editable else '')
        target = ROOT / filename
        if args.check:
            if not target.exists() or target.read_text() != content:
                mismatches.append(filename)
            else:
                print(f'{filename}: matches {len(pins)} installed distributions')
        else:
            target.write_text(content)
            print(f'{filename}: exported {len(pins)} exact pins from {environment}')
    if mismatches:
        raise SystemExit('Snapshots differ: ' + ', '.join(mismatches))


if __name__ == '__main__':
    main()
