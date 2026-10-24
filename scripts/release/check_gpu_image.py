"""Test a candidate image using archived ASR inputs and real browser/model flows."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--python', default=str(ROOT / '.venv/bin/python'))
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum():
        parser.error('Use a simple unique run ID')
    fixtures = args.fixtures.resolve()
    local = ROOT / '.evaluation-local'
    allowed = {entry.name for entry in fixtures.iterdir()}
    if local.exists() and any(p.name not in allowed and not p.name.startswith('image-') for p in local.iterdir()):
        raise RuntimeError('Use a clean release checkout; unrelated private experiments must not be mounted into the candidate')
    local.mkdir(exist_ok=True)
    # Copy only the dedicated non-secret fixture bundle into a fresh checkout.
    for entry in fixtures.iterdir():
        destination = local / entry.name
        if not destination.exists():
            shutil.copytree(entry, destination) if entry.is_dir() else shutil.copy2(entry, destination)
    output = local / ('image-' + args.run_id)
    output.mkdir(parents=True, exist_ok=False)
    mount = f'{local}:/app/.evaluation-local'
    user = f'{os.getuid()}:{os.getgid()}'
    docker = ['docker', 'run', '--rm', '--gpus', 'all', '--user', user,
              '-v', mount, '--entrypoint', '/app/.venv/bin/python']
    asr = output.relative_to(ROOT) / 'asr'
    run(*docker, args.image, '-m', 'scripts.evaluation.benchmark_mixed_asr',
        '--stage', 'evaluate', '--split', 'all', '--method', 'speech_gated_pause_batch2',
        '--inputs', '.evaluation-local/mixed-asr-all-inputs-v2', '--output', asr)
    run(*docker, args.image, '-m', 'scripts.evaluation.compare_mixed_asr',
        '--candidate', asr, '--baseline', '.evaluation-local/mixed-asr-guarded-v3',
        '--baseline-method', 'speech_gated_pause_beam3', '--strict-quality',
        '--output', output.relative_to(ROOT) / 'asr-gate.json')
    name = 'linguabridge-qa-' + args.run_id.lower()
    browser = output.relative_to(ROOT) / 'browser'
    try:
        run('docker', 'run', '-d', '--name', name, '--gpus', 'all', '--user', user,
            '-v', mount, '-p', '127.0.0.1:8767:8767',
            '-e', 'OLLAMA_API_KEY', '-e', 'OLLAMA_BASE_URL', '-e', 'OLLAMA_MODEL',
            '-e', 'VOICE_BRIDGE_QA_BIND_HOST=0.0.0.0',
            '-e', 'VOICE_BRIDGE_QA_AUDIT_FINAL=0',
            '-e', 'VOICE_BRIDGE_QA_MAX_REQUESTS=20',
            '-e', f'VOICE_BRIDGE_QA_DIR={browser}',
            '--entrypoint', '/app/.venv/bin/python', args.image,
            '-m', 'scripts.application.start_voice_app', '-m', 'scripts.validation.start_test_server')
        deadline = time.monotonic() + 180
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8767/health', timeout=3) as response:
                    if json.load(response)['ready']:
                        break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('GPU container did not become ready')
                time.sleep(1)
        run(args.python, '-m', 'scripts.evaluation.run_browser_flow_suite',
            '--external-server', '--inputs', '.evaluation-local/flow-inputs-v2', '--output', browser)
    finally:
        with (output / 'container.log').open('w') as stream:
            subprocess.run(['docker', 'logs', name], stdout=stream, stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL)
    (output / 'release-gate.json').write_text(json.dumps({
        'image': args.image, 'passed': True, 'asr_waveforms': 52, 'browser_cases': 17,
        'scope': 'Image GPU models; external host browser. Browser runner RSS excludes container.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
