"""Start localhost server with project-local CUDA libraries on Linux."""
import os
from pathlib import Path
import subprocess
import sys
import runpy

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    config = runpy.run_path(str(ROOT / "config.py"))["CONFIG"]
    env = os.environ.copy()
    env["HF_HOME"] = str(ROOT / config["paths"]["cache"])
    library_runtime = config['tts'].get('cuda_library_runtime','.venv') if config['tts']['device']=='cuda' else '.venv'
    libraries = list((ROOT / library_runtime / 'lib').glob("python*/site-packages/nvidia/*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in libraries) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [sys.executable, "-m", "uvicorn", "voice_bridge.server:app",
               "--host", config["server"]["host"], "--port", str(config["server"]["port"]),
               "--workers", str(config["server"]["web_workers"]),
               "--log-level", config["logging"]["level"].lower(),
               "--ws-max-size", str(config["audio"]["max_websocket_message_bytes"]), "--no-access-log"]
    if len(sys.argv) > 1:
        command = [sys.executable, *sys.argv[1:]]
    process = subprocess.Popen(command, env=env, cwd=ROOT)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        # Uvicorn receives the same terminal SIGINT and shuts down its workers.
        code = process.wait()
    raise SystemExit(code)
