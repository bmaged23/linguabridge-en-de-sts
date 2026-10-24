# Project layout and generated files

The repository separates the running application, evaluation, delivery and documentation. Run commands from the repository root so package imports, relative model paths and pinned requirements resolve consistently.

## Source and operational files

| Location | Responsibility |
| --- | --- |
| `src/voice_bridge/` | FastAPI server, VAD/turn handling, model workers, bilingual routing, memory and speech segmentation |
| `web/` | Browser UI, audio capture, bounded microphone transport and ordered playback |
| `config.py` | Application models, thresholds, queues, worker counts and memory settings |
| `scripts/application/`, `scripts/setup/` | Start the application and download verified model assets |
| `scripts/runtime/` | Persistent ASR worker used by the application |
| `scripts/validation/`, `scripts/evaluation/`, `scripts/story_tests/` | Configuration checks, model measurements and full conversation tests |
| `scripts/reports/`, `scripts/documentation/`, `scripts/demo/` | Report export, animated pipeline and demo generation |
| `scripts/release/` | Image qualification, real startup checks and Docker Hub metadata sync |
| `scripts/dependencies/` | Installed dependency snapshot export/checking |
| `scripts/archive/` | Historical, one-off audit orchestration; not a fresh-clone startup path |
| `tests/` | Offline behavior regressions with descriptive `test_*.py` filenames |
| `evaluation/datasets/` | Versioned reference cases and semantic expectations |
| `evaluation/results/` | Reviewed, dated results; private raw runs are stored separately |
| `deployment/docker/` | Dockerfile, model-build settings, entrypoint and Compose configuration |
| `deployment/dockerhub/` | Maintained public Docker Hub Overview |
| `.github/workflows/` | Offline CI, GPU qualification, registry delivery and documentation sync |
| `requirements/release/` | Supported application/voice install pins; ASR pins are in `requirements/stt-cuda.txt` |
| `requirements/archive/` | Historical environments and the obsolete uv lock, preserved for provenance |

The [tool index](../../scripts/README.md), [dependency index](../../requirements/README.md), [API README](../api/README.md) and [deployment guide](../../deployment/README.md) provide commands and usage details.

## Documentation and published assets

- `docs/api/`: HTTP endpoints and the live audio WebSocket protocol.
- `docs/guides/`: model deployment, memory and evaluation instructions.
- `docs/reference/`: configuration, asset hashes, model notices and layout reference.
- `docs/reports/`: report narratives and the [current/historical report index](../reports/README.md).
- `docs/demo/`, `docs/assets/`: reviewed demo media and animated pipeline assets.
- `docs/advanced-story/`, `docs/full-qa/`, `docs/mixed-output-check/`, `docs/tts-comparison/`: older reviewed report exports. Their directories intentionally retain stable published URLs and relative audio/dashboard links. Their READMEs identify them as historical; use the latest dated release report for current results.

The larger synthetic WAVs, videos and report JSON files in these published folders are intentional evidence or demonstration assets. File size alone does not make them disposable. New reviewed evaluation summaries belong in `evaluation/results/<descriptive-run-name>/`; new private traces belong in `.evaluation-local/` or `logs/`.

## Local content excluded from Git

| Location or pattern | Contents |
| --- | --- |
| `.env`, `.env.*` except `.env.example` | Credentials and local overrides |
| `.venv/`, `.venv-*/`, `venv/` | Installed Python environments |
| `models/`, `.cache/`, `.uv-cache/` | Downloaded weights and reusable caches |
| `data/`, `*.sqlite3`, `*.db` | Personal notes and conversation memory |
| `.evaluation-local/`, `logs/`, `recordings/` | Private runs, raw traces, GPU-runner workspaces and recordings |
| `description/` | Private project handoff/history |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | Regenerable Python/tool caches |
| `playwright-report/`, `test-results/`, `blob-report/` | Generated browser test output |
| `tmp/`, `temp/`, `*.tmp`, `*.part`, `*.download` | Temporary/incomplete files |
| `/uv.lock` | A newly generated root resolution; supported installs use the documented pinned files |

Ignoring content prevents future commits; it does not delete local files or remove tracked evidence. The Docker build uses a separate strict allowlist in [`.dockerignore`](../../.dockerignore), excluding credentials, personal memory, local environments and raw private recordings.
