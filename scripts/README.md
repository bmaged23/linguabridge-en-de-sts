# Project tools

Run commands from the repository root. Activate `.venv`, or use its Python explicitly. Grouped tools use `python -m scripts.<folder>.<tool>` so cross-folder imports resolve consistently.

## Everyday commands

```bash
# Download the local speech models.
.venv/bin/python -m scripts.setup.download_speech_models

# Check settings, then start the GPU app at http://localhost:8766.
.venv/bin/python -m scripts.validation.validate_configuration
.venv/bin/python -m scripts.application.start_voice_app

# Export or verify exact installed dependency versions.
.venv/bin/python -m scripts.dependencies.export_installed_versions
.venv/bin/python -m scripts.dependencies.export_installed_versions --check
```

`application/start_voice_app.py` adds the configured CUDA library paths. It can also run another module with those paths:

```bash
.venv/bin/python -m scripts.application.start_voice_app -m scripts.validation.validate_speech_models
```

## Tool folders

| Folder | Purpose | Starting point |
| --- | --- | --- |
| [`application/`](application/) | Start the server with project CUDA library paths | `start_voice_app.py` |
| [`setup/`](setup/) | Download pinned local speech models and manifests | `download_speech_models.py` |
| [`dependencies/`](dependencies/) | Export and check installed package versions | `export_installed_versions.py` |
| [`documentation/`](documentation/) | Build the interactive pipeline and README GIF | `build_pipeline_animation.py` |
| [`runtime/`](runtime) | Persistent GPU Qwen subprocess used by the app | `qwen_asr_worker.py`; started automatically, not a standalone CLI |
| [`demo/`](demo) | Record, render and verify the bilingual demo | `record_bilingual_demo.py` |
| [`validation/`](validation) | Configuration, cloud, translation, speech and browser checks | `validate_configuration.py`, `validate_voice_session.py` |
| [`evaluation/`](evaluation) | Recognition/synthesis comparisons and resource profiling | `evaluate_mixed_language_asr.py`, `compare_speech_synthesis.py` |
| [`story_tests/`](story_tests) | Continuous multilingual story inputs, pipeline tests and resource monitoring | `prepare_story_inputs.py`, `run_story_pipeline_tests.py` |
| [`reports/`](reports) | Export existing results as shareable reports and dashboards | `export_story_report.py`, `build_story_dashboard.py` |
| [`release/`](release) | Qualify real GPU images, check container startup and synchronize Docker Hub documentation | `check_gpu_image.py`, `smoke_container.py`, `update_dockerhub.py`; see the [release guide](../deployment/README.md) |
| [`archive/`](archive) | One-off October 2026 audit orchestration, retained as historical reference | See its README before use |

## Demo

```bash
.venv/bin/python -m scripts.demo.record_bilingual_demo
.venv/bin/python -m scripts.demo.render_video_cover
.venv/bin/python -m scripts.demo.validate_demo_media
```

See [recording details](../docs/demo/README.md). Recording uses the real cloud model and a separate synthetic database; it temporarily restarts the app to share GPU capacity.

## Checks and continuous stories

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.validation.validate_cloud_connection
.venv/bin/python -m scripts.validation.validate_voice_session

.venv/bin/python -m scripts.story_tests.prepare_story_inputs logs/advanced-story/my-run
.venv/bin/python -m scripts.story_tests.run_story_pipeline_tests --folder logs/advanced-story/my-run
.venv/bin/python -m scripts.reports.export_story_report logs/advanced-story/my-run
```

Live cloud checks use account quota. Some GPU evaluations need the isolated ASR or TTS interpreter and exclusive GPU capacity. Read the relevant [test guide](../docs/guides/advanced-story-testing.md) before running them. Inputs and raw traces belong in ignored `logs/`; reviewed synthetic exports belong in `docs/`.

The [filename migration map](../docs/reference/filename-migration.json) lists all renamed tools and requirement files. Old script names were replaced rather than kept as duplicate wrappers.

## Animated README pipeline

```bash
.venv/bin/python -m scripts.documentation.build_pipeline_animation
```

This writes `docs/architecture.html` and `docs/assets/pipeline-animation.gif`. It needs Playwright Chromium and FFmpeg; no cloud calls or speech models are loaded.

To preview the player and animation without loading models:

```bash
.venv/bin/python -m scripts.documentation.serve_docs
# Open http://localhost:8768/demo.html
```

This preview supports HTTP byte ranges, which video seeking needs.
