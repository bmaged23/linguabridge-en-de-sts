"""The image's separately cached model downloads must match runtime assets."""
import runpy
import os
from voice_bridge.settings import ROOT, load_config


def test_cached_container_models_match_runtime():
    runtime = load_config()
    build = runpy.run_path(str(ROOT / 'deployment/docker/model_config.py'))['CONFIG']
    for key, value in build['stt'].items():
        assert runtime['stt'][key] == value, f'Container ASR pin differs: {key}'
    for key, value in build['paths'].items():
        assert runtime['paths'][key] == value, f'Container model path differs: {key}'
    assert set(build['tts']['voices']) == set(runtime['tts']['voices'])
    for language, voice in build['tts']['voices'].items():
        for key, value in voice.items():
            assert runtime['tts']['voices'][language][key] == value


def test_environment_dedup_keeps_different_libraries_separate(tmp_path, monkeypatch):
    implementation = runpy.run_path(str(ROOT / 'deployment/docker/deduplicate_envs.py'))
    files = []
    for environment, contents in [('.venv', b'a'), ('.venv-asr', b'a'), ('.venv-tts-gpu', b'b')]:
        path = tmp_path / environment / 'lib' / 'provider.so'
        path.parent.mkdir(parents=True)
        path.write_bytes(contents * 8192)
        files.append(path)
    monkeypatch.chdir(tmp_path)
    implementation['main']()
    assert os.stat(files[0]).st_ino == os.stat(files[1]).st_ino
    assert os.stat(files[0]).st_ino != os.stat(files[2]).st_ino
    assert files[2].read_bytes() == b'b' * 8192
