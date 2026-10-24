import numpy as np
import pytest
from scripts.evaluation.audio_tools import resample, write_audio, signal_metrics
from scripts.evaluation.run_component_benchmarks import settings


def test_resampling_preserves_duration_and_wav_round_trip(tmp_path):
    rate = 22050
    signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
    converted = resample(signal, rate, 24000)
    assert abs(len(converted) - 24000) <= 1
    assert np.isfinite(converted).all()
    assert len(write_audio(tmp_path / "audio.wav", converted, 24000)) == 64


def test_signal_statistics_and_nonfinite_audio_are_explicit(tmp_path):
    audio = np.concatenate([np.zeros(1600), np.full(1600, 0.1)]).astype(np.float32)
    result = signal_metrics(audio, 16000, settings())
    assert result["silent_frame_fraction"] == 0.5
    assert result["longest_silent_run_seconds"] == 0.1
    assert result["clipped_sample_fraction"] == 0
    with pytest.raises(ValueError):
        write_audio(tmp_path / "bad.wav", np.array([np.nan]), 16000)
