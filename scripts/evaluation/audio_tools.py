"""Local audio artifact helpers; decoding supports integer and IEEE-float WAV input."""

import hashlib
import wave
from pathlib import Path
import av
import numpy as np


def read_audio(path, rate=16000):
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=rate)


def resample(samples, source_rate, target_rate):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return samples
    frame = av.AudioFrame.from_ndarray(samples[None, :], format="fltp", layout="mono")
    frame.sample_rate = source_rate
    converter = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
    frames = converter.resample(frame) + converter.resample(None)
    return (
        np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
        if frames
        else np.zeros(0, np.float32)
    )


def write_audio(path, samples, rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite waveform")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(values, -1, 1) * 32767).astype("<i2").tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal_metrics(samples, rate, settings):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(samples):
        return {"audio_seconds": 0, "finite": True, "empty": True}
    size = max(1, int(rate * settings["signal_frame_ms"] / 1000))
    blocks = [samples[i : i + size] for i in range(0, len(samples), size)]
    active = [
        bool(np.sqrt(np.mean(b * b)) >= settings["signal_rms_threshold"])
        for b in blocks
    ]
    longest = current = 0
    for flag in active:
        current = 0 if flag else current + 1
        longest = max(current, longest)
    return {
        "audio_seconds": len(samples) / rate,
        "finite": bool(np.isfinite(samples).all()),
        "empty": False,
        "rms": float(np.sqrt(np.mean(samples * samples))),
        "peak_amplitude": float(np.max(np.abs(samples))),
        "clipped_sample_fraction": float(
            np.mean(np.abs(samples) >= settings["clipping_amplitude_threshold"])
        ),
        "silent_frame_fraction": sum(not f for f in active) / len(active),
        "longest_silent_run_seconds": min(
            len(samples) / rate, longest * settings["signal_frame_ms"] / 1000
        ),
    }
