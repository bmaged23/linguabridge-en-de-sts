"""Real Silero/endpoint tests with known silence controls and weak waveform boundaries.

Speech onset/offset references are energy-derived, not phonetic human annotations.
Interruption measurements cover the speech gate, not browser cancellation latency.
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np
from voice_bridge.settings import ROOT, load_config
from voice_bridge.workers import Silero
from voice_bridge.core import Endpoint
from scripts.evaluation.audio_tools import read_audio
from scripts.evaluation.run_component_benchmarks import (
    settings,
    rows,
    append,
    configuration,
)
from scripts.story_tests.resource_monitor import Resources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / settings()["output_directory"])
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output = (ROOT / a.output).resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    a.stage = "vad"
    cfg = load_config()
    fixtures = []
    for row in rows(a.output / "tts.jsonl"):
        if not row["completed"] or row["variant"] != "streamed_phrases":
            continue
        fixtures.append(
            {
                "id": row["case_id"],
                "group": row["group"],
                "audio": row["audio"],
                "expected_speech": True,
                "variant": "clean",
            }
        )
        path = a.output / "stt-inputs" / (row["id"] + "--noise20.wav")
        if path.exists():
            fixtures.append(
                {
                    "id": row["case_id"] + "--noise20",
                    "group": row["group"],
                    "audio": str(path.relative_to(ROOT)),
                    "expected_speech": True,
                    "variant": "noise20",
                }
            )
    for kind in ["silence", "white_noise"]:
        path = a.output / "stt-inputs" / ("control-" + kind + ".wav")
        if path.exists():
            fixtures.append(
                {
                    "id": "control-" + kind,
                    "group": "non_speech",
                    "audio": str(path.relative_to(ROOT)),
                    "expected_speech": False,
                    "variant": kind,
                }
            )
    if not fixtures:
        raise ValueError("Generate component inputs first")
    configuration(
        a,
        {
            "selected_input_ids": [r["id"] for r in fixtures],
            "profiles": [
                "conversation",
                "translation",
                "teaching",
                "interruption_gate",
            ],
        },
    )
    destination = a.output / "vad.jsonl"
    done = {r["id"] for r in rows(destination)}
    monitor = Resources(
        os.getpid(),
        a.output / "resources-vad.jsonl",
        settings()["resource_sample_interval_seconds"],
    )
    monitor.start()
    start = time.monotonic()
    try:
        detector = Silero(ROOT / cfg["paths"]["vad_model"], cfg["vad"])
        for profile in ["conversation", "translation", "teaching", "interruption_gate"]:
            gate = dict(cfg["vad"])
            if profile == "interruption_gate":
                gate.update(
                    {
                        k: cfg["interruptions"][k]
                        for k in ["min_speech_ms", "speech_threshold", "minimum_rms"]
                    }
                )
            else:
                gate.update(
                    {
                        k: cfg["modes"][profile][k]
                        for k in ["end_of_turn_silence_ms", "max_utterance_seconds"]
                    }
                )
            for fixture in fixtures:
                identity = fixture["id"] + "--" + profile
                if identity in done:
                    continue
                detector.state.fill(0)
                detector.context.fill(0)
                endpoint = Endpoint(gate)
                source = read_audio(ROOT / fixture["audio"])
                prefix = np.zeros(6400, np.float32)
                suffix = np.zeros(32000, np.float32)
                audio = np.concatenate([prefix, source, suffix])
                audio = np.pad(audio, (0, (-len(audio)) % 512))
                active = np.flatnonzero(
                    np.abs(source) >= settings()["signal_rms_threshold"]
                )
                onset = 0.4 + active[0] / 16000 if len(active) else None
                offset = 0.4 + active[-1] / 16000 if len(active) else None
                monitor.label = identity
                began = time.monotonic()
                events = []
                probabilities = []
                false_padding_frames = padding_frames = 0
                for i in range(0, len(audio), 512):
                    frame = audio[i : i + 512]
                    probability = detector(frame)
                    probabilities.append(probability)
                    gate_positive = (
                        probability >= gate["speech_threshold"]
                        and float(np.sqrt(np.mean(frame * frame)))
                        >= gate["minimum_rms"]
                    )
                    if i + 512 <= len(prefix) or i >= len(prefix) + len(source):
                        padding_frames += 1
                        false_padding_frames += int(gate_positive)
                    for kind, segment in endpoint.feed(frame, probability):
                        event = {"kind": kind, "at_seconds": (i + 512) / 16000}
                        if segment is not None:
                            event.update(
                                captured_seconds=len(segment) / 16000,
                                captured_start_seconds=(i + 512 - len(segment)) / 16000,
                            )
                        events.append(event)
                elapsed = time.monotonic() - began
                starts = [e for e in events if e["kind"] == "start"]
                ends = [e for e in events if e["kind"] == "end"]
                result = {
                    **fixture,
                    "id": identity,
                    "profile": profile,
                    "events": events,
                    "speech_detected": bool(starts),
                    "missed_speech_case": fixture["expected_speech"] and not starts,
                    "false_speech_case": not fixture["expected_speech"]
                    and bool(starts),
                    "endpoint_count": len(ends),
                    "recording_split_before_end": any(
                        e["at_seconds"] < 0.4 + len(source) / 16000 for e in ends
                    ),
                    "configured_duration_limit_can_split": len(source) / 16000
                    > gate["max_utterance_seconds"],
                    "speech_gate_delay_from_energy_onset_seconds": starts[0][
                        "at_seconds"
                    ]
                    - onset
                    if starts and onset is not None
                    else None,
                    "final_endpoint_delay_from_energy_offset_seconds": ends[-1][
                        "at_seconds"
                    ]
                    - offset
                    if ends and offset is not None
                    else None,
                    "initial_energy_clipped_seconds": max(
                        0, ends[0]["captured_start_seconds"] - onset
                    )
                    if ends and onset is not None
                    else None,
                    "false_positive_inserted_silence_frames": false_padding_frames,
                    "inserted_silence_frames": padding_frames,
                    "audio_seconds": len(audio) / 16000,
                    "compute_seconds": elapsed,
                    "real_time_factor": elapsed / (len(audio) / 16000),
                    "resources": monitor.summarize(began, time.monotonic()),
                    "vad_probabilities": probabilities,
                }
                append(destination, result)
                print(
                    json.dumps(
                        {"id": identity, "speech": bool(starts), "endpoints": len(ends)}
                    ),
                    flush=True,
                )
        data = rows(destination)
        summary = {
            "cases": len(data),
            "expected_cases": len(fixtures) * 4,
            "missed_speech_cases": sum(bool(r["missed_speech_case"]) for r in data),
            "false_speech_cases": sum(r["false_speech_case"] for r in data),
            "profiles": {},
            "resources": monitor.summarize(start, time.monotonic()),
            "limitations": [
                "Known inserted silence and synthetic noise are true non-speech controls.",
                "Speech boundaries are energy-derived, not manual phonetic labels; no frame-F1 claim.",
                "Interruption gate latency is not full browser barge-in latency.",
                "Offline compute speed is not wall-clock microphone delivery.",
            ],
        }
        for profile in ["conversation", "translation", "teaching", "interruption_gate"]:
            subset = [r for r in data if r["profile"] == profile]
            summary["profiles"][profile] = {
                "cases": len(subset),
                "missed_speech": sum(bool(r["missed_speech_case"]) for r in subset),
                "false_speech": sum(r["false_speech_case"] for r in subset),
                "recordings_split_before_end": sum(
                    r["recording_split_before_end"] for r in subset
                ),
                "median_compute_real_time_factor": float(
                    np.median([r["real_time_factor"] for r in subset])
                ),
            }
        (a.output / "summary-vad.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
