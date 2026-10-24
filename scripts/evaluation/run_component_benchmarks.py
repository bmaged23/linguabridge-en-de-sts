"""Isolated speech-component measurements. Run GPU stages in separate processes.

Launch through scripts.application.start_voice_app to supply the same CUDA libraries
as the application. Raw artifacts stay in .evaluation-local; no cloud calls.
"""

import argparse, asyncio, copy, hashlib, json, os, runpy, statistics, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from voice_bridge.settings import ROOT, load_config
from scripts.evaluation.audio_tools import (
    read_audio,
    resample,
    write_audio,
    signal_metrics,
)
from scripts.evaluation.speech_metrics import (
    speech_metrics,
    aggregate_metrics,
    normalized_words,
)
from scripts.story_tests.resource_monitor import Resources


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def settings():
    return copy.deepcopy(runpy.run_path(str(ROOT / "config.py"))["EVALUATION_CONFIG"])


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def configuration(args, details):
    path = args.output / f"configuration-{args.stage}.json"
    source = hashlib.sha256()
    for p in [ROOT / "config.py", *sorted((ROOT / "src/voice_bridge").glob("*.py"))]:
        source.update(p.relative_to(ROOT).as_posix().encode() + p.read_bytes())
    implementation = hashlib.sha256()
    for p in [
        *[
            ROOT / "scripts/evaluation" / name
            for name in [
                "run_component_benchmarks.py",
                "speech_metrics.py",
                "audio_tools.py",
                "evaluate_vad.py",
            ]
        ],
        ROOT / "scripts/story_tests/resource_monitor.py",
    ]:
        implementation.update(p.relative_to(ROOT).as_posix().encode() + p.read_bytes())
    value = {
        "application_sha256": source.hexdigest(),
        "evaluation_implementation_sha256": implementation.hexdigest(),
        "evaluation_config": settings(),
        **details,
    }
    if path.exists():
        if not args.resume:
            raise RuntimeError(
                "Existing experiment; use --resume to retain all recorded cases or a new output folder"
            )
        assert json.loads(path.read_text()) == value, (
            "Configuration changed; choose a new experiment"
        )
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    return value


async def streamed_synthesis(case, config, workers, memory_path):
    from voice_bridge import server
    from voice_bridge.memory import Memory

    memory = Memory(memory_path, config["memory"])
    packets = []
    first = None
    started = time.monotonic()
    server.app.state.settings = config
    server.app.state.memory = memory
    server.app.state.diagnostics = {}

    class Socket:
        async def send_json(self, packet):
            nonlocal first
            if packet["type"] == "audio":
                if first is None:
                    first = time.monotonic() - started
                packets.append(packet)
                session.acknowledge({"id": packet["id"], "complete": True})

    session = server.Session(Socket())
    session.output_language = case["spans"][0][0]
    executor = ThreadPoolExecutor(1)
    server.app.state.tts = executor
    tokens = asyncio.Queue()
    phrases = asyncio.Queue(config["workers"]["tts_queue_max_phrases"])
    text = " ".join(f"[{language}]{text}" for language, text in case["spans"])
    for i in range(0, len(text), 11):
        tokens.put_nowait(text[i : i + 11])
    tokens.put_nowait(None)
    captured = []
    original = server.synthesize
    original_batch = server.synthesize_batch

    def single(text, language):
        value = workers.synthesize(text, language)
        captured.append((text, language, value))
        return value

    def batch(texts, language):
        values = workers.synthesize_batch(texts, language)
        captured.extend((text, language, value) for text, value in zip(texts, values))
        return values

    server.synthesize = single
    server.synthesize_batch = batch
    try:
        await asyncio.gather(
            session.chunk_tokens(tokens, phrases), session.speak(phrases, 0)
        )
        elapsed = time.monotonic() - started
        return captured, first, elapsed
    finally:
        server.synthesize = original
        server.synthesize_batch = original_batch
        executor.shutdown()
        memory.close()


def synthesize(args, cases):
    from voice_bridge import workers

    cfg = load_config()
    evaluation = settings()
    manifest = ROOT / "evaluation/datasets/bilingual-tts-v1.json"
    if args.tts_device:
        cfg["tts"]["device"] = args.tts_device
    configuration(
        args,
        {
            "dataset_sha256": digest(manifest),
            "selected_case_ids": [c["case_id"] for c in cases],
            "tts_settings": cfg["tts"],
        },
    )
    result_path = args.output / "tts.jsonl"
    existing = rows(result_path)
    seen = {r["id"] for r in existing}
    monitor = Resources(
        os.getpid(),
        args.output / "resources-tts.jsonl",
        evaluation["resource_sample_interval_seconds"],
    )
    monitor.start()
    start = time.monotonic()
    try:
        workers.init_tts(cfg["tts"], ROOT)
        workers.warmup_tts()
        startup = time.monotonic() - start
        providers = {
            lang: (
                model.sess if hasattr(model, "sess") else model.session
            ).get_providers()
            for lang, model in workers._model.items()
        }
        expected = (
            "CUDAExecutionProvider"
            if cfg["tts"]["device"] == "cuda"
            else "CPUExecutionProvider"
        )
        assert all(value[0] == expected for value in providers.values())
        for case in cases:
            for variant in evaluation["synthesis_variants"]:
                identity = case["case_id"] + "--" + variant
                if identity in seen:
                    continue
                monitor.label = identity
                started = time.monotonic()
                directory = args.output / "tts" / identity
                directory.mkdir(parents=True, exist_ok=True)
                row = {
                    "id": identity,
                    "case_id": case["case_id"],
                    "group": case["group"],
                    "variant": variant,
                    "reference": case["reference"],
                    "expected_spans": case["spans"],
                }
                try:
                    if variant == "streamed_phrases":
                        captured, first, elapsed = asyncio.run(
                            streamed_synthesis(
                                case,
                                cfg,
                                workers,
                                directory / "synthetic-memory.sqlite3",
                            )
                        )
                    else:
                        captured = []
                        first = None
                        for language, text in case["spans"]:
                            value = workers.synthesize(text, language)
                            captured.append((text, language, value))
                            if first is None:
                                first = time.monotonic() - started
                        elapsed = time.monotonic() - started
                    combined = []
                    chunks = []
                    boundaries = []
                    length = 0
                    for i, (text, language, (pcm, rate)) in enumerate(captured):
                        samples = (
                            np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                        )
                        native = directory / f"chunk-{i:03d}-{language}.wav"
                        sha = write_audio(native, samples, rate)
                        converted = resample(
                            samples, rate, evaluation["combined_audio_sample_rate"]
                        )
                        chunks.append(
                            {
                                "id": i,
                                "language": language,
                                "text": text,
                                "audio": str(native.relative_to(ROOT)),
                                "audio_sha256": sha,
                                "sample_rate": rate,
                                **signal_metrics(samples, rate, evaluation),
                            }
                        )
                        if combined:
                            boundaries.append(
                                {
                                    "sample": length,
                                    "absolute_sample_jump": abs(
                                        float(converted[0] - combined[-1][-1])
                                    ),
                                }
                            )
                        combined.append(converted)
                        length += len(converted)
                    audio = np.concatenate(combined)
                    path = directory / "combined.wav"
                    sha = write_audio(
                        path, audio, evaluation["combined_audio_sample_rate"]
                    )
                    expected = [lang for lang, _ in case["spans"]]
                    observed = [p["language"] for p in chunks]

                    def compress(items):
                        return [
                            v
                            for i, v in enumerate(items)
                            if i == 0 or v != items[i - 1]
                        ]

                    row.update(
                        completed=True,
                        audio=str(path.relative_to(ROOT)),
                        audio_sha256=sha,
                        chunks=chunks,
                        boundaries=boundaries,
                        signal=signal_metrics(
                            audio, evaluation["combined_audio_sample_rate"], evaluation
                        ),
                        generation_seconds=elapsed,
                        first_audio_seconds=first,
                        real_time_factor=elapsed
                        / (len(audio) / evaluation["combined_audio_sample_rate"]),
                        language_sequence_matches=compress(expected)
                        == compress(observed),
                        text_preserved=normalized_words(
                            " ".join(c["text"] for c in chunks)
                        )
                        == normalized_words(case["reference"]),
                        resources=monitor.summarize(started, time.monotonic()),
                    )
                except Exception as error:
                    row.update(
                        completed=False,
                        error_type=type(error).__name__,
                        error=str(error),
                        generation_wall_seconds=time.monotonic() - started,
                        resources=monitor.summarize(started, time.monotonic()),
                    )
                append(result_path, row)
                existing.append(row)
                print(
                    json.dumps(
                        {
                            "stage": "tts",
                            "id": identity,
                            "completed": row["completed"],
                            "seconds": row.get("generation_seconds"),
                            "error": row.get("error_type"),
                        }
                    ),
                    flush=True,
                )
        summary = {
            "selected_cases": len(cases),
            "expected_outputs": len(cases) * len(evaluation["synthesis_variants"]),
            "completed_outputs": sum(r["completed"] for r in existing),
            "warmup_and_model_load_seconds": startup,
            "resources": monitor.summarize(start, time.monotonic()),
            "device": cfg["tts"]["device"],
            "execution_providers": providers,
            "scope": "Direct TTS and actual application chunk/parser path with immediate synthetic acknowledgements; no cloud or browser playback timing.",
        }
        (args.output / "summary-tts.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    finally:
        monitor.stop()


def recognition_inputs(args, model):
    human = json.loads(
        (ROOT / "evaluation/datasets/fleurs-calibration-v1.json").read_text()
    )["cases"]
    selected = set(args.case or [])
    output = []
    for case in human:
        if selected and case["case_id"] not in selected:
            continue
        output.append(
            {
                **case,
                "category": "human",
                "group": case["language"],
                "id": case["case_id"],
            }
        )
    for row in rows(args.output / "tts.jsonl"):
        if not row["completed"] or (selected and row["case_id"] not in selected):
            continue
        if model == "qwen" and row["variant"] != "streamed_phrases":
            continue
        output.append(
            {
                "id": row["id"],
                "reference": row["reference"],
                "audio": row["audio"],
                "audio_sha256": row["audio_sha256"],
                "group": row["group"],
                "category": "tts_full" if model == "whisper" else "synthetic_clean",
            }
        )
        if model == "whisper":
            output.extend(
                {
                    "id": row["id"] + f"--chunk-{c['id']}",
                    "reference": c["text"],
                    "audio": c["audio"],
                    "audio_sha256": c["audio_sha256"],
                    "group": c["language"],
                    "language_hint": c["language"],
                    "category": "tts_chunk",
                }
                for c in row["chunks"]
            )
    if model == "qwen":
        for item in list(output):
            if item["category"] != "synthetic_clean":
                continue
            audio = read_audio(ROOT / item["audio"])
            rms = float(np.sqrt(np.mean(audio * audio)))
            seed = settings()["synthetic_noise_seed"] ^ int(
                hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16
            )
            noise = (
                np.random.default_rng(seed)
                .normal(
                    0,
                    rms / (10 ** (settings()["synthetic_noise_snr_db"] / 20)),
                    len(audio),
                )
                .astype(np.float32)
            )
            path = args.output / "stt-inputs" / (item["id"] + "--noise20.wav")
            sha = write_audio(path, audio + noise, 16000)
            output.append(
                {
                    **item,
                    "id": item["id"] + "--noise20",
                    "category": "synthetic_noise20",
                    "audio": str(path.relative_to(ROOT)),
                    "audio_sha256": sha,
                }
            )
    for kind in ["silence", "white_noise"]:
        identity = "control-" + kind
        if selected and identity not in selected:
            continue
        audio = (
            np.zeros(64000, np.float32)
            if kind == "silence"
            else np.random.default_rng(settings()["seed"])
            .normal(0, 0.006, 64000)
            .astype(np.float32)
        )
        path = args.output / "stt-inputs" / (identity + ".wav")
        sha = write_audio(path, audio, 16000)
        output.append(
            {
                "id": identity,
                "reference": "",
                "audio": str(path.relative_to(ROOT)),
                "audio_sha256": sha,
                "group": "non_speech",
                "category": "non_speech",
            }
        )
    return output[: args.limit] if args.limit else output


def recognize(args, model_name):
    evaluation = settings()
    inputs = recognition_inputs(args, model_name)
    if not inputs:
        raise ValueError(
            "No recognition inputs; prepare human data and/or synthesize first"
        )
    cfg = load_config()
    details = {
        "selected_input_ids": [r["id"] for r in inputs],
        "model": model_name,
        "synthetic_input_variant": "streamed_phrases"
        if model_name == "qwen"
        else "all_completed_tts_variants",
        "human_manifest_sha256": digest(
            ROOT / "evaluation/datasets/fleurs-calibration-v1.json"
        ),
        "tts_manifest_sha256": digest(
            ROOT / "evaluation/datasets/bilingual-tts-v1.json"
        ),
    }
    if model_name == "whisper":
        details.update(
            model_weights_sha256=digest(
                ROOT / evaluation["independent_asr"]["model_path"] / "model.bin"
            )
        )
    else:
        details.update(
            model_weights_sha256=digest(
                ROOT / cfg["paths"]["stt_model"] / "model.safetensors"
            )
        )
    configuration(args, details)
    path = args.output / (model_name + ".jsonl")
    saved = rows(path)
    done = {r["id"] for r in saved}
    monitor = Resources(
        os.getpid(),
        args.output / f"resources-{model_name}.jsonl",
        evaluation["resource_sample_interval_seconds"],
    )
    monitor.start()
    started = time.monotonic()
    bridge = None
    try:
        if model_name == "whisper":
            from faster_whisper import WhisperModel

            settings_ = evaluation["independent_asr"]
            model = WhisperModel(
                str(ROOT / settings_["model_path"]),
                device=settings_["device"],
                compute_type=settings_["compute_type"],
                cpu_threads=settings_["cpu_threads"],
                local_files_only=True,
            )
        else:
            from voice_bridge.qwen_bridge import QwenBridge

            settings_ = dict(
                cfg["stt"], model_path=cfg["paths"]["stt_model"], audit_raw_outputs=True
            )
            bridge = QwenBridge(settings_, ROOT)
        startup = time.monotonic() - started
        for item in inputs:
            methods = (
                ["auto", "known_language"]
                if model_name == "whisper" and item["category"] == "human"
                else ["fresh_final", "progressive_final"]
                if model_name == "qwen"
                else ["known_language"]
                if item.get("language_hint")
                else ["auto"]
            )
            for method in methods:
                identity = item["id"] + "--" + method
                if identity in done:
                    continue
                audio_path = ROOT / item["audio"]
                assert digest(audio_path) == item["audio_sha256"], (
                    "Audio fingerprint mismatch"
                )
                audio = read_audio(audio_path)
                monitor.label = identity
                start = time.monotonic()
                row = {
                    "id": identity,
                    "input_id": item["id"],
                    "category": item["category"],
                    "group": item["group"],
                    "method": method,
                    "reference": item["reference"],
                    "audio": item["audio"],
                    "audio_sha256": item["audio_sha256"],
                    "audio_seconds": len(audio) / 16000,
                }
                try:
                    previews = []
                    if model_name == "whisper":
                        hint = item.get("language_hint") or (
                            item["group"] if method == "known_language" else None
                        )
                        segments, info = model.transcribe(
                            audio,
                            language=hint,
                            task="transcribe",
                            beam_size=settings_["beam_size"],
                            multilingual=settings_["multilingual"]
                            if hint is None
                            else False,
                            condition_on_previous_text=False,
                            vad_filter=False,
                        )
                        segments = list(segments)
                        text = " ".join(s.text.strip() for s in segments)
                        result = {
                            "language": info.language,
                            "language_probability": info.language_probability,
                            "segments": [
                                {
                                    "start": s.start,
                                    "end": s.end,
                                    "text": s.text,
                                    "avg_logprob": s.avg_logprob,
                                }
                                for s in segments
                            ],
                        }
                    else:
                        if method == "progressive_final":
                            interval = cfg["stt"]["partial_interval_ms"] * 16
                            for end in range(interval, len(audio), interval):
                                t = time.monotonic()
                                part = bridge.transcribe(audio[:end], identity, False)
                                previews.append(
                                    {
                                        "audio_seconds": end / 16000,
                                        "seconds": time.monotonic() - t,
                                        **part,
                                    }
                                )
                        final_started = time.monotonic()
                        result = bridge.transcribe(audio, identity, True)
                        text = result["text"]
                        row["final_decode_seconds"] = time.monotonic() - final_started
                    elapsed = time.monotonic() - start
                    row.update(
                        completed=True,
                        hypothesis=text,
                        recognizer_output=result,
                        previews=previews,
                        metrics=speech_metrics(item["reference"], text),
                        decode_seconds=elapsed,
                        real_time_factor=elapsed / (len(audio) / 16000),
                        resources=monitor.summarize(start, time.monotonic()),
                    )
                    if item["group"] in ["en", "de"] and (
                        method == "auto" or model_name == "qwen"
                    ):
                        row["detected_language_correct"] = (
                            result["language"] == item["group"]
                        )
                except Exception as error:
                    row.update(
                        completed=False,
                        error_type=type(error).__name__,
                        error=str(error),
                        decode_seconds=time.monotonic() - start,
                        previews=previews,
                        resources=monitor.summarize(start, time.monotonic()),
                    )
                append(path, row)
                saved.append(row)
                print(
                    json.dumps(
                        {
                            "stage": model_name,
                            "id": identity,
                            "completed": row["completed"],
                            "wer": row.get("metrics", {}).get("wer"),
                            "error": row.get("error_type"),
                        }
                    ),
                    flush=True,
                )
        grouped = {}
        for row in saved:
            if row["completed"]:
                grouped.setdefault(
                    row["category"] + "/" + row["group"] + "/" + row["method"], []
                ).append(row)
        summary = {
            "model": model_name,
            "inputs": len(inputs),
            "completed": sum(r["completed"] for r in saved),
            "failed": sum(not r["completed"] for r in saved),
            "model_load_seconds": startup,
            "groups": {
                key: {
                    **aggregate_metrics(items),
                    "detected_language_accuracy": statistics.mean(
                        r["detected_language_correct"]
                        for r in items
                        if "detected_language_correct" in r
                    )
                    if any("detected_language_correct" in r for r in items)
                    else None,
                    "median_decode_seconds": statistics.median(
                        r["decode_seconds"] for r in items
                    ),
                    "median_real_time_factor": statistics.median(
                        r["real_time_factor"] for r in items
                    ),
                }
                for key, items in grouped.items()
            },
            "resources": monitor.summarize(started, time.monotonic()),
            "latency_scope": "Model load is separate; ASR has no explicit warm-up, so first decode includes kernel initialization.",
            "scope": "Offline reference-based component test. Progressive decoder timing includes cumulative previews; excludes real-time microphone delivery. Whisper TTS round-trip is an intelligibility proxy, not human ground truth.",
        }
        (args.output / f"summary-{model_name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    finally:
        if bridge:
            bridge.close()
        monitor.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["tts", "whisper", "qwen"], required=True)
    p.add_argument("--output", type=Path, default=ROOT / settings()["output_directory"])
    p.add_argument("--case", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tts-device", choices=["cpu", "cuda"])
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error("limit must be positive")
    a.output = (ROOT / a.output).resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == "tts":
        cases = json.loads(
            (ROOT / "evaluation/datasets/bilingual-tts-v1.json").read_text()
        )["cases"]
        if a.case:
            cases = [c for c in cases if c["case_id"] in a.case]
        if a.limit:
            cases = cases[: a.limit]
        if not cases:
            p.error("No matching TTS cases")
        synthesize(a, cases)
    else:
        recognize(a, a.stage)
    summary = json.loads((a.output / f"summary-{a.stage}.json").read_text())
    failures = (
        summary["expected_outputs"] - summary["completed_outputs"]
        if a.stage == "tts"
        else summary["failed"]
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
