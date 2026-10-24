"""Real browser speech-to-speech journeys against isolated memory and actual cloud.

Inputs are prerecorded synthetic speech delivered in real time through WebAudio.
No normal user history is loaded or changed. No fake LLM replies are substituted.
"""

import argparse, asyncio, hashlib, json, os, shutil, signal, socket, subprocess, time
from pathlib import Path
import httpx
import numpy as np
from playwright.async_api import async_playwright
from voice_bridge.settings import ROOT, load_config
from scripts.evaluation.audio_tools import resample, write_audio
from scripts.evaluation.speech_metrics import speech_metrics
from scripts.evaluation.semantic_checks import check_meaning
from scripts.evaluation.run_component_benchmarks import streamed_synthesis, digest
from scripts.story_tests.resource_monitor import Resources
from scripts.validation.validate_browser_interface import BOOTSTRAP
from evaluation.config import BROWSER_EVALUATION as CONFIG

INSTRUMENT = r"""
window.__out=[];window.__playback=[];
const SendSocket=window.WebSocket;
window.WebSocket=class extends SendSocket {
 send(data) {if(typeof data==='string')window.__out.push({...JSON.parse(data),at_ms:performance.now()});return super.send(data);}
};
const instrumentedCreate=AudioContext.prototype.createBufferSource;
AudioContext.prototype.createBufferSource=function(...args) {
 const node=instrumentedCreate.apply(this,args),ctx=this,start=node.start.bind(node),stop=node.stop.bind(node);
 node.start=(...values)=>{node.__qaStarted=ctx.currentTime;node.__qaId=(typeof playing!=='undefined'&&playing?.source===node)?playing.id:null;
  if(node.__qaId)window.__playback.push({event:'start',id:node.__qaId,at_ms:performance.now(),audio_seconds:node.buffer.duration});return start(...values);};
 node.stop=(...values)=>{if(node.__qaId)window.__playback.push({event:'stop',id:node.__qaId,at_ms:performance.now(),rendered_seconds:Math.max(0,Math.min(node.buffer.duration,ctx.currentTime-node.__qaStarted))});return stop(...values);};
 return node;
};
"""


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def compact(sequence):
    return [v for i, v in enumerate(sequence) if i == 0 or v != sequence[i - 1]]


def traces(folder):
    output = []
    for path in sorted((folder / "pipeline-traces").glob("*/events.jsonl")):
        for line in path.read_text().splitlines():
            value = json.loads(line)
            value["trace_session"] = path.parent.name
            output.append(value)
    return output


def prepare(args):
    from voice_bridge import workers

    folder = args.inputs
    folder.mkdir(parents=True, exist_ok=True)
    dataset = load(args.dataset)
    manifest = folder / "inputs.json"
    if manifest.exists():
        previous = load(manifest)
        assert previous["dataset_sha256"] == digest(args.dataset)
        assert all(
            digest(ROOT / r["audio"]) == r["audio_sha256"] for r in previous["cases"]
        )
        print("Frozen flow inputs verified; no regeneration.", flush=True)
        return
    cfg = load_config()
    workers.init_tts(cfg["tts"], ROOT)
    workers.warmup_tts()
    cases = []
    for session in dataset["sessions"]:
        for case in session["turns"]:
            captured, _, _ = asyncio.run(
                streamed_synthesis(
                    case, cfg, workers, folder / (case["case_id"] + ".sqlite3")
                )
            )
            pieces = []
            for i, (_, _, (pcm, rate)) in enumerate(captured):
                if i:
                    pieces.append(
                        np.zeros(dataset["input_chunk_gap_ms"] * 16, np.float32)
                    )
                pieces.append(
                    resample(
                        np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768,
                        rate,
                        16000,
                    )
                )
            speech = np.concatenate(pieces)
            if "snr_db" in case:
                seed = dataset["noise_seed"] ^ int(
                    hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16
                )
                rms = float(np.sqrt(np.mean(speech * speech)))
                noise = (
                    np.random.default_rng(seed)
                    .normal(0, rms / (10 ** (case["snr_db"] / 20)), len(speech))
                    .astype(np.float32)
                )
                speech = speech + noise
            audio = np.concatenate(
                [
                    np.zeros(dataset["input_prefix_ms"] * 16, np.float32),
                    speech,
                    np.zeros(dataset["input_suffix_ms"] * 16, np.float32),
                ]
            )
            path = folder / (case["case_id"] + ".wav")
            sha = write_audio(path, audio, 16000)
            cases.append(
                {
                    **case,
                    "session": session["id"],
                    "audio": str(path.relative_to(ROOT)),
                    "audio_sha256": sha,
                    "audio_seconds": len(audio) / 16000,
                }
            )
            save(folder / "preparation-progress.json", cases)
            print("prepared", case["case_id"], round(len(audio) / 16000, 2), flush=True)
    save(
        manifest,
        {
            "dataset_sha256": digest(args.dataset),
            "source": "Production Kokoro/Piper phrase pipeline; synthetic speaker voices, not real microphone speech.",
            "cases": cases,
        },
    )


async def state(page):
    return await page.evaluate("fetch('/state').then(r=>r.json())")


async def execute(page, case, folder, monitor):
    if case.get("reload_before"):
        await page.reload(wait_until="networkidle")
        await page.locator("#start").click()
        await page.wait_for_function(
            "active && context.state==='running'", timeout=30000
        )
    if not case.get("barge_in"):
        await page.wait_for_function(
            "!playing && !queue.length && !responsePending",
            timeout=CONFIG["turn_timeout_ms"],
        )
    else:
        await page.wait_for_function("Boolean(playing)", timeout=10000)
    await page.wait_for_timeout(
        100 if case.get("barge_in") else CONFIG["idle_settle_ms"]
    )
    before = await page.evaluate("__events.length")
    sent = await page.evaluate("__out.length")
    began = time.monotonic()
    monitor.label = case["case_id"]
    failure = None
    prior = await page.evaluate("({active,turn,playing_id:playing?.id??null})")
    timing = {}
    try:
        timing = await asyncio.wait_for(
            page.evaluate(
                "async url=>{const start=performance.now();await __say(url);return {start_ms:start,end_ms:performance.now()}}",
                "/__flow__/" + case["case_id"] + ".wav",
            ),
            timeout=case['audio_seconds'] + CONFIG['input_timeout_margin_seconds'],
        )
        if case.get("leave_playing"):
            await page.wait_for_function(
                "Boolean(playing)", timeout=CONFIG["turn_timeout_ms"]
            )
        else:
            await page.wait_for_function(
                "n=>(__events.slice(n).some(e=>e.type==='audio'||e.type==='error'||e.type==='notice'))&&!playing&&!queue.length&&!responsePending&&document.body.dataset.state!=='hearing'",
                arg=before,
                timeout=CONFIG["turn_timeout_ms"],
            )
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)[:500]}
    ended = time.monotonic()
    events = await page.evaluate("n=>__events.slice(n)", before)
    acks = await page.evaluate("n=>__out.slice(n)", sent)
    current = await state(page)
    trace = [e for e in traces(folder) if began <= e["monotonic_seconds"] <= ended]
    finals = [e for e in events if e["type"] == "final"]
    recognized = " ".join(e["text"] for e in finals)
    response_turn = await page.evaluate("turn")
    answer = "".join(
        e["text"]
        for e in events
        if e["type"] == "token" and e.get("turn") == response_turn
    )
    audio = [
        e for e in events if e["type"] == "audio" and e.get("turn") == response_turn
    ]
    languages = [e["language"] for e in audio]
    checks = {
        "no_browser_or_pipeline_timeout": failure is None,
        "final_transcript_present": bool(finals),
        "one_input_turn": len(finals) == 1,
        "no_pipeline_errors": not any(e["type"] == "error" for e in events),
        "mode_correct": current["mode"] == case["expected_mode"],
        "audio_emitted": bool(audio),
    }
    if "pending_confirmation" in case:
        checks["confirmation_state_correct"] = (
            bool(current.get("pending_confirmation")) == case["pending_confirmation"]
        )
    if case["expected_audio_languages"]:
        checks["requested_audio_languages_present"] = set(
            case["expected_audio_languages"]
        ) <= set(languages)
    if len(case["expected_audio_languages"]) == 1:
        checks["no_unrequested_voice"] = set(languages) <= set(
            case["expected_audio_languages"]
        )
    if "expected_language_sequence" in case:
        checks["voice_switch_order_correct"] = (
            compact(languages) == case["expected_language_sequence"]
        )
    for i, alternatives in enumerate(case["required_answer_alternatives"]):
        checks[f"answer_content_{i + 1}"] = any(
            part in answer.lower() for part in alternatives
        )
    for name, passed in check_meaning(answer, case.get('semantic_checks', [])).items():
        checks['meaning_'+name] = passed
    if case.get("expected_note"):
        checks["explicit_note_present"] = any(
            case["expected_note"].lower() in r["text"].lower() for r in current["facts"]
        )
    completed = {p["id"] for p in acks if p["type"] == "played" and p.get("complete")}
    if not case.get("leave_playing"):
        checks["emitted_audio_finished_playback"] = all(
            p["id"] in completed for p in audio
        )
    interruption = None
    if case.get("barge_in"):
        interrupted = [
            p for p in acks if p["type"] == "played" and not p.get("complete")
        ]
        packets = [
            p
            for p in events
            if p["type"] == "interrupt" and p["at_ms"] >= timing.get("start_ms", 0)
        ]
        checks["speech_started_during_reply"] = prior["playing_id"] is not None
        checks["old_reply_playback_cancelled"] = any(
            p["id"] == prior["playing_id"] for p in interrupted
        )
        checks["interrupt_packet_received"] = bool(packets)
        interruption = {
            "old_playing_id": prior["playing_id"],
            "cancelled_chunks": interrupted,
            "input_start_to_interrupt_seconds": (
                packets[0]["at_ms"] - timing["start_ms"]
            )
            / 1000
            if packets and timing
            else None,
        }
    if case.get("idle_after_seconds"):
        start_idle = await page.evaluate("__events.length")
        await page.wait_for_timeout(case["idle_after_seconds"] * 1000)
        quiet = await page.evaluate("n=>__events.slice(n)", start_idle)
        checks["silent_microphone_stays_active"] = await page.evaluate("active")
        checks["no_false_turn_during_silence"] = not any(
            e["type"] in ["final", "audio", "error"] for e in quiet
        )
    first_audio = audio[0]["at_ms"] if audio else None
    times = {
        "recording_seconds": case["audio_seconds"],
        "request_to_test_completion_seconds": ended - began,
        "first_audio_after_input_recording_end_seconds": (
            first_audio - timing["end_ms"]
        )
        / 1000
        if first_audio is not None and timing
        else None,
    }
    for i, event in enumerate(trace):
        if event["event"] == "llm_messages":
            first = next(
                (
                    e
                    for e in trace[i + 1 :]
                    if e["event"] == "llm_raw" and e.get("text")
                ),
                None,
            )
            if first:
                times["cloud_first_text_seconds"] = (
                    first["monotonic_seconds"] - event["monotonic_seconds"]
                )
                break
    row = {
        **case,
        "checks": checks,
        "passed": all(checks.values()),
        "hypothesis": recognized,
        "stt_metrics": speech_metrics(case["reference"], recognized),
        "visible_answer": answer,
        "response_turn": response_turn,
        "raw_llm_content": "".join(
            e["text"]
            for e in trace
            if e["event"] == "llm_raw"
            and e.get("text")
            and e.get("turn") == response_turn
        ),
        "audio_languages": languages,
        "browser_events": events,
        "playback_acks": acks,
        "interruption": interruption,
        "timing": times,
        "failure": failure,
        "browser_connection": await page.evaluate("({active,socket_state:ws?.readyState,buffered_bytes:ws?.bufferedAmount,error:document.getElementById('error')?.textContent})"),
        "resources": monitor.summarize(began, ended),
        "pipeline_trace": trace,
        "synthetic_memory_state": current,
    }
    (folder / "cases").mkdir(exist_ok=True)
    save(folder / "cases" / (case["case_id"] + ".json"), row)
    print(
        json.dumps(
            {
                "case_id": case["case_id"],
                "passed": row["passed"],
                "wer": row["stt_metrics"]["wer"],
                "failed_checks": [k for k, v in checks.items() if not v],
            }
        ),
        flush=True,
    )
    return row


async def run(args):
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "results.json").exists():
        raise RuntimeError(
            "Preserve the existing experiment; choose a new output folder"
        )
    dataset = load(args.dataset)
    inputs = load(args.inputs / "inputs.json")
    assert inputs["dataset_sha256"] == digest(args.dataset)
    by_id = {r["case_id"]: r for r in inputs["cases"]}
    for row in inputs["cases"]:
        assert digest(ROOT / row["audio"]) == row["audio_sha256"]
    sessions = [
        s for s in dataset["sessions"] if not args.session or s["id"] in args.session
    ]
    results = []
    errors = []
    listener = socket.socket()
    occupied = listener.connect_ex(("127.0.0.1", CONFIG["port"])) == 0
    listener.close()
    if occupied and not args.external_server:
        raise RuntimeError("QA port already in use; no data was changed")
    if args.external_server and not occupied:
        raise RuntimeError("External QA server is not listening on the configured localhost port")
    process_env = os.environ.copy()
    process_env["VOICE_BRIDGE_QA_DIR"] = str(folder.relative_to(ROOT))
    process_env["VOICE_BRIDGE_QA_AUDIT_FINAL"] = "0"
    process_env["VOICE_BRIDGE_QA_MAX_REQUESTS"] = str(CONFIG["maximum_cloud_requests"])
    source=hashlib.sha256()
    for path in [ROOT/'config.py',ROOT/'scripts/runtime/qwen_asr_worker.py',ROOT/'scripts/validation/start_test_server.py',*sorted((ROOT/'src/voice_bridge').glob('*.py'))]:source.update(path.relative_to(ROOT).as_posix().encode()+path.read_bytes())
    save(
        folder / "configuration.json",
        {
            "dataset_sha256": inputs["dataset_sha256"],
            "dataset": str(args.dataset.relative_to(ROOT)),
            "input_directory": str(args.inputs.relative_to(ROOT)),
            "application_source_sha256": source.hexdigest(),
            "selected_sessions": [s["id"] for s in sessions],
            "parameters": CONFIG,
            "qa_asr_overrides": {key:process_env[key] for key in ('VOICE_BRIDGE_QA_STREAM_CACHE','VOICE_BRIDGE_QA_ASR_BATCH_SIZE') if key in process_env},
            "external_server": args.external_server,
            "resource_scope": "Browser runner process tree only; external container excluded" if args.external_server else "Browser runner and model worker process tree",
            "browser_microphone": "Real-time prerecorded synthetic WebAudio source; silent real-time output sink, not a physical microphone/speaker test.",
        },
    )
    async with async_playwright() as engine:
        browser = await engine.chromium.launch(
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--autoplay-policy=no-user-gesture-required",
            ]
        )
        server_log = (folder / "server.log").open("w")
        server = None if args.external_server else subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "scripts.application.start_voice_app",
                "-m",
                "scripts.validation.start_test_server",
            ],
            cwd=ROOT,
            env=process_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor = Resources(
            os.getpid(), folder / "resources.jsonl", CONFIG["resource_interval_seconds"]
        )
        monitor.start()
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                deadline = time.monotonic() + CONFIG["startup_timeout_seconds"]
                while True:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError("QA server exited; inspect server.log")
                    try:
                        if (
                            await client.get(
                                f"http://127.0.0.1:{CONFIG['port']}/health"
                            )
                        ).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("QA server startup timeout")
                    await asyncio.sleep(0.5)
            for session in sessions:
                context = await browser.new_context(
                    viewport={"width": 1400, "height": 1050}
                )
                bootstrap = BOOTSTRAP.replace(
                    "super(...args);window.__ctx=this;",
                    "super({sinkId:{type:'none'}});window.__ctx=this;",
                ).replace(
                    "window.__events.push(safe)",
                    "window.__events.push({...safe,at_ms:performance.now()})",
                )
                await context.add_init_script(bootstrap + INSTRUMENT)
                page = await context.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))

                async def serve_audio(route):
                    identity = Path(route.request.url).name.removesuffix(".wav")
                    row = by_id.get(identity)
                    if row:
                        await route.fulfill(
                            path=str(ROOT / row["audio"]), content_type="audio/wav"
                        )
                    else:
                        await route.abort()

                await page.route("**/__flow__/*", serve_audio)
                await page.goto(
                    f"http://127.0.0.1:{CONFIG['port']}", wait_until="networkidle"
                )
                if await page.locator("#confirmation-card").is_visible():
                    await page.locator("#cancel-change").click()
                await page.locator(f'[data-mode="{session["initial_mode"]}"]').click()
                await page.wait_for_function(
                    "mode=>savedState?.mode===mode", arg=session["initial_mode"]
                )
                response = await page.evaluate(
                    "fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'clear_memory'})}).then(r=>r.status)"
                )
                assert response == 200
                await page.locator("#start").click()
                await page.wait_for_function(
                    "active && context.state==='running'", timeout=30000
                )
                selected = (
                    session["turns"][: args.limit_turns]
                    if args.limit_turns
                    else session["turns"]
                )
                for case in selected:
                    if (
                        Path("/mnt/c").exists()
                        and shutil.disk_usage("/mnt/c").free < 128 * 1024**2
                    ):
                        raise RuntimeError(
                            "Host C has less than128MiB free; partial results preserved"
                        )
                    result = await execute(
                        page, by_id[case["case_id"]], folder, monitor
                    )
                    results.append(result)
                    save(
                        folder / "results.json",
                        {
                            "dataset_sha256": inputs["dataset_sha256"],
                            "cases": results,
                            "javascript_errors": errors,
                            "complete": False,
                        },
                    )
                await page.screenshot(
                    path=str(folder / (session["id"] + ".png")), full_page=True
                )
                save(
                    folder / (session["id"] + "-playback.json"),
                    await page.evaluate("__playback"),
                )
                if await page.evaluate("active"):
                    await page.locator("#start").click()
                await context.close()
                await asyncio.sleep(0.5)
            save(
                folder / "results.json",
                {
                    "dataset_sha256": inputs["dataset_sha256"],
                    "cases": results,
                    "javascript_errors": errors,
                    "complete": True,
                },
            )
        finally:
            monitor.stop()
            await browser.close()
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=10)
            server_log.close()
    summary = {
        "selected_cases": sum(
            len(s["turns"][: args.limit_turns] if args.limit_turns else s["turns"])
            for s in sessions
        ),
        "completed_cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "javascript_errors": errors,
        "failed_cases": [
            {
                "case_id": r["case_id"],
                "checks": [k for k, v in r["checks"].items() if not v],
            }
            for r in results
            if not r["passed"]
        ],
        "scope": "Actual browser, model workers and cloud; isolated synthetic memory and real-time prerecorded microphone. Browser playback uses silent sink.",
    }
    save(folder / "summary.json", summary)
    return int(bool(errors) or summary["passed"] != summary["selected_cases"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["prepare", "run"], default="run")
    p.add_argument("--dataset", type=Path, default=CONFIG["dataset"])
    p.add_argument("--inputs", type=Path, default=CONFIG["input_directory"])
    p.add_argument("--output", type=Path, default=CONFIG["output_directory"])
    p.add_argument("--session", action="append")
    p.add_argument("--limit-turns", type=int)
    p.add_argument("--external-server", action="store_true", help="Use an already running QA server on localhost; do not start or stop it")
    a = p.parse_args()
    a.dataset = (ROOT / a.dataset).resolve()
    a.inputs = (ROOT / a.inputs).resolve()
    a.output = (ROOT / a.output).resolve()
    if a.stage == "prepare":
        prepare(a)
        return 0
    return asyncio.run(run(a))


if __name__ == "__main__":
    raise SystemExit(main())
