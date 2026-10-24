"""Stream a small, deterministic official FLEURS test subset; never download whole archives."""

import argparse, csv, hashlib, io, json, tarfile, urllib.request
from faster_whisper.audio import decode_audio
from pathlib import Path
from voice_bridge.settings import ROOT

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/"


def prepare(count, output, manifest):
    existing = json.loads(manifest.read_text()) if manifest.exists() else None
    if existing:
        assert len(existing["cases"]) == 2 * count, (
            "Count differs from frozen manifest; use another manifest version"
        )
        if all(
            (ROOT / r["audio"]).is_file()
            and hashlib.sha256((ROOT / r["audio"]).read_bytes()).hexdigest()
            == r["audio_sha256"]
            for r in existing["cases"]
        ):
            print("All frozen human inputs already present and verified.", flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for language, config in [("en", "en_us"), ("de", "de_de")]:
        transcript_url = BASE + f"data/{config}/test.tsv"
        with urllib.request.urlopen(transcript_url, timeout=60) as response:
            raw = response.read(3_000_000)
        table = {
            fields[1]: fields
            for fields in csv.reader(io.StringIO(raw.decode()), delimiter="\t")
            if len(fields) >= 4
        }
        archive_url = BASE + f"data/{config}/audio/test.tar.gz"
        selected = 0
        with (
            urllib.request.urlopen(archive_url, timeout=90) as response,
            tarfile.open(fileobj=response, mode="r|gz") as archive,
        ):
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or name not in table:
                    continue
                if member.size > 10_000_000:
                    raise ValueError("Unexpectedly large individual audio sample")
                payload = archive.extractfile(member).read()
                path = output / f"{config}-{name}"
                path.write_bytes(payload)
                rate = 16000
                seconds = len(decode_audio(str(path), sampling_rate=rate)) / rate
                fields = table[name]
                rows.append(
                    {
                        "case_id": f"fleurs-{config}-{fields[0]}-{name[:-4]}",
                        "language": language,
                        "reference": fields[2],
                        "audio": str(path.relative_to(ROOT)),
                        "audio_sha256": hashlib.sha256(payload).hexdigest(),
                        "sample_rate": rate,
                        "audio_seconds": seconds,
                        "source_id": fields[0],
                        "source_file": name,
                        "dataset": "google/fleurs",
                        "revision": REVISION,
                        "split": "test",
                        "archive_url": archive_url,
                    }
                )
                selected += 1
                if selected >= count:
                    break
        assert selected == count
        print(
            f"Prepared {selected} {config} human clips; archive connection closed early.",
            flush=True,
        )
    data = {
        "version": "fleurs-calibration-v1",
        "dataset": "google/fleurs",
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "attribution": "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, Conneau et al., Google (2022).",
        "dataset_card": "https://huggingface.co/datasets/google/fleurs",
        "selection": "First requested count of matching test WAV members in each pinned archive, before recognition; not selected by error rate.",
        "limitations": [
            "Small recognizer calibration/read-speech subset, not a representative population benchmark.",
            "Training overlap is unknown; no unseen-data claim.",
            "No naturally code-switched human speech in this subset.",
        ],
        "cases": rows,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        assert data == existing, "Downloaded references differ from frozen manifest"
    else:
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("Frozen", len(rows), "human references:", manifest, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-language", type=int, default=12)
    p.add_argument(
        "--output", type=Path, default=ROOT / ".evaluation-local/inputs/fleurs"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation/datasets/fleurs-calibration-v1.json",
    )
    a = p.parse_args()
    if not 1 <= a.per_language <= 100:
        p.error("Choose 1..100 clips per language")
    prepare(a.per_language, a.output, a.manifest)


if __name__ == "__main__":
    main()
