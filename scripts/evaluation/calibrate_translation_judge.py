"""Separate capped protocol: repair only unscored judgments and test known bad outputs.

Never rescores a valid judgment. Original artifacts remain unchanged.
"""

import asyncio, json, hashlib
import httpx
from voice_bridge.settings import ROOT, load_settings
from scripts.evaluation import evaluate_cloud_translation as evaluator

CONFIG = {
    **evaluator.CONFIG,
    "judge_think": "low",
    "judge_max_output_tokens": 1024,
    "judge_attempts": 1,
    "maximum_requests": 9,
}


async def main():
    source = ROOT / ".evaluation-local/cloud-v1"
    output = ROOT / ".evaluation-local/judge-calibration-v2"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "judgments.jsonl").exists():
        raise RuntimeError(
            "Existing calibration; preserve it and choose a new explicit protocol rather than rerunning valid scores"
        )
    generations = evaluator.saved(source / "translations.jsonl")
    old = evaluator.saved(source / "judgments.jsonl")
    invalid = {r["case_id"] for r in old if not r["valid"]}
    repair = [row for row in generations if row["case_id"] in invalid]
    assert len(repair) == 3, "Inspect original coverage before changing the repair set"
    controls = []

    def add(identity, source, target, text, reference, translation, correct):
        controls.append(
            {
                "case_id": "calibration-" + identity,
                "source_language": source,
                "target_language": target,
                "source": text,
                "reference_translation": reference,
                "translation": translation,
                "expected_good": correct,
            }
        )

    add(
        "good-en-de",
        "en",
        "de",
        "The train leaves at ten.",
        "Der Zug fährt um zehn Uhr ab.",
        "Der Zug fährt um zehn Uhr ab.",
        True,
    )
    add(
        "good-de-en",
        "de",
        "en",
        "Ich möchte Wasser.",
        "I would like water.",
        "I would like water.",
        True,
    )
    add(
        "negation",
        "en",
        "de",
        "I do not agree with the price.",
        "Ich bin mit dem Preis nicht einverstanden.",
        "Ich bin mit dem Preis einverstanden.",
        False,
    )
    add(
        "amount",
        "en",
        "de",
        "It costs twelve euros, not twenty.",
        "Es kostet zwölf Euro, nicht zwanzig.",
        "Es kostet zwanzig Euro, nicht zwölf.",
        False,
    )
    add(
        "wrong-language",
        "de",
        "en",
        "Ich möchte Wasser.",
        "I would like water.",
        "Ich möchte Wasser.",
        False,
    )
    add(
        "invented-action",
        "en",
        "de",
        "Please do not book the tickets yet.",
        "Bitte buchen Sie die Fahrkarten noch nicht.",
        "Ich habe Ihre Fahrkarten bereits gebucht.",
        False,
    )
    assert len(repair) + len(controls) <= CONFIG["maximum_requests"]
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "protocol": "judge-calibration-v2",
                "parameters": CONFIG,
                "source_translations_sha256": hashlib.sha256(
                    (source / "translations.jsonl").read_bytes()
                ).hexdigest(),
                "repair_case_ids": [r["case_id"] for r in repair],
                "control_cases": controls,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    evaluator.CONFIG = CONFIG
    budget = evaluator.Budget(
        CONFIG["maximum_requests"], output / "request-budget.json"
    )
    settings = load_settings()
    records = []
    async with httpx.AsyncClient(timeout=CONFIG["request_timeout_seconds"]) as client:
        for row in [*repair, *controls]:
            result = await evaluator.judge(row, settings, client, budget)
            result["role"] = (
                "calibration"
                if row["case_id"].startswith("calibration-")
                else "repair_unscored"
            )
            if "expected_good" in row:
                result["expected_good"] = row["expected_good"]
            evaluator.append(output / "judgments.jsonl", result)
            records.append(result)
            print(row["case_id"], result["valid"], result.get("scores"), flush=True)
    combined = [r for r in old if r["valid"]] + [
        r for r in records if r["role"] == "repair_unscored" and r["valid"]
    ]
    controls = [r for r in records if r["role"] == "calibration"]
    checks = []
    for row in controls:
        passed = row["valid"] and (
            min(row["scores"].values()) >= 0.75
            if row["expected_good"]
            else min(row["scores"].values()) <= 0.5
        )
        checks.append({"case_id": row["case_id"], "passed": passed})
    summary = {
        "original_valid_judgments": sum(r["valid"] for r in old),
        "additional_repaired_valid_judgments": sum(
            r["role"] == "repair_unscored" and r["valid"] for r in records
        ),
        "combined_valid_judgments": len(combined),
        "expected_translations": 20,
        "requests": budget.calls,
        "maximum_requests": CONFIG["maximum_requests"],
        "calibration_checks": checks,
        "combined_scores": {
            key: sum(r["scores"][key] for r in combined) / len(combined)
            for key in ["adequacy", "faithfulness", "target_language"]
        },
        "limitations": [
            "Controls are simple synthetic sanity checks, not full human calibration.",
            "Initial invalid/truncated judgments remain preserved; no valid low score was replaced.",
            "Combined means use two recorded decoding protocols; inspect coverage and control failures.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(len(combined) != 20 or not all(r["passed"] for r in checks))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
