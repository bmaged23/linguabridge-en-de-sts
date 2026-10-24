"""Reference-based speech metrics; empty-reference hallucinations are scored separately."""

import re
import unicodedata


def normalized_words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def edit_counts(reference, hypothesis):
    """Minimum Levenshtein alignment; ties prefer match/substitution, deletion, insertion."""
    previous = [(j, 0, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(i, 0, i, 0, 0)]
        for j, right in enumerate(hypothesis, 1):
            cost, s, d, ins, h = previous[j - 1]
            diagonal = (
                cost + (left != right),
                s + (left != right),
                d,
                ins,
                h + (left == right),
            )
            cost, s, d, ins, h = previous[j]
            deletion = (cost + 1, s, d + 1, ins, h)
            cost, s, d, ins, h = current[-1]
            insertion = (cost + 1, s, d, ins + 1, h)
            current.append(min([diagonal, deletion, insertion], key=lambda x: x[0]))
        previous = current
    cost, s, d, ins, h = previous[-1]
    return {
        "errors": cost,
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "hits": h,
    }


def speech_metrics(reference, hypothesis):
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    words = edit_counts(ref, hyp)
    chars = edit_counts(" ".join(ref), " ".join(hyp))
    return {
        "wer": words["errors"] / len(ref) if ref else None,
        "cer": chars["errors"] / len(" ".join(ref)) if ref else None,
        "reference_words": len(ref),
        "reference_characters": len(" ".join(ref)),
        "word_errors": words["errors"],
        "character_errors": chars["errors"],
        **{k: words[k] for k in ["substitutions", "deletions", "insertions", "hits"]},
        "normalized_exact_match": ref == hyp,
        "empty_hypothesis": not hyp,
        "hallucinated_words_on_empty_reference": len(hyp) if not ref else 0,
    }


def aggregate_metrics(rows):
    values = [r["metrics"] if "metrics" in r else r for r in rows]
    scored = [r for r in values if r["reference_words"] > 0]
    ref = sum(r["reference_words"] for r in scored)
    chars = sum(r["reference_characters"] for r in scored)
    return {
        "samples": len(values),
        "speech_samples": len(scored),
        "empty_reference_samples": len(values) - len(scored),
        "reference_words": ref,
        "reference_characters": chars,
        "wer": sum(r["word_errors"] for r in scored) / ref if ref else None,
        "cer": sum(r["character_errors"] for r in scored) / chars if chars else None,
        **{
            k: sum(r[k] for r in scored)
            for k in [
                "word_errors",
                "character_errors",
                "substitutions",
                "deletions",
                "insertions",
                "hits",
            ]
        },
        "hallucinated_words_on_empty_reference": sum(
            r["hallucinated_words_on_empty_reference"] for r in values
        ),
        "normalized_exact_match_rate": sum(r["normalized_exact_match"] for r in values)
        / len(values)
        if values
        else None,
        "empty_hypotheses": sum(r["empty_hypothesis"] for r in values),
    }
