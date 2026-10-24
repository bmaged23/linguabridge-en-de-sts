from scripts.evaluation.speech_metrics import speech_metrics, aggregate_metrics


def test_known_edit_operations_and_corpus_weighting():
    row = speech_metrics("the red train leaves today", "the blue train leaves")
    assert (
        row["substitutions"] == 1 and row["deletions"] == 1 and row["insertions"] == 0
    )
    assert row["wer"] == 0.4
    exact = speech_metrics("Hallo", "Hallo")
    assert aggregate_metrics([row, exact])["wer"] == 2 / 6


def test_german_unicode_and_empty_reference_hallucination():
    assert speech_metrics("Ärztin, Grüße!", "Ärztin Grüße")["wer"] == 0
    result = speech_metrics("", "Unexpected speech")
    assert result["wer"] is None and result["cer"] is None
    assert result["hallucinated_words_on_empty_reference"] == 2
    assert speech_metrics("eins", "eins zwei drei")["wer"] == 2


def test_silence_hallucinations_do_not_pollute_speech_denominator():
    result = aggregate_metrics(
        [speech_metrics("hello", "hello"), speech_metrics("", "invented words")]
    )
    assert result["wer"] == 0 and result["speech_samples"] == 1
    assert result["hallucinated_words_on_empty_reference"] == 2
    assert speech_metrics("Maße", "Masse")["wer"] == 1
