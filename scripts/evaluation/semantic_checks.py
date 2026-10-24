"""Case-specific meaning regressions; not a general semantic similarity metric."""
import re
from scripts.evaluation.speech_metrics import normalized_words


def check_meaning(text, specifications):
    normalized = ' '.join(normalized_words(text))
    results = {}
    for item in specifications:
        required = item.get('any_patterns', [])
        forbidden = item.get('forbidden_patterns', [])
        results[item['name']] = (
            (not required or any(re.search(pattern, normalized) for pattern in required))
            and not any(re.search(pattern, normalized) for pattern in forbidden)
        )
    return results
