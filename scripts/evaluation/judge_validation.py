"""Strict validation of public judge assessments; invalid scores stay unscored."""
import json


def parse_assessment(raw, dimensions, done_reason):
    value = json.loads(raw)
    if done_reason == 'length' or not isinstance(value, dict):
        raise ValueError('Truncated or non-object judge assessment')
    if set(value) != {*dimensions, 'reason'}:
        raise ValueError('Missing or unexpected judge fields')
    if not all(value[k] is None or type(value[k]) is int and 0 <= value[k] <= 4 for k in dimensions):
        raise ValueError('Scores must be integers 0..4 or null')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Judge explanation is required')
    return value
