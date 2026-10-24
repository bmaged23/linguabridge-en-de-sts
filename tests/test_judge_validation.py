import json
import pytest
from scripts.evaluation.judge_validation import parse_assessment


@pytest.mark.parametrize('value', [
    {'quality': True, 'reason': 'Boolean is not a numeric rating.'},
    {'quality': 5, 'reason': 'Out of range.'},
    {'quality': 4},
    {'quality': 4, 'reason': ''},
    ['quality', 'reason'],
])
def test_invalid_assessments_cannot_enter_score_means(value):
    with pytest.raises(ValueError):
        parse_assessment(json.dumps(value), ('quality',), 'stop')


def test_truncated_assessment_is_rejected_even_when_json_parses():
    with pytest.raises(ValueError):
        parse_assessment('{"quality":4,"reason":"Looks valid"}', ('quality',), 'length')


def test_inapplicable_score_remains_null():
    value = parse_assessment('{"quality":null,"reason":"Not applicable"}', ('quality',), 'stop')
    assert value['quality'] is None
