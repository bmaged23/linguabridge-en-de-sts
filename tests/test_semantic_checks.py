from scripts.evaluation.semantic_checks import check_meaning


STORY = [
    {'name': 'cancelled', 'any_patterns': [r'zug fiel aus', r'train was cancel[el]*d'],
     'forbidden_patterns': [r'train was full']},
    {'name': 'not_bought', 'any_patterns': [r'keine fahrkarten', r'did not buy.*tickets']},
]


def test_entities_do_not_hide_changed_meaning():
    text = 'Anna and I reached Berlin. The train was full and we bought the tickets.'
    assert check_meaning(text, STORY) == {'cancelled': False, 'not_bought': False}


def test_bilingual_facts_survive_normalization():
    text = 'Anna reached Berlin. Unser Zug fiel aus; wir kauften noch KEINE Fahrkarten.'
    assert check_meaning(text, STORY) == {'cancelled': True, 'not_bought': True}


def test_negation_cannot_be_dropped():
    specs = [{'name': 'not_cancelled', 'any_patterns': [r'nicht abgesagt']}]
    assert check_meaning('Der Termin wurde nicht abgesagt.', specs)['not_cancelled']
    assert not check_meaning('Der Termin wurde abgesagt.', specs)['not_cancelled']


def test_browser_story_rejects_previously_passing_wrong_summary():
    import json
    from pathlib import Path
    dataset = json.loads((Path(__file__).resolve().parents[1]/'evaluation/datasets/bilingual-flows-v2.json').read_text())
    case = next(case for session in dataset['sessions'] for case in session['turns']
                if case['case_id']=='mixed-connected-story')
    result = check_meaning('Anna and I went to Berlin. Our train was full and we arrived without tickets.',
                           case['semantic_checks'])
    assert result == {'train_cancelled': False, 'tickets_not_bought': True}
