"""Ordered actual-cloud regressions for completeness and English/German voice spans."""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
from voice_bridge.memory import Memory
from voice_bridge.routing import build_messages
from voice_bridge.settings import ROOT, load_settings
from voice_bridge.speech_segments import LanguageSegments
from evaluation.config import TEACHING_REGRESSION as CONFIG
from scripts.evaluation.evaluate_cloud_translation import Budget, StopCloud


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.jsonl').exists():
        raise RuntimeError('Keep previous results and choose a new output directory')
    dataset_path = ROOT / CONFIG['dataset']
    dataset = json.loads(dataset_path.read_text())
    settings = load_settings()
    memory = Memory(output / 'memory.sqlite3', settings['memory'], default_mode='teaching')
    budget = Budget(CONFIG['maximum_requests'], output / 'budget.json')
    source = hashlib.sha256()
    for path in (ROOT/'src/voice_bridge/routing.py', ROOT/'src/voice_bridge/speech_segments.py'):
        source.update(path.read_bytes())
    (output / 'configuration.json').write_text(json.dumps({
        'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        'application_prompt_sha256': source.hexdigest(), 'parameters': CONFIG,
        'model': settings['llm']['model']}, indent=2) + '\n')
    rows = []
    try:
        async with httpx.AsyncClient(timeout=CONFIG['request_timeout_seconds']) as client:
            for case in dataset['cases']:
                messages = build_messages(settings, memory, 'teaching', case['source'], case['source'], case['text'])
                llm = settings['llm']
                budget.take()
                began = time.monotonic()
                parts = []
                done = False
                row = {'case_id': case['case_id'], 'request': case['text']}
                async with client.stream('POST', llm['base_url']+'/api/chat',
                    headers={'Authorization': 'Bearer '+llm['api_key']},
                    json={'model': llm['model'], 'messages': messages, 'stream': True,
                          'think': llm['think'], 'options': {'temperature': llm['temperature'],
                          'num_predict': llm['max_response_tokens']}}) as response:
                    if response.status_code in (401, 403, 429):
                        raise StopCloud('Teaching regression HTTP '+str(response.status_code))
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        packet = json.loads(line)
                        content = packet.get('message', {}).get('content', '')
                        if content:
                            parts.append(content)
                        if packet.get('done'):
                            done = packet.get('done_reason') != 'length'
                            row['output_tokens'] = packet.get('eval_count')
                            break
                raw = ''.join(parts)
                events = LanguageSegments(case['source']).push(raw, final=True)
                spans = []
                for language, text, _ in events:
                    if not text:
                        continue
                    if spans and spans[-1][0] == language:
                        spans[-1][1] += text
                    else:
                        spans.append([language, text])
                visible = ''.join(text for _, text in spans)
                languages = {language for language, text in spans if any(c.isalpha() for c in text)}
                checks = {'complete_provider_stream': done,
                          'expected_voice_languages': languages == set(case['expected_languages'])}
                for index, terms in enumerate(case['required_terms']):
                    checks[f'requested_item_{index+1}'] = any(word.lower() in visible.lower() for word in terms)
                english = ' '.join(text for language, text in spans if language == 'en').lower()
                checks['known_german_words_not_sent_to_english_voice'] = not any(
                    word.lower() in english for word in case['forbidden_terms_in_english'])
                row.update(raw_llm_content=raw, spans=spans, visible_answer=visible, checks=checks,
                           passed=all(checks.values()), seconds=time.monotonic()-began)
                rows.append(row)
                with (output/'results.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                turn = memory.add_turn('teaching', case['source'], case['source'], case['text'])
                memory.append_heard(turn, visible)
                print(json.dumps({'case_id': case['case_id'], 'passed': row['passed'],
                                  'failed_checks': [k for k,v in checks.items() if not v]}), flush=True)
    finally:
        memory.close()
    summary = {'cases':len(rows), 'passed':sum(r['passed'] for r in rows),
               'scope':'Real cloud and actual app prompt/parser; synthetic text inputs, no speech recognition or playback. '
                       'Keyword/known-language regression checks, not universal semantic or language classification.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    asyncio.run(main())
