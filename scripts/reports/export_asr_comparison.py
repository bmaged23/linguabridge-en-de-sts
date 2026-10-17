"""Publish synthetic test measurements to local docs without audio or credentials."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTED = 'Qwen3-ASR-0.6B-progressive-float32-rollback5-context63fcc93a-report.json'


def main():
    archive = {}
    for suite in ('codeswitch-check', 'codeswitch-holdout'):
        folder = ROOT / 'logs' / suite
        archive[suite] = {path.name: json.loads(path.read_text()) for path in sorted(folder.glob('*report.json'))}
        durations = {case['name']: case['seconds'] for case in archive[suite]['report.json']['cases']}
        for report in archive[suite].values():
            for case in report['cases']:
                if 'updates' in case and 'replay_walltime_ratio' not in case:
                    # Earlier prototypes reported replay wall time (including arrival
                    # waits) as RTF. Preserve it and calculate actual decoding RTF.
                    case['replay_walltime_ratio'] = case['rtf']
                    case['rtf'] = round(sum(update['decode_seconds'] for update in case['updates']) / durations[case['name']], 3)
    (ROOT / 'docs/reports/data/asr-evaluation-results.json').write_text(json.dumps(archive, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Local ASR comparison', '', 'Measured on 2026-10-16 using the GTX 1660 Ti (6 GiB).', '',
        'These are synthetic diagnostics, not a public benchmark or an estimate of human microphone accuracy. English Kokoro and German Piper voices change at language boundaries. WER/CER lowercase and remove punctuation, but do not normalize numbers or German compound spelling. Zero errors means an exact match after this normalization.', '',
        '## Selected experimental configuration', '',
        'Qwen3-ASR-0.6B, CUDA FP32, eager attention, one request at a time, two-second partial-update interval. Earlier generated text is reused after two partials with five trailing tokens allowed to change. Audio remains cumulative; this is a Transformers prototype, not native vLLM streaming. It skips stale snapshots rather than accumulating a queue.', '',
        'Fixed ASR context (no reference transcript is supplied):', '',
        '> Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate.', '',
        'This context was selected on the original four mixed examples and then frozen before evaluating four new examples. No model weights were trained.', '']
    for suite, title in [('codeswitch-check', 'Original mixed examples'), ('codeswitch-holdout', 'New held-out mixed examples')]:
        records = archive[suite]
        selected = records[SELECTED]
        baseline = {case['name']: case for case in records['report.json']['cases']}
        lines += ['## ' + title, '', '| Recording | Whisper WER / CER | Qwen WER / CER | First partial from start | Final after audio end |', '| --- | ---: | ---: | ---: | ---: |']
        for case in selected['cases']:
            old = baseline[case['name']]['results']['auto']
            lines.append(f'| {case["name"]} | {old["wer"]:.1%} / {old["cer"]:.1%} | {case["wer"]:.1%} / {case["cer"]:.1%} | {case["first_partial_seconds"]:.2f} s | {case["final_ready_after_audio_seconds"]:.2f} s |')
        lines += ['', 'These final delays exclude VAD endpoint waiting, cloud LLM latency, and TTS. Peak PyTorch allocated memory in this run: '
            + str(max(case['peak_allocated_mib'] for case in selected['cases'])) + ' MiB. CUDA context and other process allocations are additional.', '']
    standalone = archive['codeswitch-check']['Qwen3-ASR-0.6B-float32-context63fcc93a-report.json']
    lines += ['## Separate English and German', '', '| Recording | Qwen WER | Qwen CER | Complete-utterance decode |', '| --- | ---: | ---: | ---: |']
    for case in standalone['cases']:
        if case['name'].startswith(('english_', 'german_')):
            lines.append(f'| {case["name"]} | {case["wer"]:.1%} | {case["cer"]:.1%} | {case["elapsed_seconds"]:.2f} s |')
    lines += ['', 'The same verbatim hint was used. Completed-utterance decoding still dropped mixed speech in some examples, so the progressive prefix-reuse behavior matters; the hint alone is insufficient.', '', '## Other configurations tested', '',
        '- Whisper large-v3-turbo: about 1.17–1.43 seconds per completed 7.4–9.9-second utterance in the existing app decoding test. It dropped whole language spans in several mixed examples.',
        '- Qwen0.6B FP16: roughly 2.4–3.8 seconds for complete utterances except a prematurely truncated result. Short partials produced punctuation and severe latency; this setting was rejected.',
        '- Qwen1.7B FP16: roughly 3.4–5.0 seconds per complete utterance, with translation and omission errors on mixed examples. Larger was not better for this diagnostic.',
        '- Qwen0.6B FP32 without context: complete utterances took roughly 1.8–2.9 seconds except the truncated result. Progressive prefix reuse corrected three of four mixed examples, but lost the German → English → German travel passage.',
        '- A generic bilingual hint recovered that travel passage but translated a German grammar example into English. The explicit verbatim hint above was selected instead.', '',
        'All trial references, transcripts, checkpoint revisions, package versions, update timing, and measured allocation are preserved in [asr-evaluation-results.json](data/asr-evaluation-results.json). These trials are development experiments; prompt selection on the original examples can overfit.', '',
        '## Reproduction', '', 'Use the separate `.venv-asr` with [exact evaluation dependencies](../../requirements/archive/asr-evaluation.txt). It does not replace the app environment.', '', '```bash',
        '.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.evaluate_mixed_language_asr',
        '.venv/bin/python -m scripts.application.start_voice_app -m scripts.evaluation.evaluate_asr_holdout',
        '.venv-asr/bin/python -m scripts.evaluation.evaluate_qwen_asr --dtype float32 --progressive --rollback-tokens 5 --context "Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate."',
        '.venv-asr/bin/python -m scripts.evaluation.evaluate_qwen_asr --suite codeswitch-holdout --dtype float32 --progressive --rollback-tokens 5 --context "Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate."',
        '.venv/bin/python -m scripts.reports.export_asr_comparison', '```', '',
        '## Practical limits', '',
        'Earlier text can be retained incorrectly by prefix reuse. Partials must remain replaceable and must not be sent to the conversation LLM as independent user turns. A completed turn still requires VAD endpoint detection. Test real human English, German, and mixed microphone speech before treating these measurements as representative.', '',
        'This experiment does not change the app recognizer, prompts, or user memory. See [primary sources and alternative models](asr-model-research.md).', '']
    (ROOT / 'docs/reports/asr-evaluation.md').write_text('\n'.join(lines))
    print('Exported docs/reports/asr-evaluation.md and docs/reports/data/asr-evaluation-results.json')


if __name__ == '__main__':
    main()
