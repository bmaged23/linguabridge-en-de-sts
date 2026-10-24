"""Cloud-only evaluation budgets; isolated from application runtime configuration."""

CLOUD_EVALUATION = {
    "reference_dataset": "evaluation/datasets/translation-v1.json",
    "output_directory": ".evaluation-local/cloud-v2",
    "judge_model": "gpt-oss:120b",
    "judge_max_output_tokens": 1024,
    "judge_think": "low",
    "judge_attempts": 2,
    "request_timeout_seconds": 60,
    "maximum_requests": 48,
    "concurrency": 1,
    "chrF_character_order": 6,
    "chrF_word_order": 2,
    "chrF_beta": 2,
}

BROWSER_EVALUATION = {
    "dataset": "evaluation/datasets/bilingual-flows-v2.json",
    "input_directory": ".evaluation-local/flow-inputs-v2",
    "output_directory": ".evaluation-local/semantic-flow-v2",
    "port": 8767,
    "startup_timeout_seconds": 180,
    "turn_timeout_ms": 120000,
    "idle_settle_ms": 500,
    "resource_interval_seconds": 0.5,
    "maximum_cloud_requests": 20,
}

FLOW_JUDGE = {
    'model': 'gpt-oss:120b',
    'think': 'low',
    'max_output_tokens': 1536,
    'maximum_requests': 17,
    'timeout_seconds': 90,
    'temperature': 0,
}

TEACHING_REGRESSION = {
    'dataset': 'evaluation/datasets/teaching-regressions-v2.json',
    'maximum_requests': 12,
    'request_timeout_seconds': 60,
}

MIXED_ASR_EVALUATION = {
    'dataset': 'evaluation/datasets/mixed-asr-robustness-v1.json',
    'input_directory': '.evaluation-local/mixed-asr-inputs-v1',
    'maximum_decodes': 200,
    'expected_guarded_waveforms': 52,
    'gap_ms': 100,
    'padding_ms': 200,
    'resource_interval_seconds': 0.25,
    'reference_context': 'Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate.',
    'methods': {
        'configured_greedy': {'context': 'configured', 'num_beams': 1},
        'empty_greedy': {'context': '', 'num_beams': 1},
        'configured_beam3': {'context': 'configured', 'num_beams': 3},
        'empty_beam3': {'context': '', 'num_beams': 3},
        'canonical_greedy': {'context': 'configured', 'num_beams': 1, 'padding_ms': 200},
        'canonical_beam3': {'context': 'configured', 'num_beams': 3, 'padding_ms': 200},
        'pause_greedy': {'context': 'configured', 'num_beams': 1, 'padding_ms': 200, 'strategy': 'pause_chunks'},
        'pause_beam3': {'context': 'configured', 'num_beams': 3, 'padding_ms': 200, 'strategy': 'pause_chunks'},
        'adaptive_pause_beam3': {'context': 'configured', 'num_beams': 3, 'padding_ms': 200, 'strategy': 'adaptive_pause_chunks', 'speech_gate': False},
        'speech_gated_pause_beam3': {'context': 'configured', 'num_beams': 3, 'padding_ms': 200, 'strategy': 'adaptive_pause_chunks', 'speech_gate': True},
        'speech_gated_pause_batch2': {'context': 'configured', 'num_beams': 3, 'padding_ms': 200, 'strategy': 'adaptive_pause_chunks', 'speech_gate': True, 'batch_size': 2},
    },
    'pause_segmentation': {
        'minimum_total_seconds': 8.0,
        'use_speech_gate': True, 'vad_model_path': 'models/silero_vad.onnx',
        'vad_speech_threshold': 0.5, 'vad_intra_op_threads': 1, 'vad_inter_op_threads': 1,
        'speech_head_padding_seconds': 0.2, 'speech_tail_padding_seconds': 0.32,
        'frame_ms': 20, 'minimum_pause_ms': 80,
        'minimum_segment_seconds': 2.0, 'preferred_maximum_segment_seconds': 8.0,
        'minimum_tail_seconds': 2.0, 'maximum_segments': 16,
        'maximum_chunk_words_per_second': 8.0, 'minimum_context_echo_words': 8,
        'minimum_words_per_language': 4,
        'minimum_rms': 0.003, 'noise_multiplier': 1.8,
        'maximum_relative_threshold': 0.3,
    },
}

ASR_LATENCY_EVALUATION = {
    'inputs': '.evaluation-local/mixed-asr-all-inputs-v2',
    'preview_interval_seconds': 2.0,
    'maximum_recordings': 12,
    'resource_interval_seconds': 0.25,
    'accuracy_baseline': '.evaluation-local/mixed-asr-guarded-v3',
}
