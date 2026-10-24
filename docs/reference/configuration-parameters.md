# Configuration reference

Generated from `config.py` without loading secrets. Saved, confirmed mode/output-language override their initial defaults.

- `language`: `auto`

## audio

| Parameter | Default |
| --- | --- |
| `input_sample_rate` | `16000` |
| `input_channels` | `1` |
| `frame_samples` | `512` |
| `input_device_id` | `None` |
| `output_device_id` | `None` |
| `echo_cancellation` | `True` |
| `playback_echo_guard` | `True` |
| `playback_echo_tail_ms` | `250` |
| `audio_clock_stall_timeout_ms` | `4000` |
| `audio_clock_min_progress_ms` | `100` |
| `noise_suppression` | `True` |
| `auto_gain_control` | `True` |
| `max_buffered_bytes` | `65536` |
| `max_websocket_message_bytes` | `65536` |
| `microphone_signal_timeout_ms` | `8000` |
| `microphone_signal_threshold` | `0.002` |

## vad

| Parameter | Default |
| --- | --- |
| `provider` | `silero` |
| `device` | `cpu` |
| `speech_threshold` | `0.5` |
| `minimum_rms` | `0.002` |
| `min_speech_ms` | `200` |
| `end_of_turn_silence_ms` | `700` |
| `prefix_buffer_ms` | `300` |
| `max_utterance_seconds` | `30` |
| `intra_op_threads` | `1` |
| `inter_op_threads` | `1` |

## stt

| Parameter | Default |
| --- | --- |
| `provider` | `qwen_asr` |
| `model` | `Qwen3-ASR-0.6B` |
| `runtime_python` | `.venv-asr/bin/python` |
| `dtype` | `float32` |
| `attention` | `eager` |
| `max_new_tokens` | `256` |
| `unfixed_chunk_num` | `2` |
| `rollback_tokens` | `5` |
| `num_beams` | `1` |
| `final_padding_ms` | `200` |
| `final_strategy` | `adaptive_pause_chunks` |
| `final_num_beams` | `3` |
| `final_batch_size` | `2` |
| `stream_chunk_cache` | `True` |
| `stream_cache_max_entries` | `64` |
| `final_segmentation.minimum_total_seconds` | `8.0` |
| `final_segmentation.use_speech_gate` | `True` |
| `final_segmentation.vad_model_path` | `models/silero_vad.onnx` |
| `final_segmentation.vad_speech_threshold` | `0.5` |
| `final_segmentation.vad_intra_op_threads` | `1` |
| `final_segmentation.vad_inter_op_threads` | `1` |
| `final_segmentation.speech_head_padding_seconds` | `0.2` |
| `final_segmentation.speech_tail_padding_seconds` | `0.32` |
| `final_segmentation.frame_ms` | `20` |
| `final_segmentation.minimum_pause_ms` | `80` |
| `final_segmentation.minimum_segment_seconds` | `2.0` |
| `final_segmentation.preferred_maximum_segment_seconds` | `8.0` |
| `final_segmentation.minimum_tail_seconds` | `2.0` |
| `final_segmentation.maximum_segments` | `16` |
| `final_segmentation.maximum_chunk_words_per_second` | `8.0` |
| `final_segmentation.minimum_context_echo_words` | `8` |
| `final_segmentation.minimum_words_per_language` | `4` |
| `final_segmentation.minimum_rms` | `0.003` |
| `final_segmentation.noise_multiplier` | `1.8` |
| `final_segmentation.maximum_relative_threshold` | `0.3` |
| `context` | `Transcribe English and German speech verbatim. Keep each word in its spoken language; do not translate.` |
| `startup_timeout_seconds` | `180` |
| `request_timeout_seconds` | `90` |
| `silence_peak_threshold` | `0.0001` |
| `revision` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| `device` | `cuda` |
| `device_index` | `0` |
| `compute_type` | `int8_float32` |
| `mode` | `progressive` |
| `fresh_final_decode` | `True` |
| `audit_fresh_final` | `False` |
| `partial_interval_ms` | `2000` |
| `window_seconds` | `46` |
| `beam_size` | `3` |
| `language` | `None` |
| `cpu_threads` | `2` |
| `model_repo` | `Qwen/Qwen3-ASR-0.6B` |
| `language_confidence_threshold` | `0.6` |
| `language_margin_threshold` | `0.15` |
| `minimum_language_words` | `3` |
| `minimum_supported_language_probability` | `0.2` |
| `relative_language_confidence_threshold` | `0.65` |
| `allowed_languages` | `['en', 'de']` |
| `no_speech_threshold` | `0.6` |
| `log_prob_threshold` | `-1.0` |

Qwen uses `num_beams` for previews/original whole decoding and `final_num_beams` for adaptive chunks. `final_padding_ms` conditions candidate chunks; short and monolingual fallbacks retain the original unmodified waveform. The legacy `beam_size`, `compute_type` and language-confidence fields belong to the optional faster-whisper path, not Qwen.

The original whole-decoding baseline uses `final_strategy="whole"` and `final_padding_ms=None`. The adaptive strategy adds probing cost for long recordings; its existing Silero speech check runs on CPU with a separate recurrent state.

## llm

| Parameter | Default |
| --- | --- |
| `provider` | `ollama_cloud` |
| `base_url_env` | `OLLAMA_BASE_URL` |
| `api_key_env` | `OLLAMA_API_KEY` |
| `model_env` | `OLLAMA_MODEL` |
| `stream` | `True` |
| `temperature` | `0.3` |
| `max_response_tokens` | `512` |
| `history_max_turns` | `12` |
| `request_timeout_seconds` | `60` |
| `first_token_timeout_seconds` | `15` |
| `stream_idle_timeout_seconds` | `15` |
| `system_prompt` | `You are LinguaBridge, a helpful bilingual English and German voice assistant. Use plain spoken sentences, no markdown. Be warm, accurate, concise, and preserve names and numbers. Never claim to have used tools you do not have.` |
| `think` | `False` |

## tts

| Parameter | Default |
| --- | --- |
| `provider` | `hybrid` |
| `language_segments.enabled` | `True` |
| `device` | `cuda` |
| `runtime_python` | `.venv-tts-gpu/bin/python` |
| `cuda_library_runtime` | `.venv-asr` |
| `parallel_chunks` | `2` |
| `parallel_chunks_by_language.en` | `1` |
| `parallel_chunks_by_language.de` | `2` |
| `cuda.device_id` | `0` |
| `cuda.gpu_mem_limit` | `805306368` |
| `cuda.arena_extend_strategy` | `kSameAsRequested` |
| `cuda.cudnn_conv_algo_search` | `HEURISTIC` |
| `cuda.cudnn_conv_use_max_workspace` | `0` |
| `cuda.do_copy_in_default_stream` | `1` |
| `voices.en.provider` | `kokoro` |
| `voices.en.name` | `af_heart` |
| `voices.en.model` | `models/kokoro-v1.0.onnx` |
| `voices.en.voices` | `models/voices-v1.0.bin` |
| `voices.en.speed` | `1.05` |
| `voices.de.provider` | `piper` |
| `voices.de.name` | `de_DE-thorsten-high` |
| `voices.de.model` | `models/piper/de_DE-thorsten-high.onnx` |
| `length_scale` | `1.0` |
| `noise_scale` | `0.667` |
| `noise_w_scale` | `0.8` |
| `normalize_audio` | `True` |
| `intra_op_threads` | `2` |
| `inter_op_threads` | `1` |
| `chunking.strategy` | `phrase` |
| `chunking.min_characters` | `40` |
| `chunking.max_characters` | `180` |
| `chunking.flush_timeout_ms` | `350` |

## interruptions

| Parameter | Default |
| --- | --- |
| `enabled` | `True` |
| `min_speech_ms` | `320` |
| `speech_threshold` | `0.8` |
| `minimum_rms` | `0.006` |
| `history_policy` | `completed_phrases_only` |

## workers

| Parameter | Default |
| --- | --- |
| `start_method` | `spawn` |
| `stt_processes` | `1` |
| `tts_processes` | `1` |
| `keep_models_loaded` | `True` |
| `warm_up_models` | `True` |
| `tts_queue_max_phrases` | `3` |
| `playback_queue_max_chunks` | `3` |
| `llm_token_queue_size` | `32` |
| `max_sessions` | `1` |

## logging

| Parameter | Default |
| --- | --- |
| `level` | `INFO` |
| `record_audio` | `False` |
| `log_transcripts` | `False` |
| `pipeline_trace.enabled` | `False` |
| `pipeline_trace.directory` | `logs/pipeline-trace` |

## server

| Parameter | Default |
| --- | --- |
| `host` | `127.0.0.1` |
| `port` | `8766` |
| `web_workers` | `1` |

## paths

| Parameter | Default |
| --- | --- |
| `models` | `models` |
| `web` | `web` |
| `cache` | `.cache/huggingface` |
| `stt_model` | `models/qwen3-asr-0.6b` |
| `vad_model` | `models/silero_vad.onnx` |

## modes

| Parameter | Default |
| --- | --- |
| `default` | `conversation` |
| `translation.end_of_turn_silence_ms` | `1200` |
| `translation.max_utterance_seconds` | `45` |
| `translation.output_language` | `opposite` |
| `translation.suppress_microphone_during_playback` | `False` |
| `conversation.end_of_turn_silence_ms` | `700` |
| `conversation.max_utterance_seconds` | `30` |
| `conversation.output_language` | `same` |
| `teaching.end_of_turn_silence_ms` | `1000` |
| `teaching.max_utterance_seconds` | `45` |
| `teaching.output_language` | `same` |

## memory

| Parameter | Default |
| --- | --- |
| `database` | `data/memory.sqlite3` |
| `persist_transcripts` | `True` |
| `recent_turns` | `12` |
| `translation_context_turns` | `4` |
| `summary_trigger_turns` | `20` |
| `summary_max_tokens` | `384` |
| `summary_timeout_seconds` | `4` |
| `summary_max_characters` | `3000` |
| `max_recent_characters` | `18000` |
| `max_fact_characters` | `500` |
| `max_facts` | `32` |
| `command_prefixes` | `['linguabridge', 'lingua bridge', 'bridge', 'brücke']` |
| `response_language` | `same` |

## controls

| Parameter | Default |
| --- | --- |
| `confirmation_timeout_seconds` | `45` |
| `confirm_translation_changes` | `True` |
| `confirm_response_language_changes` | `True` |
