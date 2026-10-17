# English/German recognizer investigation

Research date: 2026-10-16. Focus: preserving both languages within one utterance,
including English → German → English and German → English → German.

| Candidate | Why consider it | Important limitation |
| --- | --- | --- |
| Qwen3-ASR-0.6B | Smaller checkpoint, English/German support, automatic language detection, Apache-2.0 | Must test mixed-language preservation and latency locally; ordinary multilingual benchmark scores do not establish code-switch accuracy |
| Qwen3-ASR-1.7B | Larger accuracy-oriented checkpoint with the same supported languages and license | Higher memory and decoding cost; evaluate after the small model |
| NVIDIA Parakeet TDT 0.6B v3 | English/German among 25 supported languages; CC-BY-4.0 | NVIDIA explicitly says it is not fully tuned for code-switching |

Primary sources:

- [Qwen official implementation and installation](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen small checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen larger checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen technical report](https://arxiv.org/abs/2601.21337)
- [NVIDIA Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NVIDIA maintainer's code-switching limitation](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/1)

The official Qwen package documents streaming through vLLM. This local experiment
tests completed utterances and real-time progressive decoding through Transformers.
The progressive prototype reuses earlier text with a five-token rollback while
reprocessing cumulative audio; it is not native vLLM streaming. Both FP16 and FP32
were tested with eager attention. FP32 worked better for short partials on this GTX
1660 Ti. See [measured results and limitations](asr-evaluation.md).

The evaluation environment `.venv-asr/` and model cache `.cache/asr-evaluation/`
are ignored and separate from the application's `.venv`. The comparison script
`scripts/evaluation/evaluate_qwen_asr.py` resolves checkpoint commits and records them with
dependency versions, per-example latency, peak PyTorch allocation, and normalized
WER/CER in `logs/codeswitch-check/`. It reuses the existing WAVs without regenerating
or segmenting them according to the known reference languages. App defaults and
user memory are not changed by this experiment.

A relevant real-speech evaluation set is
[DECM](https://publikationen.bibliothek.kit.edu/1000172045/153261717), a German-English
code-switching benchmark with German as the matrix language. Its paper reports
3.38 hours of spontaneous speech. It is a possible follow-up to synthetic tests;
no DECM evaluation has been run here.
