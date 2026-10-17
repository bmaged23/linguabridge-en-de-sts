> Historical CPU TTS deployment. For the current GPU deployment, see [GPU performance](gpu-performance.md).

# Medium-request performance snapshot

Measured on 2026-10-16 with the deployed app, warmed models, and synthetic microphone input. The environment exposes an Intel Core i7-9750H (6 cores / 12 logical CPUs), 7.66 GiB RAM, and GTX 1660 Ti (6 GiB VRAM).

## Timing

Two live requests contained 19–23 words and 7.42–8.38 seconds of input audio. Existing user history remained available. The LLM produced about 9–10 seconds of synthesized response audio. TTS-only tests used fixed 30–31-word outputs, three warmed runs per voice.

| Stage | First output / finalization | Complete generation |
| --- | --- | --- |
| Qwen3-ASR-0.6B GPU FP32 | Final transcript 0.56–0.64 s after VAD endpoint | Progressive work occurs while speaking |
| Ollama Cloud Gemma4 31B | First token 0.44–0.54 s after Thinking | Last token 0.54–0.66 s after Thinking |
| English Kokoro CPU | First chunk 1.55–1.72 s | 4.71–4.81 s for 8.53–8.56 s of audio |
| German Piper CPU | First chunk 1.23–1.50 s | 3.75–4.06 s for 9.82–10.05 s of audio |

The live pipeline returned its first audio 3.64–3.79 seconds after the source WAV ended. Recorded WAVs contain some trailing quiet, so this is not an exact physical end-of-speech measurement. Measured VAD wait from WAV end was 0.52–0.55 seconds; the conversation endpoint setting is 700 ms. Stages overlap: audio playback can begin before the complete response is generated. The test decoded and immediately acknowledged playback packets rather than playing them through a speaker; it does not measure browser audio-device latency or sustained real-playback backpressure.

## Resources

CPU percentages below divide process CPU usage by all 12 logical CPUs. A process value of 200% in Linux tools means two logical CPUs, or about 16.7% in this table. Sampling was approximately every 250–400 ms; brief peaks can be missed.

| Workload | Average CPU capacity | Sampled CPU peak | GPU utilization |
| --- | --- | --- | --- |
| Whole app during English/German capture and STT | 4.1–4.7% | 9.0% | 15–16% average; 52% sampled peak |
| Whole app while LLM and TTS overlap | 14.6% | 17.2% | ASR resident; occasional preceding decode activity |
| Isolated English TTS worker | 16.5% | 16.9% | 0% inference on GPU |
| Isolated German TTS worker | 16.7% | 17.1% | 0% inference on GPU |

- Actual whole-device VRAM: 3.67 GiB loaded idle; 3.79 GiB sampled peak. Qwen is the only GPU inference model; CUDA overhead makes this larger than the earlier ~3.1 GiB PyTorch allocation measurement. VRAM stays allocated while the LLM and CPU TTS run.
- Warm RAM footprint by proportional set size (PSS): Qwen worker 1,691 MiB, shared bilingual TTS worker 814 MiB, app server 94 MiB, tracker 7 MiB: about 2.54 GiB total. The two TTS RAM figures must not be added as separate models/processes; both voices share one worker.
- Normal live-test whole-system RAM usage peaked at 41.7%. The separate TTS benchmark temporarily started a second TTS process, which is not part of the normal deployed footprint.
- Ollama Cloud server CPU, RAM and VRAM are unavailable to this local profiler. No LLM weights are loaded locally. Cloud/network latency and reply length can change these timings substantially.

[Raw measurements](data/runtime-profile.json). Reproduce with `.venv/bin/python -m scripts.evaluation.profile_voice_pipeline`; it removes only its own synthetic conversation rows and leaves existing user settings and history intact. Main-environment exact dependencies include psutil. The app stays running after the benchmark.
