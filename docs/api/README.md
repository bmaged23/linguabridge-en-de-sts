# LinguaBridge application API

LinguaBridge provides a FastAPI backend with HTTP endpoints for application state and controls, plus a WebSocket endpoint for live speech-to-speech sessions. These are the endpoints used by the browser interface; they are implemented in [server.py](../../src/voice_bridge/server.py).

## Start the server

Start the application using the [Docker pull/run instructions](../../deployment/README.md#pull-and-run-without-cloning) or the [native installation steps](../../README.md#installation--clone-to-first-conversation). The API becomes available after the GPU speech models finish loading and warming up.

- HTTP base URL: `http://localhost:8766`
- Streaming URL: `ws://localhost:8766/ws`
- Browser application: `http://localhost:8766/`
- Interactive HTTP documentation: `http://localhost:8766/docs`
- OpenAPI schema: `http://localhost:8766/openapi.json`

The API does not start Docker or load the server on demand. Start the server first, then send requests. Interactive documentation describes HTTP routes; the WebSocket protocol is documented below. Control request bodies are handled as JSON commands, so their fields are listed here rather than fully described by generated OpenAPI models.

## HTTP endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness, configured cloud model, speech model and CPU/GPU settings |
| `GET` | `/client-config` | Audio framing, microphone buffering and interruption settings for clients |
| `GET` | `/state` | Current mode, reply-language preference, stored notes, summary and pending confirmation |
| `GET` | `/diagnostics` | Current pipeline diagnostics and whether a voice session is active |
| `POST` | `/control` | Request/confirm mode or language changes and manage memory when no voice session is active |

For example, after startup:

```bash
curl http://localhost:8766/health
curl http://localhost:8766/state
curl http://localhost:8766/diagnostics
```

`/health` reports local model readiness; it is not a live authentication or quota check against Ollama Cloud.

### Control commands

Send JSON with a `type` and the corresponding fields:

| `type` | Additional fields | Effect |
| --- | --- | --- |
| `set_mode` | `mode`: `conversation`, `translation` or `teaching` | Request a mode change; translation requires confirmation by default |
| `set_language` | `language`: `same`, `en` or `de` | Request the reply-language preference; confirmation is required by default |
| `confirm_change` | `id`: pending confirmation ID | Apply the matching unexpired change |
| `cancel_change` | None | Cancel a pending change |
| `remember` | `text`: note text | Save an explicit note |
| `forget` | `id`: numeric note ID | Remove the selected note |
| `clear_history` | None | Erase the current mode's conversation history |
| `clear_memory` | None | Erase saved notes and histories |

Example: request translation while no voice session is connected:

```bash
curl http://localhost:8766/control \
  -H 'Origin: http://localhost:8766' \
  -H 'Content-Type: application/json' \
  --data '{"type":"set_mode","mode":"translation"}'
```

Read `pending_confirmation.id` from the response, then explicitly confirm:

```bash
curl http://localhost:8766/control \
  -H 'Origin: http://localhost:8766' \
  -H 'Content-Type: application/json' \
  --data '{"type":"confirm_change","id":"REPLACE_WITH_PENDING_ID"}'
```

Successful requests return the updated application state. Invalid commands or expired confirmations return HTTP `400`; a missing/mismatched Origin returns `403`; using HTTP controls during an active voice session returns `409`. During a voice session, send the same control commands over its WebSocket instead. Configuration determines which changes require confirmation.

## Streaming speech-to-speech: `WS /ws`

The WebSocket carries microphone input, partial/final recognition, streamed LLM text, synthesized audio and playback acknowledgements. It runs the complete **microphone → VAD → STT → cloud LLM → TTS → playback** flow.

1. Connect with a matching Origin, for example `Origin: http://localhost:8766` when using that host. Wait for the server's `ready` and `mode` JSON messages.
2. Send mono audio at **16,000 Hz**, as binary messages of **512 little-endian float32 samples** each: **2,048 bytes / 32 ms per frame**. Samples should be finite and normalized to approximately `[-1, 1]`. Send frames in real time, including silence, so VAD can detect turn endings. WAV headers, MP3 files and base64 input are not this protocol.
3. Receive JSON events. The server chooses when a turn ends, transcribes it and streams the response. Keep receiving while sending microphone frames so interruption remains possible.
4. Play `audio` chunks in order and acknowledge each chunk. Stop queued playback when an `interrupt` event arrives, and discard audio whose `turn` does not match the current turn.
5. Send `{"type":"stop"}` or close the socket to end the session.

### Server events

| `type` | Important fields | Meaning |
| --- | --- | --- |
| `ready` | `model`, `audio` | Session initialized; audio configuration supplied |
| `mode` | State fields, `pending_confirmation` | Current controls and memory state |
| `partial` | `text`, `language` | Provisional STT text; later recognition can revise it |
| `final` | `text`, `language`, `certain` | Final recognized input |
| `route` | `source`, `target`, `mode` | Selected response languages and mode |
| `token` | `text`, `language`, `turn` | Streamed reply text with language information |
| `audio` | `id`, `turn`, `sample_rate`, `pcm`, `language` | Synthesized audio chunk |
| `interrupt` | `turn` | A new turn invalidates previous reply playback |
| `status` | `text`, optional `turn` | Pipeline status such as transcribing, thinking or listening |
| `latency` | `seconds` | Server-measured time to first output audio |
| `notice`, `error` | `text` | User-facing information or a pipeline failure |

The output `pcm` field is **base64-encoded mono signed 16-bit little-endian PCM**. Use each chunk's `sample_rate`; input float32 framing and output PCM16 are different formats.

After a chunk finishes playing, send its exact ID:

```json
{"type":"played","id":"3:1","complete":true}
```

If playback is interrupted, acknowledge the interrupted playing chunk with `complete:false`. IDs above are examples; use those received from the server. Playback acknowledgements release the bounded output queue and record which assistant phrases were heard. A client that never acknowledges audio can stall synthesis after the queue fills. See the working client in [client.js](../../web/client.js), [microphone capture](../../web/capture.js) and [microphone transport](../../web/microphone-transport.js).

## Current scope and verification

- One active voice session is supported. A second connection is rejected with an error and WebSocket close code `1008`.
- WebSocket connections and HTTP controls require an Origin matching the request host, including its port. `localhost` and `127.0.0.1` must be used consistently. A WebSocket with a missing/mismatched Origin is rejected before acceptance.
- This is a local application API. It has no user-account or API-token authentication layer; Origin checks do not provide authentication for arbitrary remote clients. The documented Docker commands bind to localhost.
- The Ollama key stays in the server environment. Clients do not provide it in microphone packets. The GPU speech models and cloud account allowance are required for the complete flow.
- There are no separate REST file-upload transcription, standalone TTS or text-chat endpoints. Use `/ws` for the implemented full streaming pipeline.

The qualified GPU image completed **17/17 browser conversation turns**, exercising these endpoints through the real client, alongside **94 offline regressions** and **52 STT recordings**. The published image also passed anonymous pull, GPU warmup, health and UI startup. These are application/integration checks, not an exhaustive API load or security test. See the [verified release report](../../evaluation/results/gpu-container-release-2026-10-24/README.md).
