'use strict';
const el = id => document.getElementById(id);
const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
let ws, context, stream, microphone, capture, inputMeter, outputMeter, outputGain;
let microphoneTransport;
let queue = [], playing = null, turn = 0, active = false, muted = false, generation = 0;
let userMessage = null, assistantMessage = null, responsePending = false, totalTurns = 0;
let timer = null, sessionStart = 0, lastLatency = null;
let playbackEchoGuard=true, playbackEchoTailMs=250, resumeMicrophoneAt=0;
let audioClockAt=0, audioClockValue=0, audioClockTimeout=4000, audioClockMinimum=.1;
let microphoneSignalSeen=false, microphoneSignalThreshold=.002, microphoneSignalTimeout=8000;
let bridgeMode = "conversation", suppressPlaybackInput = true, savedState = null, lastOutputLanguage = null;

function state(mode, detail) {
  const guardedPlayback = (bridgeMode==='translation' && suppressPlaybackInput) || (playbackEchoGuard && !el('allow-interruptions').checked);
  const effective = muted && mode === 'listening' ? 'muted' : mode;
  document.body.dataset.state = effective;
  const labels = {
    idle: ['Ready when you are.', 'One click. A more natural way to talk.', 'Microphone off', 'STANDBY'],
    connecting: ['Getting comfortable…', 'Connecting your microphone and assistant.', 'Connecting', 'CONNECTING'],
    listening: ['Ready when you are.', 'The microphone is ready. Speak whenever you like.', 'Waiting for speech', 'LIVE'],
    paused: ['Audio is paused.', 'Click Resume audio to continue the microphone and reply.', 'Audio paused', 'PAUSED'],
    hearing: ['I’m listening.', 'Finish your thought; I’ll reply after your pause.', 'Hearing your speech', 'HEARING'],
    transcribing: ['I heard you.', 'Turning your speech into text.', 'Recognizing your words', 'TRANSCRIBING'],
    thinking: ['A thought is taking shape.', 'Your assistant is putting a reply together.', 'Thinking', 'THINKING'],
    speaking: ['Let’s keep talking.', guardedPlayback ? 'Microphone resumes when the reply finishes.' : 'Jump in whenever something comes to mind.', guardedPlayback ? 'Speaking · echo protection on' : 'Speaking · feel free to interrupt', 'SPEAKING'],
    muted: ['Take your time.', 'You can still listen. Unmute whenever you’re ready.', 'Microphone muted', 'MUTED']
  };
  const [title, description, status, badge] = labels[effective] || labels.idle;
  el('stage-title').textContent = title;
  el('stage-description').textContent = bridgeMode === 'translation' && effective === 'listening' ? 'Speak English or German. Take your time; the microphone stays ready.' : description;
  if (bridgeMode === 'translation' && effective === 'speaking' && suppressPlaybackInput) el('stage-description').textContent = 'Listening resumes after the translation. This prevents speaker echo.';
  el('status').textContent = detail || (bridgeMode==='translation' && suppressPlaybackInput && effective==='speaking' ? 'Speaking translation · microphone resumes afterwards' : status);
  el('live-pill').querySelector('span').textContent = badge;
}
function connection(online, text) {
  el('connection').className = `connection ${online ? 'online' : 'offline'}`;
  el('connection-text').textContent = text;
}
function showError(text = '') {
  el('error').textContent = text;
  el('error').hidden = !text;
}
function send(packet, socket = ws) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(packet));
}
function scrollTranscript() {
  const panel = el('conversation-scroll');
  // Preserve the reader’s position when they have scrolled up.
  if (panel.scrollHeight - panel.scrollTop - panel.clientHeight < 160) panel.scrollTop = panel.scrollHeight;
}
function message(role) {
  el('empty-state').hidden = true;
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const meta = document.createElement('div'); meta.className = 'message-meta';
  const avatar = document.createElement('span'); avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? 'EN/DE' : 'B';
  const name = document.createElement('span'); name.textContent = role === 'user' ? 'Speaker' : `LinguaBridge${lastOutputLanguage ? ' · '+lastOutputLanguage.toUpperCase() : ''}`;
  const timestamp = document.createElement('time'); timestamp.className = 'message-time';
  timestamp.dateTime = new Date().toISOString();
  timestamp.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  const body = document.createElement('div'); body.className = 'message-body';
  meta.append(avatar, name, timestamp); article.append(meta, body); el('messages').append(article);
  scrollTranscript();
  return {article, body, name, languages: new Set()};
}
function haltPlayback() {
  resumeMicrophoneAt=performance.now()+playbackEchoTailMs;
  queue = [];
  if (playing) {
    const old = playing; playing = null; old.source.onended = null;
    try { old.source.stop(); } catch { /* already stopped */ }
    send({type: 'played', id: old.id, complete: false});
  }
}
function playNext() {
  if (!active) return;
  if (context?.state !== 'running') { el('resume-audio').hidden=false;state('paused');return; }
  if (playing) return;
  while (queue.length && queue[0].turn !== turn) queue.shift();
  if (!queue.length) {
    state(responsePending ? 'thinking' : 'listening');
    return;
  }
  const chunk = queue.shift();
  const bytes = Uint8Array.from(atob(chunk.pcm), c => c.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const buffer = context.createBuffer(1, bytes.length / 2, chunk.sample_rate);
  const audio = buffer.getChannelData(0);
  for (let i = 0; i < audio.length; i++) audio[i] = view.getInt16(i * 2, true) / 32768;
  const source = context.createBufferSource(); source.buffer = buffer; source.connect(outputGain);
  playing = {id: chunk.id, source}; state('speaking');
  const thisGeneration = generation;
  source.onended = () => {
    if (thisGeneration !== generation) return;
    resumeMicrophoneAt=performance.now()+playbackEchoTailMs;
    send({type: 'played', id: chunk.id, complete: true});
    playing = null; playNext();
  };
  source.start();
}
async function listMicrophones() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const select = el('microphone-select'), selected = select.value;
  select.replaceChildren(new Option('System default', ''));
  let index = 0;
  for (const device of devices.filter(item => item.kind === 'audioinput')) {
    if (device.deviceId && device.deviceId !== 'default') {
      select.add(new Option(device.label || `Microphone ${++index}`, device.deviceId));
    }
  }
  if ([...select.options].some(option => option.value === selected)) select.value = selected;
}
async function stopSession(detail = 'Microphone off') {
  generation++;
  active = false; responsePending = false;
  microphoneTransport?.close(); microphoneTransport = null;
  el('resume-audio').hidden=true;
  haltPlayback();
  clearInterval(timer); timer = null;
  const oldStream = stream, oldContext = context, oldSocket = ws;
  stream = context = ws = null;
  oldStream?.getTracks().forEach(track => track.stop());
  capture?.disconnect(); microphone?.disconnect();
  capture = microphone = inputMeter = outputMeter = outputGain = null;
  if (oldSocket && oldSocket.readyState === WebSocket.OPEN) send({type: 'stop'}, oldSocket);
  if (oldSocket && oldSocket.readyState < WebSocket.CLOSING) oldSocket.close();
  if (oldContext && oldContext.state !== 'closed') {
    try { await oldContext.close(); } catch { /* already closing */ }
  }
  muted = false;
  el('mute').disabled = true; el('mute').setAttribute('aria-pressed', 'false');
  el('mute').setAttribute('aria-label', 'Mute microphone');
  el('start').disabled = false; el('start').classList.remove('is-active');
  el('start-label').textContent = 'Start conversation';
  el('start-icon').querySelector('use').setAttribute('href', '#i-mic');
  el('control-hint').textContent = 'Your microphone turns on only when you start.';
  el('transcript-indicator').lastChild.textContent = 'Session stopped';
  el('partial').textContent = '';
  connection(true, 'Server ready');
  state('idle', detail);
}
function onPacket(packet) {
  switch (packet.type) {
    case 'notice':
      showError(packet.text);
      el('stage-description').textContent=packet.text;
      break;
    case 'mode':
      applyBridgeState(packet);
      el('allow-interruptions').disabled=bridgeMode==='translation' && suppressPlaybackInput;
      break;
    case 'route':
      lastOutputLanguage=packet.target;
      el('mode-summary').textContent = `${packet.mode} · ${packet.source.toUpperCase()} → ${packet.target.toUpperCase()} · automatic language routing`;
      break;
    case 'interrupt':
      showError();
      if ((playing || responsePending) && assistantMessage) assistantMessage.article.classList.add('interrupted');
      haltPlayback(); turn = packet.turn; userMessage = assistantMessage = null;
      responsePending = false; el('partial').textContent = ''; state('listening');
      break;
    case 'partial':
      el('empty-state').hidden = true;
      el('partial').textContent = packet.text; scrollTranscript(); break;
    case 'final':
      el('partial').textContent = '';
      if (packet.text) {
        userMessage = message('user'); userMessage.body.textContent = packet.text;
        if (packet.language) { userMessage.article.dataset.language = packet.language;userMessage.article.querySelector('.message-meta span:nth-child(2)').textContent='Speaker · '+packet.language.toUpperCase(); }
        totalTurns++; el('turn-count').textContent = totalTurns; scrollTranscript();
      }
      break;
    case 'token':
      if (packet.turn !== turn) break;
      if(packet.language && /[\p{L}\p{N}]/u.test(packet.text))lastOutputLanguage=packet.language;
      if (!assistantMessage) assistantMessage = message('assistant');
      if (packet.language && /[\p{L}\p{N}]/u.test(packet.text)) {
        assistantMessage.languages.add(packet.language.toUpperCase());
        assistantMessage.name.textContent = 'LinguaBridge · '+[...assistantMessage.languages].join(' / ');
      }
      assistantMessage.body.append(document.createTextNode(packet.text)); scrollTranscript(); break;
    case 'audio':
      if (packet.turn === turn) { queue.push(packet); playNext(); } break;
    case 'status':
      if (packet.turn != null && packet.turn !== turn) break;
      if (packet.text === 'Transcribing') {responsePending=true;if(!playing)state('transcribing');}
      else if (packet.text === 'Thinking') { responsePending = true; if (!playing) state('thinking'); }
      else if (packet.text === 'Listening to you') { if (!playing) state('hearing'); }
      else if (packet.text === 'Listening') { responsePending = false; if (!playing) state('listening'); }
      else if (!playing) state('listening');
      break;
    case 'latency':
      lastLatency = packet.seconds;
      el('latency').replaceChildren(document.createTextNode(packet.seconds.toFixed(2)));
      const unit = document.createElement('span'); unit.textContent = ' s'; el('latency').append(unit);
      break;
    case 'error':
      showError(packet.text); responsePending = false;
      if (active && !playing) state('listening', packet.text);
      break;
  }
}
async function startSession() {
  const token = ++generation;
  el('start').disabled = true; showError(); state('connecting');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Open this page on localhost or HTTPS to use your microphone.');
    // Resume audio inside the click gesture before waiting for network requests.
    context = new AudioContext();
    context.onstatechange=()=>{
      if(!active || !context)return;
      const paused=context.state!=='running';el('resume-audio').hidden=!paused;
      if(paused)state('paused');
      else {audioClockAt=performance.now();audioClockValue=context.currentTime;if(playing)state('speaking');else playNext();}
    };
    await context.resume();
    const response = await fetch('/client-config');
    if (!response.ok) throw new Error('The voice server is unavailable. Start it, then try again.');
    const audioConfig = await response.json();
    audioClockTimeout=audioConfig.audio_clock_stall_timeout_ms;audioClockMinimum=audioConfig.audio_clock_min_progress_ms/1000;
    playbackEchoGuard=audioConfig.playback_echo_guard;playbackEchoTailMs=audioConfig.playback_echo_tail_ms;
    resumeMicrophoneAt=0;
    if (!el('allow-interruptions').dataset.touched) el('allow-interruptions').checked=audioConfig.interruptions_enabled ?? true;
    microphoneSignalThreshold=audioConfig.microphone_signal_threshold??.002;
    microphoneSignalTimeout=audioConfig.microphone_signal_timeout_ms??8000;
    microphoneSignalSeen=false;
    if (token !== generation) return;
    outputGain = context.createGain(); outputGain.gain.value = Number(el('volume').value) / 100;
    outputMeter = context.createAnalyser(); outputMeter.fftSize = 512;
    outputGain.connect(outputMeter); outputMeter.connect(context.destination);
    if (audioConfig.output_device_id && context.setSinkId) await context.setSinkId(audioConfig.output_device_id);
    const constraints = {channelCount: audioConfig.input_channels, echoCancellation: audioConfig.echo_cancellation,
      noiseSuppression: audioConfig.noise_suppression, autoGainControl: audioConfig.auto_gain_control};
    const device = el('microphone-select').value || audioConfig.input_device_id;
    if (device) constraints.deviceId = {exact: device};
    const selectedStream = await navigator.mediaDevices.getUserMedia({audio: constraints});
    if (token !== generation) { selectedStream.getTracks().forEach(track => track.stop()); return; }
    stream = selectedStream;
    await listMicrophones();
    await context.audioWorklet.addModule('/static/capture.js');
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws = socket;
    socket.onmessage = event => {
      if (token !== generation) return;
      const packet = JSON.parse(event.data);
      if (packet.type === 'ready') {
        active = true; turn = 0; userMessage = assistantMessage = null;
        el('messages').replaceChildren(); el('empty-state').hidden = false;
        totalTurns = 0; lastLatency = null;
        el('turn-count').textContent = '0'; el('latency').textContent = '—';
        sessionStart = performance.now();audioClockAt=sessionStart;audioClockValue=context.currentTime;
        clearInterval(timer); timer = setInterval(() => { el('duration').textContent = clock(Math.floor((performance.now() - sessionStart) / 1000));checkAudioClock(); }, 1000);
        el('duration').textContent = '00:00'; el('mute').disabled = false;
        el('start').disabled = false; el('start').classList.add('is-active');
        el('start-label').textContent = 'End conversation';
        el('start-icon').querySelector('use').setAttribute('href', '#i-stop');
        refreshInterruptionCopy();
        el('transcript-indicator').lastChild.textContent = 'Live';
        el('model-name').textContent = packet.model;
        connection(true, 'Connected'); state('listening');refreshInterruptionCopy();
        microphone = context.createMediaStreamSource(stream);
        inputMeter = context.createAnalyser(); inputMeter.fftSize = 512;
        capture = new AudioWorkletNode(context, 'microphone-capture');
        microphoneTransport = new BridgeMicrophoneTransport(socket, {
          maxBufferedBytes: audioConfig.max_buffered_bytes,
          maxPendingFrames: audioConfig.max_pending_microphone_frames,
          drainIntervalMs: audioConfig.microphone_drain_interval_ms,
          onOverflow: () => { showError('The audio connection fell behind. Please start a new session.'); stopSession(); }
        });
        capture.addEventListener('processorerror',()=>{showError('Microphone processing stopped. Start the conversation again.');stopSession('Microphone stopped');});
        capture.port.onmessage = e => {
          if (!active || socket.readyState !== WebSocket.OPEN || token !== generation) return;
          if (BridgePlaybackPolicy.suppressMicrophone({mode:bridgeMode,translationGuard:suppressPlaybackInput,echoGuard:playbackEchoGuard,allowInterruptions:el('allow-interruptions').checked,playing:Boolean(playing),queued:queue.length,resumeAt:resumeMicrophoneAt,now:performance.now()})) microphoneTransport.send(new Float32Array(512).buffer);
          else microphoneTransport.send(e.data);
        };
        microphone.connect(inputMeter); inputMeter.connect(capture); capture.connect(context.destination);
      } else onPacket(packet);
    };
    socket.onerror = () => { if (token === generation) showError('Unable to connect to the voice server. Check that it is running.'); };
    socket.onclose = () => {
      if (token !== generation) return;
      if (active) showError('The connection closed. Start again to continue.');
      stopSession('Session ended'); connection(false, 'Disconnected');
    };
  } catch (error) {
    if (token !== generation) return;
    const descriptions = {NotAllowedError: 'Microphone access was denied. Allow microphone access in your browser, then try again.',
      NotFoundError: 'No microphone was found. Connect one and try again.',
      NotReadableError: 'Your microphone is busy or unavailable. Close other recording apps and try again.',
      OverconstrainedError: 'The selected microphone is unavailable. Choose System default in audio settings.'};
    showError(descriptions[error.name] || error.message);
    await stopSession();
  }
}
el('start').onclick = () => active ? stopSession('Session ended') : startSession();
el('mute').onclick = () => {
  if (!active) return;
  muted = !muted; stream.getAudioTracks().forEach(track => { track.enabled = !muted; });
  el('mute').setAttribute('aria-pressed', String(muted));
  el('mute').setAttribute('aria-label', muted ? 'Unmute microphone' : 'Mute microphone');
  if (!playing && !responsePending) state('listening');
};
el('audio-settings').onclick = async () => {
  const open = el('settings-panel').hidden;
  el('settings-panel').hidden = !open; el('audio-settings').setAttribute('aria-expanded', String(open));
  try { if (open) await listMicrophones(); } catch { /* Permission may not be granted yet. */ }
};
el('volume').oninput = event => {
  const value = Number(event.target.value); el('volume-label').textContent = `${value}%`;
  if (outputGain) outputGain.gain.value = value / 100;
};
el('new-session').onclick = async () => {
  await bridgeControl({type:'clear_history'});
  el('messages').replaceChildren();el('empty-state').hidden=false;
  userMessage=assistantMessage=null;showError();
};
window.addEventListener('pagehide', () => {
  active = false; stream?.getTracks().forEach(track => track.stop()); ws?.close();
});

// Display actual microphone/playback energy. Decorative idle bars stay quiet.
const canvas = el('waveform'), pen = canvas.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const samples = new Float32Array(512);
let energy = 0;
function drawWave(time) {
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  }
  pen.setTransform(dpr, 0, 0, dpr, 0, 0); pen.clearRect(0, 0, rect.width, rect.height);
  const meter = playing ? outputMeter : muted ? null : inputMeter;
  let rms = 0;
  if (meter) { meter.getFloatTimeDomainData(samples); for (const value of samples) rms += value * value; rms = Math.sqrt(rms / samples.length); }
  if(active && !muted && !playing && rms>microphoneSignalThreshold)microphoneSignalSeen=true;
  if(active && !muted && !playing && !responsePending && !microphoneSignalSeen && performance.now()-sessionStart>microphoneSignalTimeout){
    el('stage-description').textContent='No microphone signal detected yet. Check your microphone in Audio settings.';
  }
  energy = energy * .72 + Math.min(rms * 5, 1) * .28;
  const count = 49, gap = 5, width = 2, start = (rect.width - (count - 1) * gap - width) / 2;
  for (let i = 0; i < count; i++) {
    const taper = Math.sin(Math.PI * (i + 1) / (count + 1));
    const phase = reducedMotion ? i : i * .9 + time / 130;
    const height = 3 + energy * 32 * taper * (.35 + .65 * Math.abs(Math.sin(phase)));
    pen.fillStyle = active ? `rgba(90,139,88,${.28 + taper * .5})` : `rgba(145,166,125,${.18 + taper * .28})`;
    pen.beginPath(); pen.roundRect(start + i * gap, (rect.height - height) / 2, width, height, 2); pen.fill();
  }
  requestAnimationFrame(drawWave);
}
requestAnimationFrame(drawWave);
async function checkServer() {
  try {
    const response = await fetch('/health'); if (!response.ok) throw new Error('Unavailable');
    const health = await response.json();
    if (!active) connection(true, 'Server ready'); el('model-name').textContent = health.model;
  } catch { if (!active) connection(false, 'Server offline'); }
}
checkServer();

function applyBridgeState(data) {
  const changedMode=bridgeMode!==data.mode;
  savedState=data;bridgeMode=data.mode;
  el('confirmation-card').hidden=!data.pending_confirmation;
  if(data.pending_confirmation)el('confirmation-message').textContent=data.pending_confirmation.message;
  if(changedMode || !el('messages').children.length){
    el('messages').replaceChildren();userMessage=assistantMessage=null;
    for(const row of data.recent||[]){
      const user=message('user');user.body.textContent=row.user;
      user.article.querySelector('.message-meta span:nth-child(2)').textContent='Speaker · '+row.source.toUpperCase();
      if(row.heard){lastOutputLanguage=row.target;const answer=message('assistant');answer.body.textContent=row.heard;}
    }
    if(!el('messages').children.length)el('empty-state').hidden=false;
  }
  if(data.suppress_playback_input!=null)suppressPlaybackInput=data.suppress_playback_input;
  document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===data.mode)));
  el('reply-language').value=data.response_language;
  el('reply-language').disabled=data.mode==='translation';
  const guarded=data.mode==='translation' && suppressPlaybackInput;
  refreshInterruptionCopy();
  el('mode-summary').textContent=data.mode==='translation'
    ? `English ⇄ German · automatic direction · ${data.silence_ms||1200} ms pause allowance · waits indefinitely for the next speaker`
    : `${data.mode==='teaching'?'Teaching':'Conversation'} · ${data.silence_ms||700} ms pause allowance · persistent context`;
  el('memory-count').textContent=`${data.facts.length} saved notes`;
  el('saved-summary').textContent=data.summary||'No summary yet. Older conversations are summarized as history grows.';
  el('memory-list').replaceChildren();
  for(const fact of data.facts){
    const item=document.createElement('li'),text=document.createElement('span'),remove=document.createElement('button');
    text.textContent=fact.text;remove.textContent='Forget';remove.setAttribute('aria-label',`Forget: ${fact.text}`);
    remove.onclick=()=>bridgeControl({type:'forget',id:fact.id});item.append(text,remove);el('memory-list').append(item);
  }
}
async function bridgeControl(packet){
  showError();
  if(active){send(packet);return;}
  try{
    const response=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(packet)});
    const body=await response.json();
    if(!response.ok)throw new Error(body.detail||'Could not update memory or mode. Check the server.');
    applyBridgeState(body);
  }catch(error){showError(error.message);}
}
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>bridgeControl({type:'set_mode',mode:button.dataset.mode}));
el('reply-language').onchange=event=>bridgeControl({type:'set_language',language:event.target.value});
el('memory-form').onsubmit=event=>{event.preventDefault();const text=el('memory-note').value.trim();if(text){bridgeControl({type:'remember',text});el('memory-note').value='';}};
el('clear-memory').onclick=()=>{if(confirm('Erase all saved notes, summaries, and text history? The active mode will remain.'))bridgeControl({type:'clear_memory'});};
fetch('/state').then(response=>{if(!response.ok)throw new Error();return response.json();}).then(applyBridgeState).catch(()=>{});

el('confirm-change').onclick=()=>{if(savedState?.pending_confirmation)bridgeControl({type:'confirm_change',id:savedState.pending_confirmation.id});};
el('cancel-change').onclick=()=>bridgeControl({type:'cancel_change'});

function refreshInterruptionCopy(){
 const translation=bridgeMode==='translation' && suppressPlaybackInput;
 const enabled=!translation && (!playbackEchoGuard || el('allow-interruptions').checked);
 document.querySelector('.transcript-footer').lastChild.textContent=translation ? 'Listening resumes after each translation. The next person can take their time.' : enabled ? 'You can interrupt a reply by speaking.' : 'Enable voice interruptions in audio settings when using headphones.';
 if(active)el('control-hint').textContent=translation ? 'Automatic English ⇄ German. Mode stays active until you change it.' : enabled ? 'Speak naturally. You can interrupt at any time.' : 'Speak naturally. Your microphone resumes after each reply.';
}
el('allow-interruptions').onchange=()=>{el('allow-interruptions').dataset.touched='true';refreshInterruptionCopy();};

function checkAudioClock(){
  if(!active || !context)return;
  const now=performance.now();
  if(context.state!=='running'){audioClockAt=now;audioClockValue=context.currentTime;return;}
  if(now-audioClockAt<audioClockTimeout)return;
  if(context.currentTime-audioClockValue<audioClockMinimum){
    showError('The audio device stopped advancing. Start the conversation again.');
    stopSession('Audio device stopped');return;
  }
  audioClockAt=now;audioClockValue=context.currentTime;
}
el('resume-audio').onclick=async()=>{
  if(!active || !context)return;
  try{await context.resume();if(context.state==='running'){el('resume-audio').hidden=true;audioClockAt=performance.now();audioClockValue=context.currentTime;if(playing)state('speaking');else playNext();}}
  catch{showError('Audio could not resume. End the conversation and start again.');}
};
