'use strict';
(() => {
  const video = document.getElementById('demo');
  const button = document.getElementById('play');
  const label = document.getElementById('play-label');
  const status = document.getElementById('playback-status');
  const sources = [...video.querySelectorAll('source')].map(source => ({url: source.src, type: source.type}));
  const supported = sources.filter(source => video.canPlayType(source.type));
  const formats = supported.length ? supported : sources;
  let selected = 0, requested = false, timer;
  function update(text, message = '') {
    label.textContent = text; status.textContent = message;
    button.disabled = false; button.hidden = false;
  }
  function fallback() {
    clearTimeout(timer);
    if (selected + 1 >= formats.length) {
      requested = false;
      update('Try again', 'Playback could not start. Use Open MP4 below, or try another browser.');
      return false;
    }
    selected += 1; video.src = formats[selected].url; video.load();
    status.textContent = 'Trying the alternative video format…';
    if (requested) start();
    return true;
  }
  async function start() {
    requested = true; button.disabled = true; label.textContent = 'Loading demo…';
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (requested && video.paused && video.readyState < 3) fallback();
    }, 15000);
    try { await video.play(); }
    catch (error) {
      clearTimeout(timer);
      if (error.name === 'AbortError') return;
      if (error.name === 'NotSupportedError') fallback();
      else { requested = false; update('Play demo', 'Click play to allow audio in this browser.'); }
    }
  }
  if (formats.length && !video.canPlayType(sources[0].type)) video.src = formats[0].url;
  button.addEventListener('click', start);
  video.addEventListener('playing', () => {
    requested = true; clearTimeout(timer); button.disabled = false; button.hidden = true;
    status.textContent = '';
  });
  video.addEventListener('pause', () => {
    clearTimeout(timer); requested = false;
    update(video.ended ? 'Replay demo' : video.currentTime > 0 ? 'Resume demo' : 'Watch the demo');
  });
  video.addEventListener('ended', () => {requested = false; update('Replay demo');});
  video.addEventListener('waiting', () => { if (!video.paused) status.textContent = 'Buffering…'; });
  video.addEventListener('error', fallback);
})();
