'use strict';
globalThis.BridgePlaybackPolicy = {
  suppressMicrophone({mode, translationGuard, echoGuard, allowInterruptions, playing, queued, resumeAt, now}) {
    const protectedPlayback = (mode === 'translation' && translationGuard) || (echoGuard && !allowInterruptions);
    return Boolean(protectedPlayback && (playing || queued || now < resumeAt));
  }
};
