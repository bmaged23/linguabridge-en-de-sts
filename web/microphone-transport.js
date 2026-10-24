/* Preserve frame order across brief browser/network scheduling bursts. */
globalThis.BridgeMicrophoneTransport = class {
  constructor(socket, {maxBufferedBytes, maxPendingFrames, drainIntervalMs, onOverflow}) {
    Object.assign(this, {socket, maxBufferedBytes, maxPendingFrames, drainIntervalMs, onOverflow});
    this.pending = []; this.timer = null; this.closed = false;
  }
  send(frame) {
    if (this.closed) return;
    this.pending.push(frame);
    this.drain();
    if (this.pending.length > this.maxPendingFrames) {
      this.close(); this.onOverflow();
    }
  }
  drain() {
    if (this.closed) return;
    while (this.pending.length && this.socket.readyState === 1 &&
           this.socket.bufferedAmount < this.maxBufferedBytes) {
      this.socket.send(this.pending.shift());
    }
    if (this.pending.length && this.timer === null) {
      this.timer = setTimeout(() => { this.timer = null; this.drain(); }, this.drainIntervalMs);
    }
  }
  close() {
    this.closed = true; this.pending = [];
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }
};
