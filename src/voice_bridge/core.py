"""Small independently testable pieces of the realtime pipeline."""
import re
from collections import deque

import numpy as np


class PhraseBuffer:
    def __init__(self, minimum=40, maximum=180):
        self.minimum, self.maximum = minimum, maximum
        self.text = ""

    def push(self, text="", flush=False):
        self.text += text
        phrases = []
        while self.text:
            match = re.search(r"[.!?;:,](?:\s|$)", self.text[self.minimum - 1:])
            end = self.minimum + match.end() - 1 if match else 0
            if not end and len(self.text) >= self.maximum:
                end = self.text.rfind(" ", 0, self.maximum + 1)
                if end <= 0:
                    end = self.maximum
            if end > self.maximum:
                end = self.text.rfind(" ", 0, self.maximum + 1) or self.maximum
            if not end:
                if not flush:
                    break
                end = len(self.text)
            phrase, self.text = self.text[:end].strip(), self.text[end:].lstrip()
            if phrase:
                phrases.append(phrase)
        return phrases


class Endpoint:
    """Accumulate 512-sample frames using externally supplied VAD scores."""
    def __init__(self, settings):
        self.settings = settings
        self.frame_ms = 32
        self.prefix = deque(maxlen=max(1, int(settings["prefix_buffer_ms"] / 32)))
        self.reset()

    def reset(self):
        self.frames = []
        self.speech_ms = 0
        self.silence_ms = 0
        self.started = False

    def feed(self, frame, probability):
        events = []
        rms=float(np.sqrt(np.mean(np.square(frame))))
        speech = (probability >= self.settings["speech_threshold"] and
                  rms >= self.settings.get("minimum_rms",0))
        if not self.frames:
            if not speech:
                self.prefix.append(frame.copy())
                return events
            self.frames = list(self.prefix)
            self.prefix.clear()
        self.frames.append(frame.copy())
        if speech:
            self.speech_ms += 32
            self.silence_ms = 0
        else:
            self.silence_ms += 32
            if not self.started:self.speech_ms=0
        if not self.started and self.speech_ms >= self.settings["min_speech_ms"]:
            self.started = True
            events.append(("start", None))
        if (self.silence_ms >= self.settings["end_of_turn_silence_ms"] or
                len(self.frames) * 32 >= self.settings["max_utterance_seconds"] * 1000):
            if self.started:
                events.append(("end", np.concatenate(self.frames)))
            self.reset()
        return events

    def snapshot(self, seconds):
        count = max(1, int(seconds * 1000 / 32))
        return np.concatenate(self.frames[-count:]) if self.frames else np.zeros(0, np.float32)
