"""Streaming language switches for bilingual speech; no second language model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechPhrase:
    text: str
    language: str


class LanguageSegments:
    """Opening markers select a voice. Legacy closing tags are optional.

    Repeated/mismatched closing tags, punctuation between tags, and an unfinished
    final marker must not cancel an otherwise valid reply. Unmarked prefixes use
    the requested main language. Unknown square brackets remain ordinary text.
    """
    def __init__(self,default_language='en'):
        self.pending=''
        self.language=default_language
        self.has_text=False
        self.repairs=0

    def push(self,text='',final=False):
        self.pending+=text
        events=[]
        tags={'[en]':'en','[de]':'de','[/en]':None,'[/de]':None}
        while self.pending:
            position=self.pending.find('[')
            if position!=0:
                end=position if position>0 else len(self.pending)
                body=self.pending[:end];self.pending=self.pending[end:]
                events.append((self.language,body,False))
                self.has_text=self.has_text or bool(body.strip())
                continue
            tag=next((tag for tag in tags if self.pending.startswith(tag)),None)
            if tag:
                selected=tags[tag]
                if selected is None:
                    if tag!=f'[/{self.language}]':self.repairs+=1
                    if self.has_text:events.append((self.language,'',True))
                    self.has_text=False
                elif selected!=self.language:
                    if self.has_text:events.append((self.language,'',True))
                    self.language=selected;self.has_text=False
                self.pending=self.pending[len(tag):]
                continue
            if any(tag.startswith(self.pending) for tag in tags):break
            # Literal brackets (for example a grammar exercise) are not controls.
            events.append((self.language,'[',False));self.has_text=True
            self.pending=self.pending[1:]
        if final:
            if self.pending:
                # Only a partial control-marker prefix can remain buffered.
                self.pending='';self.repairs+=1
            if self.has_text:events.append((self.language,'',True));self.has_text=False
        return events


def speech_protocol(config):
    if not config['tts']['language_segments']['enabled']:return ''
    return (' Speech voice markers: start with [en] for English or [de] for German. '
            'Insert a new [en] or [de] ONLY when the spoken language changes, including quotations and examples. '
            'These markers choose the voice and are not spoken. Do not use closing tags. '
            'Keep the original word order and spaces, and keep punctuation with the words it belongs to. '
            'Never switch language for punctuation alone. Output plain spoken text, without markdown or lists. '
            'Mark isolated foreign vocabulary too: [de]obwohl[en] means although, and [de]weil[en] means because. '
            'Example: [en]You can say [de]Guten Morgen[en] to greet someone.')
