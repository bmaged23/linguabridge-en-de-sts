"""Explicit, expiring confirmations for persistent translation/language changes."""
import re
import secrets
import time
from .memory import MODES


class Controls:
    def __init__(self, memory, settings):
        self.memory=memory;self.settings=settings;self._pending=None
        # Earlier versions enabled translation without recording consent.
        # Keep history/notes, but migrate that legacy mode to conversation.
        if memory.get('mode')=='translation' and not memory.get('translation_confirmed',False):
            memory.set('mode','conversation')
        if memory.get('response_language')!='same' and not memory.get('response_language_confirmed',False):
            memory.set('response_language','same')

    @property
    def pending(self):
        if self._pending and time.monotonic()>self._pending['deadline']:
            self._pending=None
        return self._pending

    def snapshot(self):
        value=self.pending
        return {key:item for key,item in value.items() if key!='deadline'} if value else None

    def _apply(self,kind,value):
        if kind=='mode':
            self.memory.set('mode',value)
            self.memory.set('translation_confirmed',value=='translation')
        else:
            self.memory.set('response_language',value)
            self.memory.set('response_language_confirmed',True)
        self._pending=None

    def request(self,kind,value):
        if kind not in ('mode','language'):raise ValueError('Unknown change')
        if kind=='mode' and value not in MODES:raise ValueError('Unknown mode')
        if kind=='language' and value not in ('same','en','de'):raise ValueError('Unknown language')
        current=self.memory.get('mode' if kind=='mode' else 'response_language')
        if current==value:
            self._pending=None;return None
        confirm=(kind=='mode' and value=='translation' and self.settings['confirm_translation_changes']) or (
            kind=='language' and self.settings['confirm_response_language_changes'])
        if not confirm:
            self._apply(kind,value);return None
        descriptions={'en':'English','de':'German','same':'the speaker’s language'}
        label='Enable translation between English and German until you turn it off?' if kind=='mode' else f'Change conversation and teaching replies to {descriptions[value]}?'
        self._pending={'id':secrets.token_hex(12),'kind':kind,'value':value,'message':label,
                       'deadline':time.monotonic()+self.settings['confirmation_timeout_seconds']}
        return self.snapshot()

    def confirm(self,confirmation_id):
        value=self.pending
        if not value or value['id']!=confirmation_id:raise ValueError('Confirmation expired or no longer matches')
        self._apply(value['kind'],value['value'])

    def cancel(self):self._pending=None

    def voice_answer(self,text,prefixes):
        if not self.pending:return None
        text=text.lower().strip()
        text=re.sub(r'^(?:'+'|'.join(re.escape(p) for p in prefixes)+r')[\s,.:;!]+','',text)
        text=re.sub(r'[^\w\s]','',text).strip()
        if text in ('yes','yes please','yes translate','confirm','yes confirm','enable translation','ja','ja bitte','bestätigen','bestätige'):return 'confirm'
        if text in ('no','no thanks','cancel','no cancel','nein','nein danke','abbrechen','abbruch'):return 'cancel'
        return None
