"""Application-owned mode and memory. Summaries never modify operating rules."""
import json
import os
from pathlib import Path
import sqlite3
import time

MODES = ('translation', 'conversation', 'teaching')


class Memory:
    def __init__(self, path, settings, default_mode='conversation'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.db = sqlite3.connect(self.path)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, mode TEXT NOT NULL,
          source TEXT NOT NULL, target TEXT NOT NULL, user TEXT NOT NULL,
          heard TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS summaries (mode TEXT PRIMARY KEY, through_id INTEGER NOT NULL,
          text TEXT NOT NULL);
        ''')
        for key, value in {'mode':default_mode, 'response_language':settings['response_language']}.items():
            self.db.execute('INSERT OR IGNORE INTO state VALUES (?,?)', (key,json.dumps(value)))
        self.db.commit()
        os.chmod(self.path, 0o600)

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        if key == 'mode' and value not in MODES:
            raise ValueError('Unknown mode')
        if key == 'response_language' and value not in ('same','en','de'):
            raise ValueError('Unknown output language')
        self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',(key,json.dumps(value)))
        self.db.commit()

    def facts(self):
        return [{'id':row[0],'text':row[1]} for row in self.db.execute('SELECT id,text FROM facts ORDER BY id')]

    def remember(self, text):
        if not isinstance(text,str):raise ValueError('Memory note must be text')
        text = text.strip()
        if not text or len(text)>self.settings['max_fact_characters']:
            raise ValueError('Memory note is empty or too long')
        if len(self.facts()) >= self.settings['max_facts']:
            raise ValueError('Memory is full; remove a note first')
        self.db.execute('INSERT OR IGNORE INTO facts(text) VALUES (?)',(text,));self.db.commit()

    def forget(self, fact_id):
        self.db.execute('DELETE FROM facts WHERE id=?',(fact_id,));self.db.commit()

    def add_turn(self, mode, source, target, text):
        if not self.settings['persist_transcripts']:
            return None
        cursor = self.db.execute('INSERT INTO turns(mode,source,target,user,created) VALUES (?,?,?,?,?)',
                                 (mode,source,target,text,time.time()))
        self.db.commit();return cursor.lastrowid

    def append_heard(self, turn_id, phrase):
        if turn_id is not None:
            self.db.execute("UPDATE turns SET heard=heard || ? WHERE id=?",(phrase+' ',turn_id));self.db.commit()

    def recent(self, mode, limit=None):
        rows = self.db.execute('SELECT id,source,target,user,heard FROM turns WHERE mode=? ORDER BY id DESC LIMIT ?',
            (mode,limit or self.settings['recent_turns'])).fetchall()
        return [dict(zip(('id','source','target','user','heard'),row)) for row in reversed(rows)]

    def summary(self, mode):
        row=self.db.execute('SELECT through_id,text FROM summaries WHERE mode=?',(mode,)).fetchone()
        return row if row else (0,'')

    def summary_batch(self, mode):
        through,_=self.summary(mode)
        rows=self.db.execute('SELECT id,user,heard FROM turns WHERE mode=? AND id>? ORDER BY id',(mode,through)).fetchall()
        if len(rows)<self.settings['summary_trigger_turns']:
            return []
        return rows[:-self.settings['recent_turns']]

    def save_summary(self, mode, through, text):
        self.db.execute('INSERT OR REPLACE INTO summaries VALUES (?,?,?)',
                        (mode,through,text[:self.settings['summary_max_characters']]))
        self.db.commit()

    def clear_history(self, mode):
        self.db.execute('DELETE FROM turns WHERE mode=?',(mode,))
        self.db.execute('DELETE FROM summaries WHERE mode=?',(mode,));self.db.commit()

    def clear_all(self):
        # Operational rules intentionally survive clearing conversational memory.
        self.db.executescript('DELETE FROM turns; DELETE FROM summaries; DELETE FROM facts; VACUUM;')

    def snapshot(self):
        mode=self.get('mode')
        return {'mode':mode,'response_language':self.get('response_language'),'facts':self.facts(),
                'recent':self.recent(mode),'summary':self.summary(mode)[1],
                'persistent':self.settings['persist_transcripts']}

    def close(self):
        self.db.close()
