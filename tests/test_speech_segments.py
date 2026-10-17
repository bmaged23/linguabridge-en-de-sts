import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest
from voice_bridge.speech_segments import LanguageSegments,SpeechPhrase
from voice_bridge import server
from voice_bridge.memory import Memory
from voice_bridge.settings import load_config


@pytest.mark.parametrize('text,languages',[
 ('[en]You can say [/en][de]Guten Morgen[/de][en] to greet someone.[/en]',['en','de','en']),
 ('[de]Du kannst [/de][en]Good morning[/en][de] sagen.[/de]',['de','en','de']),
])
def test_markers_split_at_every_character(text,languages):
    parser=LanguageSegments();events=[]
    for char in text:events+=parser.push(char)
    events+=parser.push(final=True)
    assert [lang for lang,_,closed in events if closed]==languages
    assert '[' not in ''.join(value for _,value,_ in events)
    assert ''.join(value for _,value,_ in events)==text.replace('[en]','').replace('[/en]','').replace('[de]','').replace('[/de]','')


@pytest.mark.parametrize('text,expected',[
 ('[en]Say [de]Guten Morgen[/en].','Say Guten Morgen.'),
 ('[de]Hallo[/de], [en]hello[/en].','Hallo, hello.'),
 ('[en]First [en]second [de]dritter','First second dritter'),
 ('[en]Hello.[e','Hello.'),
 ('Plain unmarked text','Plain unmarked text'),
])
def test_real_model_format_variations_do_not_abort(text,expected):
    parser=LanguageSegments();events=[]
    for char in text:events+=parser.push(char)
    events+=parser.push(final=True)
    assert ''.join(value for _,value,_ in events)==expected


def test_unmarked_german_uses_requested_main_language():
    parser=LanguageSegments('de')
    assert parser.push('Hallo.')[0]==('de','Hallo.',False)


@pytest.mark.asyncio
@pytest.mark.parametrize('spans',[
 [('en','Please say '),('de','Guten Morgen'),('en',' to greet someone.')],
 [('de','Bitte sage '),('en','Good morning'),('de',' zur Begrüßung.')],
])
async def test_mixed_stream_routes_each_voice_and_preserves_heard_text(tmp_path,monkeypatch,spans):
    config=load_config();memory=Memory(tmp_path/'memory.sqlite3',config['memory'])
    monkeypatch.setattr(server.app.state,'settings',config,raising=False)
    monkeypatch.setattr(server.app.state,'memory',memory,raising=False)
    monkeypatch.setattr(server.app.state,'diagnostics',{},raising=False)
    packets=[];calls=[]
    class Socket:
        async def send_json(self,packet):
            packets.append(packet)
            if packet['type']=='audio':voice.acknowledge({'id':packet['id'],'complete':True})
    voice=server.Session(Socket());voice.output_language=spans[0][0]
    row=memory.add_turn('teaching','en','en','Synthetic bilingual example')
    voice.answers[0]={'id':row,'heard':''}
    def synth(text,language):calls.append((text,language));return b'\0\0',24000 if language=='en' else 22050
    monkeypatch.setattr(server,'synthesize',synth)
    executor=ThreadPoolExecutor(1);monkeypatch.setattr(server.app.state,'tts',executor,raising=False)
    tokens=asyncio.Queue();phrases=asyncio.Queue(3)
    for lang,text in spans:
        for char in f'[{lang}]{text}[/{lang}]':tokens.put_nowait(char)
    tokens.put_nowait(None)
    try:
        await asyncio.wait_for(asyncio.gather(voice.chunk_tokens(tokens,phrases),voice.speak(phrases,0)),2)
        assert calls==[(text.strip(),lang) for lang,text in spans]
        audio=[p for p in packets if p['type']=='audio']
        assert [p['language'] for p in audio]==[lang for lang,_ in spans]
        assert [p['sample_rate'] for p in audio]==[24000 if lang=='en' else 22050 for lang,_ in spans]
        visible=''.join(p['text'] for p in packets if p['type']=='token')
        assert visible==''.join(text for _,text in spans)
        assert '[' not in memory.recent('teaching')[0]['heard']
        assert memory.recent('teaching')[0]['heard']==' '.join(text.strip() for _,text in spans)+' '
    finally:executor.shutdown();memory.close()


@pytest.mark.asyncio
async def test_first_phrase_streams_before_segment_closes(tmp_path,monkeypatch):
    config=load_config();memory=Memory(tmp_path/'memory.sqlite3',config['memory'])
    monkeypatch.setattr(server.app.state,'settings',config,raising=False)
    monkeypatch.setattr(server.app.state,'memory',memory,raising=False)
    class Socket:
        async def send_json(self,packet):pass
    voice=server.Session(Socket());tokens=asyncio.Queue();phrases=asyncio.Queue()
    task=asyncio.create_task(voice.chunk_tokens(tokens,phrases))
    text='This English sentence should start playing immediately.'
    try:
        await tokens.put('[en]'+text)
        first=await asyncio.wait_for(phrases.get(),.5)
        assert first==SpeechPhrase(text,'en')
        assert not task.done()
        await tokens.put('[/en]');await tokens.put(None)
        await asyncio.wait_for(task,.5)
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True);memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('items,expected',[
 ([SpeechPhrase('Guten Morgen','de'),SpeechPhrase('.','en')],[('Guten Morgen','de')]),
 ([SpeechPhrase('Hallo','de'),SpeechPhrase('! …','en'),SpeechPhrase('Guten Tag','de')],[('Hallo','de'),('Guten Tag','de')]),
 ([SpeechPhrase('Hello','en'),SpeechPhrase('Guten Tag','de'),SpeechPhrase('".','en')],[('Hello','en'),('Guten Tag','de')]),
 ([SpeechPhrase('I','en'),SpeechPhrase('Ja','de'),SpeechPhrase('123','de')],[('I','en'),('Ja','de'),('123','de')]),
])
async def test_punctuation_cannot_trigger_a_voice_switch(tmp_path,monkeypatch,items,expected):
    config=load_config();memory=Memory(tmp_path/'memory.sqlite3',config['memory'])
    monkeypatch.setattr(server.app.state,'settings',config,raising=False)
    monkeypatch.setattr(server.app.state,'memory',memory,raising=False)
    monkeypatch.setattr(server.app.state,'diagnostics',{},raising=False)
    packets=[];calls=[]
    class Socket:
        async def send_json(self,packet):
            packets.append(packet)
            if packet['type']=='audio':voice.acknowledge({'id':packet['id'],'complete':True})
    voice=server.Session(Socket())
    def synth(text,language):calls.append((text,language));return b'\0\0',24000
    monkeypatch.setattr(server,'synthesize',synth)
    monkeypatch.setattr(server,'synthesize_batch',lambda texts,lang:[synth(t,lang) for t in texts])
    executor=ThreadPoolExecutor(1);monkeypatch.setattr(server.app.state,'tts',executor,raising=False)
    phrases=asyncio.Queue()
    for value in [*items,None]:phrases.put_nowait(value)
    try:
        await asyncio.wait_for(voice.speak(phrases,0),2)
        assert calls==expected
        assert [p['language'] for p in packets if p['type']=='audio']==[language for _,language in expected]
    finally:executor.shutdown();memory.close()
