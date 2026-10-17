import pytest
from voice_bridge.controls import Controls
from voice_bridge.memory import Memory
from voice_bridge.routing import command
from voice_bridge.settings import load_config


@pytest.fixture
def system(tmp_path):
    config=load_config();memory=Memory(tmp_path/'memory.sqlite3',config['memory'],config['modes']['default'])
    yield memory,Controls(memory,config['controls'])
    memory.close()


def test_new_project_starts_in_conversation(system):
    memory,controls=system
    assert memory.get('mode')=='conversation' and memory.get('response_language')=='same'


def test_translation_requires_confirmation_and_can_be_declined(system):
    memory,controls=system
    proposal=controls.request('mode','translation')
    assert memory.get('mode')=='conversation' and proposal
    assert controls.voice_answer('No, thanks.',['bridge','brücke'])=='cancel'
    controls.cancel()
    assert memory.get('mode')=='conversation' and controls.pending is None


def test_confirmed_mode_persists_across_reopen(system):
    memory,controls=system
    proposal=controls.request('mode','translation');controls.confirm(proposal['id'])
    assert memory.get('mode')=='translation' and memory.get('translation_confirmed')
    Controls(memory,load_config()['controls'])
    assert memory.get('mode')=='translation'
    assert controls.request('mode','translation') is None


def test_language_change_waits_for_confirmation(system):
    memory,controls=system
    proposal=controls.request('language','de')
    assert memory.get('response_language')=='same'
    controls.confirm(proposal['id']);assert memory.get('response_language')=='de'


def test_stale_confirmation_cannot_apply_a_different_change(system):
    memory,controls=system
    first=controls.request('mode','translation');second=controls.request('language','de')
    with pytest.raises(ValueError):controls.confirm(first['id'])
    assert memory.get('mode')=='conversation' and memory.get('response_language')=='same'
    controls.confirm(second['id']);assert memory.get('response_language')=='de'


def test_confirmation_expires_and_yes_is_not_a_global_command(system):
    memory,controls=system
    assert controls.voice_answer('yes',['bridge']) is None
    proposal=controls.request('mode','translation');controls._pending['deadline']=0
    with pytest.raises(ValueError):controls.confirm(proposal['id'])
    assert memory.get('mode')=='conversation'


def test_topic_mentions_and_negations_are_not_mode_commands():
    prefixes=['bridge','brücke']
    for text in ['Bridge, can you explain how translation works?', 'Bridge, do not speak German.',
                 'Bridge, what is German grammar?', 'Bridge, tell me about teaching methods.']:
        assert command(text,prefixes) is None,text
    assert command('Bridge, translator mode.',prefixes)==('mode','translation')
    assert command('Bridge, reply in German.',prefixes)==('language','de')
    assert command('Brücke, Übersetzungsmodus.',prefixes)==('mode','translation')
