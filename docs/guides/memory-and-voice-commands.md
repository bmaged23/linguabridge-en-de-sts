# Memory and voice commands

The mode lives in SQLite rather than relying on the LLM to remember an instruction. The fresh-project default is conversation, with replies matching the speaker’s language. Enabling translation or changing the conversation/teaching reply language first creates a confirmation request; the current settings remain active until you say yes or click Yes, confirm. Say no or click Cancel to keep the current settings. Requests expire after 45 seconds. Confirmed translation then persists until explicitly changed; it does not ask on every turn. Older unconfirmed translation settings are migrated to conversation without deleting history. It survives reconnecting, refreshing, and restarting the server. Changing `modes.default` only affects a new database; select a mode in the UI to change an existing saved setting.

Use the mode controls or explicit commands:

| Say | Result |
| --- | --- |
| “LinguaBridge, translator mode.” | Ask to enable English ⇄ German translation; say yes to activate |
| “LinguaBridge, I will speak English, translate into German.” | Ask to enable the same bidirectional translator |
| “Brücke, Übersetzungsmodus.” | Ask to enable translation |
| “LinguaBridge, conversation mode.” / “Brücke, Gesprächsmodus.” | Bilingual conversation |
| “LinguaBridge, teaching mode.” / “Brücke, Lernmodus.” | Teaching mode |
| “LinguaBridge, reply in German.” | Ask to save German as the conversation/teaching reply language |
| “LinguaBridge, reply in the same language.” | Return to matching the speaker |
| “LinguaBridge, remember that my German level is A2.” | Save an explicit note across sessions |
| “LinguaBridge, what do you remember?” | Read saved notes |

The `LinguaBridge` / `Brücke` prefix (with `Bridge` retained as a compatible alias) distinguishes app commands from dialogue being translated. Prefix-free instructions in translator mode are translated as content. Supported command patterns are deterministic, not unrestricted natural-language command understanding. Use the UI if a name or command is misrecognized.

Memory has separate layers:

1. **Operating state:** mode and response-language preference; injected on every request and never overwritten by a summary.
2. **Pinned notes:** explicit user facts/preferences, editable in the memory panel. Used in conversation and teaching; not inserted into translation text.
3. **Recent history:** separate for each mode, including original text and only fully played assistant phrases. Interrupted/unplayed text is not claimed as heard.
4. **Older summaries:** after 20 unsummarized turns, compact older conversation/teaching context while preserving 12 recent turns. Summaries run after the reply with a four-second deadline, preserving the previous summary and raw history if they fail. Summaries can lose details; pin facts that must remain exact.

“Clear this history” clears the current mode's history/summary, retaining pinned notes and mode. “Erase all notes and conversation history” clears all three histories, summaries, and notes while retaining operational mode/language settings. Stop the microphone before deleting `data/memory.sqlite3` manually. Set `memory.persist_transcripts=False` to keep new raw turn history only for the active session; operating settings and explicit notes still persist.
