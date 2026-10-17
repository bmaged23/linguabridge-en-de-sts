"""One connected story, rendered as continuous recordings in four language layouts."""
STORIES={
 'english': [('en','Yesterday Anna and I travelled to Berlin for an interview. At the station she said our train was cancelled. We found a bus, but I told her not to buy tickets yet. Then the company called. The interview was at ten, not nine. We arrived at nine forty five. Please summarize what happened in two sentences.')],
 'german': [('de','Gestern fuhren Anna und ich für ein Vorstellungsgespräch nach Berlin. Am Bahnhof sagte sie, unser Zug fällt aus. Wir fanden einen Bus, aber ich sagte, kaufe noch keine Tickets. Dann rief die Firma an. Das Gespräch war um zehn, nicht um neun. Wir kamen um neun Uhr fünfundvierzig an. Bitte fasse die Geschichte in zwei Sätzen zusammen.')],
 'en_de_en': [('en','Yesterday Anna and I travelled to Berlin for an interview.'),('de','Am Bahnhof sagte sie, unser Zug fällt aus.'),('en','We found a bus, but I said,'),('de','bitte kaufe noch keine Tickets.'),('en','Then the company called. The interview was at ten, not nine.'),('de','Wir kamen um neun Uhr fünfundvierzig an.'),('en','Please summarize what happened in two sentences.')],
 'de_en_de': [('de','Gestern fuhren Anna und ich nach Berlin zu einem Vorstellungsgespräch.'),('en','At the station she said our train was cancelled.'),('de','Wir fanden einen Bus, aber ich sagte,'),('en','please do not buy tickets yet.'),('de','Dann rief die Firma an. Das Gespräch war um zehn, nicht um neun.'),('en','We arrived at nine forty five.'),('de','Bitte fasse die Geschichte in zwei Sätzen zusammen.')],
}


def cases():
 rows=[]
 for name,spans in STORIES.items():
  for variant in ('clean','quiet','noise20','noise10','pause','noise15'):
   rows.append(dict(id=f'story_{name}_{variant}',spans=spans,kind='mixed' if len(spans)>1 else spans[0][0],variant=variant,expected_speech=True,mode='conversation'))
 for name in ('en_de_en','de_en_de'):
  for variant in ('noise8','noise25','quiet_noise20','pause_noise20','repeat1','repeat2','repeat3','repeat4'):
   rows.append(dict(id=f'story_{name}_{variant}',spans=STORIES[name],kind='mixed',variant=variant,expected_speech=True,mode='conversation'))
 for row in rows:
  row['reference']=' '.join(text for _,text in row['spans'])
  row['story_facts']=['Anna','Berlin','cancelled train / Zug fällt aus','do not buy tickets / keine Tickets','interview ten, not nine / zehn, nicht neun','arrival nine forty five / neun Uhr fünfundvierzig']
 return rows


def extended_cases():
 rows=[]
 for name in ('en_de_en','de_en_de'):
  spans=[*STORIES[name][:-1],('en','Please retell this story briefly using both English and German.') if name=='en_de_en' else ('de','Bitte erzähle diese Geschichte kurz auf Englisch und Deutsch.')]
  for variant in ('clean','noise20','quiet','pause'):
   rows.append(dict(id=f'bilingual_retelling_{name}_{variant}',spans=spans,kind='mixed',variant=variant,expected_speech=True,mode='conversation'))
  continuation=[('en','Before entering the office, Anna checked the invitation again. We had brought our documents, but the receptionist asked us to wait downstairs.'),('de','Während wir warteten, erklärte ich ihr noch einmal, dass wir keine Bustickets gekauft hatten und dass die neue Uhrzeit nur eine Korrektur war.'),('en','The manager finally welcomed us and offered us some water. Please retell the complete story briefly using both English and German.')]
  rows.append(dict(id=f'long_story_{name}',spans=[*STORIES[name][:-1],*continuation],kind='mixed',variant='clean',expected_speech=True,mode='conversation',stress='May exceed the configured 30-second utterance limit; inspect premature endpoint cancellation'))
 for row in rows:row['reference']=' '.join(text for _,text in row['spans'])
 return rows
