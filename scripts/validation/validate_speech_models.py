"""Real bilingual Kokoro/Piper → multilingual Whisper round trip, no cloud calls."""
from pathlib import Path
import json
import wave
import numpy as np
from voice_bridge.settings import ROOT,load_settings
from voice_bridge.workers import init_tts,init_stt,synthesize,transcribe

SAMPLES={'en':'We are looking for the train station. The meeting starts at three in the afternoon.',
         'de':'Wir suchen den Bahnhof. Das Treffen beginnt um drei Uhr am Nachmittag.'}


def main():
    config=load_settings();out=ROOT/'logs/speech-check';out.mkdir(parents=True,exist_ok=True)
    init_tts(config['tts'],ROOT)
    for language,text in SAMPLES.items():
        pcm,rate=synthesize(text,language)
        with wave.open(str(out/(language+'.wav')),'wb') as handle:
            handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate);handle.writeframes(pcm)
    settings=dict(config['stt']);settings['model_path']=config['paths']['stt_model'];init_stt(settings,ROOT)
    report={}
    for language in SAMPLES:
        with wave.open(str(out/(language+'.wav')),'rb') as handle:
            rate=handle.getframerate();audio=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
        audio=np.interp(np.arange(0,len(audio),rate/16000),np.arange(len(audio)),audio).astype(np.float32)
        result=transcribe(audio)
        assert result['certain'] and result['language']==language,result
        assert ('station' in result['text'].lower() if language=='en' else 'bahnhof' in result['text'].lower()),result
        report[language]=result
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Kokoro English, Piper German, and GPU multilingual STT: PASS')


if __name__=='__main__':main()
