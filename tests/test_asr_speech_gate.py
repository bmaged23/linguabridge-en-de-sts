import numpy as np
from voice_bridge.qwen_bridge import QwenBridge


def test_speech_regions_reset_state_and_keep_last_partial_frame():
    class Detector:
        def reset(self):self.index=0
        def __call__(self,frame):
            assert frame.shape==(512,)
            score=[.1,.9,.8][self.index];self.index+=1;return score
    bridge=QwenBridge.__new__(QwenBridge)
    bridge.settings={'final_segmentation':{'vad_speech_threshold':.5}}
    bridge.speech_detector=Detector()
    audio=np.ones(1300,dtype=np.float32)
    assert bridge.speech_regions(audio)==[[512,1300]]
    assert bridge.speech_regions(audio)==[[512,1300]]
