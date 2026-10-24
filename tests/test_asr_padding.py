import numpy as np
from scripts.runtime.qwen_asr_worker import canonicalize_padding


def test_browser_frame_padding_does_not_change_final_model_input():
    core = np.array([1e-9, .2, 0, -.1, .03], dtype=np.float32)
    first = np.concatenate((np.zeros(3316), core, np.zeros(10352))).astype(np.float32)
    second = np.concatenate((np.zeros(3508), core, np.zeros(10672))).astype(np.float32)
    a = canonicalize_padding(first, 200)
    b = canonicalize_padding(second, 200)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[3200:-3200], core)
    np.testing.assert_array_equal(canonicalize_padding(a, 200), a)


def test_silence_is_not_shortened_into_a_false_utterance():
    audio = np.zeros(64000, dtype=np.float32)
    assert canonicalize_padding(audio, 200) is audio


def test_room_noise_and_interior_pauses_are_preserved():
    audio = np.array([1e-8, .03, 0, 0, -.02, 1e-8], dtype=np.float32)
    result = canonicalize_padding(audio, 2, sample_rate=1000)
    np.testing.assert_array_equal(result[2:-2], audio)
