import numpy as np

from audiokit import energy_vad


def test_energy_vad_detects_single_burst(burst_signal):
    sig, sr, burst = burst_signal
    segments, mask = energy_vad(sig, sr)

    assert len(segments) == 1
    # the centre of the burst must be marked active
    centre = (burst.start + burst.stop) // 2
    assert mask[centre]
    # the segment brackets the burst (with padding)
    start, end = segments[0]
    assert start <= burst.start
    assert end >= burst.stop


def test_energy_vad_empty_signal():
    segments, mask = energy_vad(np.array([], dtype=np.float32), 16000)
    assert segments == []
    assert mask.shape == (0,)


def test_energy_vad_pure_silence_detects_nothing():
    segments, mask = energy_vad(np.zeros(16000, dtype=np.float32), 16000)
    assert segments == []
    assert not mask.any()


def test_energy_vad_segment_slices_are_consistent(burst_signal):
    sig, sr, _ = burst_signal
    segments, mask = energy_vad(sig, sr)
    # mask True count equals total covered samples across segments
    covered = sum(end - start for start, end in segments)
    assert covered == int(mask.sum())


def test_energy_vad_is_scale_invariant(burst_signal):
    sig, sr, burst = burst_signal
    segments, mask = energy_vad(sig * 0.1, sr)

    assert len(segments) == 1
    centre = (burst.start + burst.stop) // 2
    assert mask[centre]
