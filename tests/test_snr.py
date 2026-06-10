import numpy as np

from audiokit import SNREstimator, compute_snr, estimate_snr


def _sine(freq, dur, sr, amp=1.0):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_estimator_high_snr_is_positive():
    sr = 16000
    loud = _sine(300.0, 0.5, sr, amp=1.0)
    quiet = _sine(300.0, 0.5, sr, amp=0.01)
    sig = np.concatenate([loud, quiet])
    snr, energies, low, high = SNREstimator(sig, sr).estimate_snr()
    assert snr > 10
    assert energies and high >= low


def test_estimator_silence_is_zero():
    snr = estimate_snr(np.zeros(16000, dtype=np.float32), 16000)
    assert abs(snr) < 1e-6


def test_estimator_short_signal_returns_zero_tuple():
    snr, energies, low, high = SNREstimator(np.ones(100, dtype=np.float32), 16000).estimate_snr()
    assert (snr, energies, low, high) == (0.0, [], 0.0, 0.0)


def test_compute_snr_event_over_background_is_positive():
    sr = 16000
    rng = np.random.default_rng(0)
    background = (rng.standard_normal(int(1.3 * sr)) * 0.005).astype(np.float32)
    sig = background.copy()
    burst = _sine(200.0, 0.3, sr, amp=1.0)
    start = int(0.5 * sr)
    sig[start : start + len(burst)] += burst
    assert compute_snr(sig, sr) > 10


def test_compute_snr_no_event_returns_zero():
    assert compute_snr(np.zeros(16000, dtype=np.float32), 16000) == 0.0
