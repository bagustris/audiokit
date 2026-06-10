"""Shared fixtures: synthetic audio written to temp WAV files."""

import numpy as np
import pytest
import soundfile as sf


def _sine(freq: float, dur: float, sr: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def sine_signal():
    """A deterministic mono 200 Hz sine: (signal, sample_rate)."""
    sr = 16000
    return _sine(200.0, 1.0, sr), sr


@pytest.fixture
def mono_wav(tmp_path, sine_signal):
    """Path to a mono 16 kHz float WAV plus its (signal, sr)."""
    sig, sr = sine_signal
    path = tmp_path / "mono.wav"
    sf.write(str(path), sig, sr, subtype="FLOAT")
    return str(path), sig, sr


@pytest.fixture
def stereo_wav(tmp_path):
    """Path to a 2-channel WAV (used to assert mono-only enforcement)."""
    sr = 16000
    sig = np.zeros((sr, 2), dtype=np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), sig, sr, subtype="FLOAT")
    return str(path)


@pytest.fixture
def burst_signal():
    """Silence, then a loud 200 Hz burst, then silence: (signal, sr, burst_slice)."""
    sr = 16000
    pre = np.zeros(int(0.5 * sr), dtype=np.float32)
    burst = _sine(200.0, 0.3, sr, amp=1.0)
    post = np.zeros(int(0.5 * sr), dtype=np.float32)
    sig = np.concatenate([pre, burst, post])
    return sig, sr, slice(len(pre), len(pre) + len(burst))
