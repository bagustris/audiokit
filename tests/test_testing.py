"""Tests for audiokit.testing (Priority 2 — shared test fixtures)."""

import numpy as np
import pytest

from audiokit.testing import (
    make_burst,
    make_noise,
    make_silence,
    make_sine,
    multi_tone,
    write_mono_wav,
)


def test_make_sine_creates_correct_frequency():
    sig = make_sine(200.0, 1.0, 16000)
    assert len(sig) == 16000
    assert sig.dtype == np.float32
    # Check the signal is roughly a sine by ensuring it crosses zero often
    zero_crossings = np.sum(np.abs(np.diff(np.sign(sig)))) // 2
    assert zero_crossings > 300  # 200 Hz * 2 seconds of zero crossings


def test_make_sine_amplitude():
    sig = make_sine(200.0, 1.0, 16000, amplitude=0.5)
    assert np.max(np.abs(sig)) == pytest.approx(0.5, abs=0.01)


def test_make_silence():
    sig = make_silence(2.0, 8000)
    assert len(sig) == 16000
    np.testing.assert_array_equal(sig, np.zeros_like(sig))


def test_make_noise():
    sig1 = make_noise(1.0, 16000, seed=42)
    sig2 = make_noise(1.0, 16000, seed=42)
    np.testing.assert_array_equal(sig1, sig2)
    assert np.std(sig1) > 0.01  # actual noise present


def test_make_noise_diff_seed():
    sig1 = make_noise(1.0, 16000, seed=1)
    sig2 = make_noise(1.0, 16000, seed=2)
    assert not np.allclose(sig1, sig2)


def test_make_burst_returns_correct_slice():
    sig, sr, burst_slice = make_burst(400.0, 0.2, 0.3, 0.3, 16000)
    assert sr == 16000
    # The burst region should have high energy
    assert np.max(np.abs(sig[burst_slice])) > 0.9
    # The pre region should be silent
    assert np.all(sig[: burst_slice.start] == 0)
    # The post region should be silent
    assert np.all(sig[burst_slice.stop :] == 0)


def test_write_mono_wav(tmp_path):
    sig = make_sine(500.0, 0.5, 16000)
    path = write_mono_wav(tmp_path / "output.wav", sig, 16000)
    import soundfile as sf
    loaded, sr = sf.read(path)
    assert sr == 16000
    assert len(loaded) == 8000
    np.testing.assert_array_almost_equal(loaded, sig)


def test_multi_tone():
    sig = multi_tone((200, 400, 600), 0.5, 16000)
    assert len(sig) == 8000
    # Should be quieter than a single tone (3 components summed, divided by 3)
    assert np.max(np.abs(sig)) < 1.0
