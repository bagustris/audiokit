"""Shared test fixtures — reusable across audiokit, coughkit, sherox, nkululeko.

Provides factory functions for synthetic audio so that each repo's ``conftest.py``
can import from here instead of redefining ``_sine()``, ``_silence()``, etc.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import soundfile as sf


def make_sine(
    freq: float = 200.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 1.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return a mono sine wave array.

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Samples per second.
        amplitude: Peak amplitude.
        dtype: Output numpy dtype.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(dtype)


def make_silence(
    duration: float = 1.0,
    sample_rate: int = 16000,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return a silent (all-zero) mono array."""
    return np.zeros(int(duration * sample_rate), dtype=dtype)


def make_noise(
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.1,
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> np.ndarray:
    """Return a white-noise mono array."""
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(duration * sample_rate))).astype(
        dtype
    )


def make_burst(
    freq: float = 200.0,
    burst_duration: float = 0.3,
    pre_duration: float = 0.5,
    post_duration: float = 0.5,
    sample_rate: int = 16000,
    amplitude: float = 1.0,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, int, slice]:
    """Return (signal, sr, burst_slice) — silence, loud sine burst, silence.

    The returned slice gives the *exact* burst region (pre padding, post padding)
    so tests can verify that event detection brackets the burst correctly.
    """
    pre = make_silence(pre_duration, sample_rate, dtype)
    burst = make_sine(freq, burst_duration, sample_rate, amplitude, dtype)
    post = make_silence(post_duration, sample_rate, dtype)
    sig = np.concatenate([pre, burst, post])
    burst_slice = slice(len(pre), len(pre) + len(burst))
    return sig, sample_rate, burst_slice


def write_mono_wav(path: "Path | str", signal: np.ndarray, sample_rate: int) -> str:
    """Write a mono WAV file and return its path as a string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, sample_rate, subtype="FLOAT")
    return str(path)


def multi_tone(
    frequencies: Tuple[float, ...],
    duration: float = 1.0,
    sample_rate: int = 16000,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return a sum-of-sines mono array (useful for testing multi-peak VAD)."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t, dtype=np.float64)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    return (signal / len(frequencies)).astype(dtype)


__all__ = [
    "make_sine",
    "make_silence",
    "make_noise",
    "make_burst",
    "write_mono_wav",
    "multi_tone",
]
