"""Energy-based voice/event activity detection.

``energy_vad`` is a generalisation of coughkit's hysteresis comparator
(``segment_cough``). Unlike speech-only neural VADs, it detects *any*
high-energy events — coughs, claps, impulsive machinery sounds — using only
NumPy, with no model download.
"""

from typing import List, Tuple

import numpy as np


def energy_vad(
    x: np.ndarray,
    sr: int,
    padding: float = 0.2,
    min_len: float = 0.2,
    low_mult: float = 0.1,
    high_mult: float = 2.0,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Segment a signal into high-energy events with a hysteresis comparator.

    An event starts when instantaneous power exceeds ``high_mult × RMS²`` and
    ends after power stays below ``low_mult × RMS²`` for longer than a 1%
    sample-rate tolerance. ``padding`` seconds are added to each side, and
    events shorter than ``min_len`` seconds (excluding padding) are dropped.

    Args:
        x: 1-D signal.
        sr: sample rate in Hz.
        padding: seconds added before/after each detected event.
        min_len: minimum event length in seconds (padding excluded).
        low_mult: low hysteresis threshold as a multiple of mean power (RMS²).
        high_mult: high hysteresis threshold as a multiple of mean power (RMS²).

    Returns:
        ``(segments, mask)`` where ``segments`` is a list of half-open
        ``(start, end)`` sample-index pairs (so ``x[start:end]`` is the event)
        and ``mask`` is a boolean array, True where an event is in progress.
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.zeros(len(x), dtype=bool)
    segments: List[Tuple[int, int]] = []

    if len(x) == 0:
        return segments, mask

    signal_power = np.square(x)
    rms = np.sqrt(np.mean(signal_power))
    if rms == 0:  # pure silence — nothing to detect
        return segments, mask

    # Compare instantaneous power against thresholds in the same units.  Using
    # RMS directly here makes detection depend on the absolute signal scale
    # (power scales with amplitude², RMS only with amplitude).
    mean_power = rms * rms
    th_low = low_mult * mean_power
    th_high = high_mult * mean_power
    pad = round(sr * padding)
    min_samples = round(sr * min_len)
    tolerance = round(0.01 * sr)

    start = 0
    in_event = False
    below_counter = 0

    def _commit(end_inclusive: int, start_idx: int) -> None:
        if end_inclusive + 1 - start_idx - 2 * pad > min_samples:
            seg_end = end_inclusive + 1
            segments.append((start_idx, seg_end))
            mask[start_idx:seg_end] = True

    for i, power in enumerate(signal_power):
        if in_event:
            if power < th_low:
                below_counter += 1
                if below_counter > tolerance:
                    end = i + pad if (i + pad < len(x)) else len(x) - 1
                    in_event = False
                    _commit(end, start)
            elif i == len(x) - 1:
                in_event = False
                _commit(i, start)
            else:
                below_counter = 0
        else:
            if power > th_high:
                start = i - pad if (i - pad >= 0) else 0
                in_event = True
                below_counter = 0

    return segments, mask
