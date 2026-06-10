"""Signal-to-noise ratio estimation.

Two complementary estimators, de-duplicated from nkululeko and coughkit:

* ``SNREstimator`` / ``estimate_snr`` — percentile log-energy method
  (nkululeko's ``estimate_snr.py``). Needs no segmentation; robust general
  estimator.
* ``compute_snr`` — mask-based RMS(signal)/RMS(noise) using ``energy_vad``
  (coughkit's ``compute_SNR``). Best when discrete high-energy events sit in
  a quieter background.
"""

from typing import List, Tuple

import numpy as np

from .vad import energy_vad


class SNREstimator:
    """Estimate SNR from a signal via frame log-energy percentiles.

    The ratio between the mean of the high-energy frames (>=75th percentile)
    and the low-energy frames (<=25th percentile) is reported in dB.
    """

    # Floor for frame energy so log/log10 stay finite on silent frames.
    _ENERGY_FLOOR = 1e-12

    def __init__(self, input_data, sample_rate, window_size: int = 320, hop_size: int = 160):
        self.audio_data = np.asarray(input_data, dtype=np.float64)
        self.sample_rate = sample_rate
        self.frame_length = window_size
        self.hop_length = hop_size

    def frame_audio(self, signal) -> List[np.ndarray]:
        if len(signal) < self.frame_length:
            return []
        num_frames = 1 + (len(signal) - self.frame_length) // self.hop_length
        return [
            signal[i * self.hop_length : i * self.hop_length + self.frame_length]
            for i in range(num_frames)
        ]

    def calculate_log_energy(self, frame) -> float:
        energy = float(np.sum(frame ** 2))
        return float(np.log(max(energy, self._ENERGY_FLOOR)))

    def calculate_snr(self, energy_high, energy_low) -> float:
        return 10 * np.log10(
            max(float(energy_high), self._ENERGY_FLOOR)
            / max(float(energy_low), self._ENERGY_FLOOR)
        )

    def estimate_snr(self) -> Tuple[float, list, float, float]:
        """Return ``(snr_db, log_energies, low_threshold, high_threshold)``.

        Returns ``(0.0, [], 0.0, 0.0)`` when the signal is shorter than one
        analysis window.
        """
        from scipy.signal.windows import hamming  # noqa: PLC0415

        frames = self.frame_audio(self.audio_data)
        if not frames:
            return 0.0, [], 0.0, 0.0

        window = hamming(self.frame_length)
        log_energies = [self.calculate_log_energy(frame * window) for frame in frames]

        low_th = float(np.percentile(log_energies, 25))
        high_th = float(np.percentile(log_energies, 75))

        low_frames = [e for e in log_energies if e <= low_th]
        high_frames = [e for e in log_energies if e >= high_th]

        snr = self.calculate_snr(
            np.exp(np.mean(high_frames)), np.exp(np.mean(low_frames))
        )
        return float(snr), log_energies, low_th, high_th


def estimate_snr(signal, sample_rate, window_size: int = 320, hop_size: int = 160) -> float:
    """Convenience wrapper returning only the estimated SNR in dB."""
    return SNREstimator(signal, sample_rate, window_size, hop_size).estimate_snr()[0]


def compute_snr(x, sr) -> float:
    """Mask-based SNR: RMS over detected events vs. RMS over the background.

    Uses ``energy_vad`` to split the signal. Returns ``0.0`` when either the
    event or background portion is empty/silent (no meaningful ratio).
    """
    x = np.asarray(x, dtype=np.float64)
    _, mask = energy_vad(x, sr)
    sig = x[mask]
    noise = x[~mask]
    rms_signal = np.sqrt(np.mean(np.square(sig))) if len(sig) else 0.0
    rms_noise = np.sqrt(np.mean(np.square(noise))) if len(noise) else 0.0
    if rms_signal == 0 or rms_noise == 0 or np.isnan(rms_noise):
        return 0.0
    return float(20 * np.log10(rms_signal / rms_noise))
