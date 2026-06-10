"""audiokit — purpose-neutral audio plumbing shared by audio/speech toolkits.

Re-exports the public API so callers can ``from audiokit import read_wav`` etc.
"""

from .cli import render_mic_level, run_cli
from .download import download_file, safe_tar_members
from .errors import AudiokitError
from .io import mic_stream, pipe_stream, read_wav, resample, wav_duration
from .snr import SNREstimator, compute_snr, estimate_snr
from .vad import energy_vad

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AudiokitError",
    # io
    "read_wav",
    "wav_duration",
    "mic_stream",
    "pipe_stream",
    "resample",
    # download
    "download_file",
    "safe_tar_members",
    # vad
    "energy_vad",
    # snr
    "SNREstimator",
    "estimate_snr",
    "compute_snr",
    # cli
    "render_mic_level",
    "run_cli",
]
