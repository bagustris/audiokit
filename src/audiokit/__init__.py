"""audiokit — purpose-neutral audio plumbing shared by audio/speech toolkits.

Re-exports the public API so callers can ``from audiokit import read_wav`` etc.
"""

from .cli import render_mic_level, run_cli
from .contract import FeatureContract, read_contract
from .download import download_file, safe_tar_members, verify_sha256, sha256_of
from .errors import AudiokitError
from .integrity import create_manifest, verify_integrity
from .io import mic_stream, pipe_stream, read_wav, resample, wav_duration
from .scaler import NumpyStandardScaler, scaler_to_json, scaler_from_json
from .segment import (
    read_segments_csv,
    segments_to_audformat_index,
    write_segments_csv,
)
from .snr import SNREstimator, compute_snr, estimate_snr
from .vad import energy_vad

__version__ = "0.2.0"

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
    "sha256_of",
    "verify_sha256",
    # vad
    "energy_vad",
    # snr
    "SNREstimator",
    "estimate_snr",
    "compute_snr",
    # cli
    "render_mic_level",
    "run_cli",
    # contract (Priority 3)
    "FeatureContract",
    "read_contract",
    # integrity (Priority 2)
    "create_manifest",
    "verify_integrity",
    # scaler (Priority 3)
    "NumpyStandardScaler",
    "scaler_to_json",
    "scaler_from_json",
    # segment (Priority 3)
    "write_segments_csv",
    "read_segments_csv",
    "segments_to_audformat_index",
]
