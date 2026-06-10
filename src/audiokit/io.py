"""Audio I/O and resampling — purpose-neutral building blocks.

Seeded from sherox's ``audio.py``. Heavy/optional backends (soundfile,
sounddevice) are imported lazily so ``import audiokit`` never fails just
because a microphone backend is missing.
"""

import logging
import queue
from math import gcd
from types import SimpleNamespace
from typing import BinaryIO, Generator, Optional

import numpy as np

from .errors import AudiokitError

_log = logging.getLogger("audiokit.io")

# Lazily-populated module handles (filled on first use).
sd = SimpleNamespace(InputStream=None)
sf = SimpleNamespace(SoundFile=None, write=None)


def _require_soundfile():
    global sf
    if getattr(sf, "SoundFile", None) is not None:
        return sf
    try:
        import soundfile as _soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AudiokitError(
            "soundfile is required for reading/writing audio files. "
            "Install it with: pip install soundfile"
        ) from exc
    sf = _soundfile
    return sf


def _require_sounddevice():
    global sd
    if getattr(sd, "InputStream", None) is not None:
        return sd
    try:
        import sounddevice as _sounddevice  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AudiokitError(
            "sounddevice is required for microphone input. "
            "Install it with: pip install 'audiokit[mic]'"
        ) from exc
    sd = _sounddevice
    return sd


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resample from ``orig_sr`` to ``target_sr``.

    Uses ``scipy.signal.resample_poly`` with a Kaiser window (β=14, ~90 dB
    stopband). Polyphase FIR is preferred over FFT resampling for audio
    because it does not assume a periodic signal and avoids wrap-around
    ringing at the edges. Falls back to linear interpolation if scipy is
    somehow unavailable.
    """
    if orig_sr == target_sr:
        return np.asarray(data, dtype=np.float32)
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415

        g = gcd(target_sr, orig_sr)
        return resample_poly(
            data,
            target_sr // g,
            orig_sr // g,
            window=("kaiser", 14.0),
            padtype="line",
        ).astype(np.float32)
    except ImportError:  # pragma: no cover - scipy is a core dep
        n_orig = len(data)
        n_new = int(n_orig * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, n_orig - 1, n_new),
            np.arange(n_orig),
            data,
        ).astype(np.float32)


def read_wav(
    path: str,
    target_sr: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read a mono audio file and yield float32 chunks of ``chunk_size`` seconds.

    When the file's sample rate matches ``target_sr``, audio is streamed
    directly without loading the full file into RAM. When resampling is
    required, the file is loaded once and resampled with ``resample`` before
    chunking, avoiding boundary artifacts that block-wise resampling causes.

    Raises ``AudiokitError`` if the file is not mono.
    """
    _sf = _require_soundfile()
    with _sf.SoundFile(path) as f:
        if f.channels != 1:
            raise AudiokitError(
                f"Expected mono audio, got {f.channels} channels. "
                f"Convert with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        orig_sr = f.samplerate
        chunk_frames = int(target_sr * chunk_size)

        if orig_sr == target_sr:
            while True:
                block = f.read(frames=chunk_frames, dtype="float32")
                if len(block) == 0:
                    break
                yield block
            return

        # Rates differ: load full file, resample once, then chunk.
        data = f.read(dtype="float32")

    data = resample(data, orig_sr, target_sr)
    offset = 0
    while offset < len(data):
        yield data[offset : offset + chunk_frames]
        offset += chunk_frames


def wav_duration(path: str) -> float:
    """Return the duration of an audio file in seconds."""
    _sf = _require_soundfile()
    with _sf.SoundFile(path) as f:
        return len(f) / f.samplerate


def mic_stream(
    capture_rate: int = 16000,
    chunk_size: float = 0.1,
) -> Generator[np.ndarray, None, None]:
    """Capture microphone audio and yield float32 chunks via a queue.

    Uses a callback-based InputStream so capture never blocks the consumer.
    PortAudio status messages are logged as warnings rather than printed.
    Requires the ``[mic]`` extra (sounddevice).
    """
    _sd = _require_sounddevice()
    chunk_frames = int(capture_rate * chunk_size)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(indata, frames, time, status):  # noqa: ANN001
        if status:
            _log.warning("[audio] %s", status)
        q.put(indata[:, 0].copy())

    stream = _sd.InputStream(
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_frames,
        callback=_callback,
    )
    stream.start()
    try:
        while True:
            yield q.get()
    finally:
        stream.stop()
        stream.close()


def pipe_stream(
    capture_rate: int = 16000,
    chunk_size: float = 0.16,
    stream: Optional[BinaryIO] = None,
) -> Generator[np.ndarray, None, None]:
    """Read raw 16-bit little-endian mono PCM and yield float32 chunks.

    Reads from ``stream`` (a binary file-like) or ``sys.stdin.buffer`` when
    ``stream`` is None. Stops cleanly at EOF. The last partial chunk is
    zero-padded so all yielded arrays have a consistent length.

    Typical usage::

        arecord -f S16_LE -r 16000 -c 1 | my-tool --pipe
    """
    import sys  # noqa: PLC0415

    chunk_frames = int(capture_rate * chunk_size)
    bytes_per_chunk = chunk_frames * 2  # int16 = 2 bytes/sample
    buf = stream if stream is not None else sys.stdin.buffer
    while True:
        data = buf.read(bytes_per_chunk)
        if not data:
            break
        if len(data) < bytes_per_chunk:
            data = data + b"\x00" * (bytes_per_chunk - len(data))
        yield np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
