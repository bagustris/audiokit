import io as _io

import numpy as np
import pytest

from audiokit import AudiokitError, read_wav, resample, wav_duration
from audiokit.io import pipe_stream


def test_read_wav_same_rate_reconstructs(mono_wav):
    path, sig, sr = mono_wav
    chunks = list(read_wav(path, target_sr=sr, chunk_size=0.1))
    assert all(c.dtype == np.float32 for c in chunks)
    out = np.concatenate(chunks)
    assert len(out) == len(sig)
    assert np.allclose(out, sig, atol=1e-6)


def test_read_wav_resamples_to_half(mono_wav):
    path, sig, sr = mono_wav
    out = np.concatenate(list(read_wav(path, target_sr=sr // 2, chunk_size=0.1)))
    assert abs(len(out) - len(sig) // 2) <= 2


def test_wav_duration(mono_wav):
    path, sig, sr = mono_wav
    assert wav_duration(path) == pytest.approx(len(sig) / sr, abs=1e-6)


def test_read_wav_rejects_stereo(stereo_wav):
    with pytest.raises(AudiokitError):
        list(read_wav(stereo_wav))


def test_resample_identity_returns_float32(sine_signal):
    sig, sr = sine_signal
    out = resample(sig, sr, sr)
    assert out.dtype == np.float32
    assert np.array_equal(out, sig)


def test_resample_changes_length(sine_signal):
    sig, sr = sine_signal
    out = resample(sig, sr, sr // 2)
    assert abs(len(out) - len(sig) // 2) <= 2


def test_pipe_stream_reads_until_eof():
    sr, chunk_size = 16000, 0.16
    chunk_frames = int(sr * chunk_size)
    samples = np.arange(chunk_frames * 2, dtype=np.int16)
    stream = _io.BytesIO(samples.tobytes())

    chunks = list(pipe_stream(capture_rate=sr, chunk_size=chunk_size, stream=stream))

    assert len(chunks) == 2
    out = np.concatenate(chunks)
    assert np.allclose(out, samples.astype(np.float32) / 32768.0)


def test_pipe_stream_pads_last_partial_chunk():
    sr, chunk_size = 16000, 0.16
    chunk_frames = int(sr * chunk_size)
    samples = np.ones(chunk_frames + 5, dtype=np.int16)  # 1.x chunks
    stream = _io.BytesIO(samples.tobytes())

    chunks = list(pipe_stream(capture_rate=sr, chunk_size=chunk_size, stream=stream))

    assert len(chunks) == 2
    assert all(len(c) == chunk_frames for c in chunks)
    # the trailing pad samples are zeros
    assert chunks[1][5:].sum() == 0
