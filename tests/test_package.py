import audiokit


def test_version_present():
    assert isinstance(audiokit.__version__, str)
    assert audiokit.__version__ == "0.1.0"


def test_public_api_exported():
    expected = {
        "AudiokitError",
        "read_wav",
        "wav_duration",
        "mic_stream",
        "pipe_stream",
        "resample",
        "download_file",
        "safe_tar_members",
        "energy_vad",
        "SNREstimator",
        "estimate_snr",
        "compute_snr",
        "render_mic_level",
        "run_cli",
    }
    missing = {name for name in expected if not hasattr(audiokit, name)}
    assert not missing, f"missing exports: {missing}"
    assert expected.issubset(set(audiokit.__all__))
