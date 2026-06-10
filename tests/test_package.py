import audiokit


def test_version_present():
    assert isinstance(audiokit.__version__, str)
    assert audiokit.__version__ == "0.2.0"


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
        "sha256_of",
        "verify_sha256",
        "energy_vad",
        "SNREstimator",
        "estimate_snr",
        "compute_snr",
        "render_mic_level",
        "run_cli",
        "FeatureContract",
        "read_contract",
        "create_manifest",
        "verify_integrity",
        "NumpyStandardScaler",
        "scaler_to_json",
        "scaler_from_json",
        "write_segments_csv",
        "read_segments_csv",
        "segments_to_audformat_index",
    }
    missing = {name for name in expected if not hasattr(audiokit, name)}
    assert not missing, f"missing exports: {missing}"
    assert expected.issubset(set(audiokit.__all__))
