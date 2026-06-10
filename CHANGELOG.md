# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-10

### Added
- Priority 2 infrastructure: SHA-256 verification for downloads, `sha256_of`,
  `verify_sha256`, `create_manifest`, and `verify_integrity`.
- Priority 2 shared synthetic audio factories in `audiokit.testing`.
- Priority 3 feature contracts via `FeatureContract` and `read_contract`.
- Priority 3 audformat-compatible segment CSV helpers: `write_segments_csv`,
  `read_segments_csv`, and `segments_to_audformat_index`.
- Priority 3 portable scaler helpers: `NumpyStandardScaler`, `scaler_to_json`,
  and `scaler_from_json`.
- Additional tests covering integrity manifests, feature contracts, segment CSV,
  scaler JSON, shared fixtures, and SHA-256 downloads.

### Changed
- `download_file` now accepts optional `sha256=` and skips re-download when an
  existing file already matches the expected digest.
- Public version bumped to `0.2.0`.

## [0.1.0] - 2026-06-10

### Added
- Initial installable `src`-layout `audiokit` package for shared,
  purpose-neutral audio plumbing.
- Audio I/O helpers: `read_wav`, `wav_duration`, `resample`, `mic_stream`, and
  `pipe_stream`.
- Download helpers: resumable `download_file` for `http(s)://` and `file://`
  URLs, plus `safe_tar_members` for path-traversal-safe tar extraction.
- Energy-based event detection with `energy_vad`, returning sample-index
  segments and a boolean mask.
- SNR utilities: `SNREstimator`, `estimate_snr`, and `compute_snr`.
- CLI support helpers: `render_mic_level`, `run_cli`, and the shared
  `AudiokitError` exception.
- Offline regression tests covering audio I/O, downloads, tar filtering, VAD,
  SNR, and public package exports.

### Fixed
- `energy_vad` compares signal power to `RMS^2` thresholds, keeping detection
  behavior scale-invariant.
- `download_file` restarts `file://` downloads cleanly when the destination is a
  partial file.
- `safe_tar_members` rejects hardlinks whose resolved targets escape the
  extraction root.
