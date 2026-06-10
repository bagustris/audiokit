# audiokit

**Purpose-neutral audio plumbing** — the small, dependency-light building blocks
(audio I/O, resampling, energy-VAD, SNR, model download, integrity
verification, feature contracts, segment CSV export, scaler JSON) that
[`coughkit`](../coughkit), [`sherox`](../sherox), and [`nkululeko`](../nkululeko)
each currently re-implement on their own.

This package is the concrete first step from
[`cross_repo_analysis.md`](../cross_repo_analysis.md) §4(B): extract **only** the
mechanical, task-neutral code into one tested library, so the other repos can
later depend on it instead of duplicating it. **No other repository has been
modified** — audiokit stands alone and passes its own test suite first.

> Scope guardrail: audiokit holds *plumbing only*. Task logic (cough classifier,
> sherpa-onnx/NeMo engines, nkululeko feature extractors) stays in its home repo.
> Heavy backends (sherpa-onnx, torch, transformers, NeMo) are deliberately **not**
> dependencies here.

## Install

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'        # dev = pytest
uv pip install -e '.[dev,mic]'    # add 'mic' for live microphone capture (sounddevice)
```

Core dependencies are only `numpy`, `scipy`, and `soundfile`. `sounddevice` is an
optional `[mic]` extra because it needs a system PortAudio library and only
`mic_stream` uses it (lazily imported with a clear error if absent).

## Public API

```python
from audiokit import (
    read_wav, wav_duration, mic_stream, pipe_stream, resample,   # io
    download_file, safe_tar_members, sha256_of, verify_sha256,    # download
    verify_integrity, create_manifest,                            # integrity
    FeatureContract, read_contract,                               # contracts
    write_segments_csv, read_segments_csv,                        # segment CSV
    NumpyStandardScaler, scaler_from_json, scaler_to_json,         # scalers
    energy_vad,                                                   # vad
    SNREstimator, estimate_snr, compute_snr,                      # snr
    render_mic_level, run_cli,                                    # cli
    AudiokitError,                                                # errors
)
```

| Module | Function/Class | What it does | Seeded from |
|--------|----------------|--------------|-------------|
| `audiokit.io` | `read_wav` | Stream a **mono** file as float32 chunks; resamples once (no block artifacts) when rates differ | sherox `audio.py` |
| | `wav_duration` | Duration in seconds without decoding | sherox |
| | `resample` | Polyphase Kaiser (β=14, ~90 dB) resample; linear fallback | sherox |
| | `mic_stream` | Callback+queue microphone capture (needs `[mic]`) | sherox |
| | `pipe_stream` | Raw 16-bit PCM from a stream/stdin → float32 chunks (injectable for tests) | sherox |
| `audiokit.download` | `download_file` | `http(s)://`/`file://` download with resume + optional SHA-256 verification | sherox `utils.py` + Priority 2 |
| | `sha256_of` / `verify_sha256` | File digest helpers used by model integrity manifests | Priority 2 |
| | `safe_tar_members` | Path-traversal-safe tar member filter (`filter="data"` for Py<3.12) | sherox |
| `audiokit.integrity` | `create_manifest` / `verify_integrity` | Shared manifest format for model/fixture integrity | Priority 2 / §7.5 |
| `audiokit.contract` | `FeatureContract` / `read_contract` | Machine-readable feature-vector contract (e.g. coughkit 68 features) | Priority 3 / §4.3 |
| `audiokit.segment` | `write_segments_csv` / `read_segments_csv` / `segments_to_audformat_index` | Audformat-compatible segment CSV bridge for sherox/coughkit → nkululeko | Priority 3 / §5.3 |
| `audiokit.scaler` | `NumpyStandardScaler` / `scaler_to_json` / `scaler_from_json` | Portable JSON scaler export/load without sklearn at inference | coughkit pattern / §6.5 |
| `audiokit.vad` | `energy_vad` | Hysteresis-comparator event detector for **any** high-energy events (coughs, claps, impulses); returns sample-index segments + boolean mask | coughkit `segment_cough` |
| `audiokit.snr` | `SNREstimator` / `estimate_snr` | Percentile log-energy SNR (general; no segmentation needed) | nkululeko `estimate_snr.py` |
| | `compute_snr` | Mask-based RMS(event)/RMS(background) SNR via `energy_vad` | coughkit `compute_SNR` |
| `audiokit.cli` | `render_mic_level` | In-place RMS level bar for mic feedback | sherox |
| | `run_cli` | Map `AudiokitError`→exit 1, `KeyboardInterrupt`→exit 130 | sherox |

## Examples

```python
import numpy as np
from audiokit import read_wav, energy_vad, estimate_snr, compute_snr

# 1) Stream a mono WAV as 100 ms float32 chunks at 16 kHz
chunks = list(read_wav("audio.wav", target_sr=16000, chunk_size=0.1))
signal = np.concatenate(chunks)

# 2) Detect high-energy events (coughs, claps, …) — pure NumPy, no model
segments, mask = energy_vad(signal, 16000)   # segments: [(start_idx, end_idx), ...]

# 3) Estimate SNR two ways
print(estimate_snr(signal, 16000))           # percentile log-energy method
print(compute_snr(signal, 16000))            # event-vs-background mask method
```

```python
# Download a model with resume support + SHA-256 verification
from audiokit import download_file
download_file(
    "https://example.com/model.onnx",
    "models/model.onnx",
    sha256="<expected-sha256-hex>",
)
```

```python
# Write a segment bridge CSV (sherox/coughkit -> nkululeko)
from audiokit import write_segments_csv
write_segments_csv("segments.csv", [
    {"source_file": "meeting.wav", "start": 1.23, "end": 2.34, "segment_file": "seg_0000.wav"}
])
```

```python
# Define and validate a model feature contract
from audiokit import FeatureContract
contract = FeatureContract(n_features=3, feature_names=["mfcc0", "mfcc1", "zcr"])
contract.validate_model_feature_names(["mfcc0", "mfcc1", "zcr"])
```

## Design notes & deliberate divergences from the seed code

- **No `rich`.** sherox prints status with `rich`; audiokit uses the stdlib
  `logging` module and raises `AudiokitError`, keeping the dependency surface to
  numpy/scipy/soundfile so it imports instantly everywhere.
- **`download_file` progress goes to stderr** and can be silenced with
  `progress=False` (libraries shouldn't pollute stdout).
- **`pipe_stream` accepts an injectable `stream`**, making it unit-testable
  without monkeypatching `sys.stdin`.
- **`energy_vad` returns sample-index `(start, end)` pairs + a mask** instead of
  copying sub-arrays, which is more reusable than the original `segment_cough`.
- **`SNREstimator` drops the matplotlib plotting method** — out of scope for a
  plumbing library; lazy-imports `scipy.signal.windows.hamming` only when used.
- Excluded on purpose: `denoise_gen` (needs `noisereduce`) and any ONNX/torch VAD
  (needs sherpa-onnx/torch). Those belong in the consuming repos.

## Tests

```bash
uv run --with pytest pytest -q       # 78 tests, all offline & deterministic
```

`tests/` (all self-contained — no network, no microphone, no external models):

| File | Covers |
|------|--------|
| `test_io.py` | round-trip chunked read, resample-on-read, duration, mono enforcement, `resample`, `pipe_stream` EOF + padding |
| `test_download.py` | fresh HTTP download, restart when server ignores `Range`, bad-URL error, tar traversal filtering, SHA-256 verification |
| `test_integrity.py` | manifest creation and integrity verification |
| `test_contract.py` | feature-contract read/write and model-feature-name validation |
| `test_segment.py` | segment CSV read/write and audformat-style index conversion |
| `test_scaler.py` | `NumpyStandardScaler` and scaler JSON export/load |
| `test_testing.py` | shared synthetic audio fixture factories |
| `test_vad.py` | single-burst detection, empty signal, pure silence, segment/mask consistency |
| `test_snr.py` | high-SNR positive, silence→0, short-signal guard, event-over-background, no-event→0 |
| `test_package.py` | version string and complete public-API export |

Network/microphone-bound paths (`mic_stream`, live capture) are intentionally not
unit-tested; `download_file` is exercised against a local threaded HTTP server.

## Status & next steps

- [x] Standalone, installable `audiokit` package (src-layout, `uv` + `.venv`).
- [x] Core modules (`io`, `download`, `vad`, `snr`, `cli`) + `errors`.
- [x] Priority 2 shared infrastructure: SHA-256 download verification, integrity manifests, shared synthetic-audio test factories.
- [x] Priority 3 bridge infrastructure: feature contracts, audformat-compatible segment CSV, scaler JSON serialisation/runtime loader.
- [x] 78 passing tests; deterministic and offline.
- [ ] Integrate **coughkit** with audiokit: delegate `segment_cough` / `compute_SNR`
  to `energy_vad` / `compute_snr`, then add feature-contract-aware model loading.
- [ ] Integrate **sherox** with audiokit: delegate `audio._resample` and
  `utils.safe_tar_members`, then add model integrity checks and segment CSV export.
- [ ] Integrate **nkululeko** with audiokit: reuse `SNREstimator`, add external
  segment CSV ingestion, and use scaler JSON helpers for portable bundles.

Consumers should declare `audiokit` in `[project.dependencies]` and, for local
workspace development, resolve it via `[tool.uv.sources] audiokit = { path =
"../audiokit", editable = true }`. See [`cross_repo_analysis.md`](../cross_repo_analysis.md)
§9 for the full roadmap.
