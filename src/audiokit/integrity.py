"""Model/fixture integrity verification.

Provides a lightweight manifest format (JSON mapping model_path → sha256_hex)
that can be bundled with a release and verified at runtime.

Seeded from the ``INTEGRITY.md`` proposal in the cross-repo analysis (§5.1).
Uses standard library only (``json``, ``hashlib``).
"""

import json
from pathlib import Path
from typing import Dict, Optional

from .download import sha256_of, verify_sha256
from .errors import AudiokitError

# ── Manifest schema ──────────────────────────────────────────────────────────
# A manifest is a JSON file with this structure:
#
# {
#   "$schema": "https://raw.githubusercontent.com/bagustris/audiokit/main/integrity-schema.json",
#   "producing_tool": "sherox 0.8.0",
#   "producing_tool_version": "0.8.0",
#   "created_at": "2026-06-10T00:00:00",
#   "files": {
#     "models/silero_vad.onnx": {
#       "sha256": "abc123...",
#       "source_url": "https://github.com/k2-fsa/...",
#       "format": "onnx",
#       "size": 1234567
#     },
#     "models/cough_classifier.json": {
#       "sha256": "def456...",
#       "source_url": "",
#       "format": "xgboost-json",
#       "size": 12345
#     }
#   },
#   "feature_contract": {
#     "version": "0.2.0",
#     "n_features": 68,
#     "groups": { ... }
#   }
# }


def create_manifest(
    root_dir: "Path | str",
    *,
    producing_tool: str = "audiokit",
    producing_tool_version: str = "",
) -> dict:
    """Walk *root_dir*, compute SHA-256 for every file, return a manifest dict.

    The returned dict can be serialised with ``json.dump`` to create a
    ``manifest.json`` that ``verify_integrity`` later consumes.
    """
    root = Path(root_dir)
    files: Dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel = str(path.relative_to(root))
            files[rel] = {
                "sha256": sha256_of(path),
                "source_url": "",
                "format": _guess_format(path),
                "size": path.stat().st_size,
            }

    return {
        "$schema": "",
        "producing_tool": producing_tool,
        "producing_tool_version": producing_tool_version,
        "files": files,
    }


def verify_integrity(
    manifest_path: "Path | str",
    root_dir: "Path | str",
    *,
    strict: bool = True,
) -> bool:
    """Verify every file listed in *manifest_path* against *root_dir*.

    Returns ``True`` if every file exists and its SHA-256 matches.
    Raises ``AudiokitError`` listing all mismatches when *strict* is True
    (default); otherwise returns ``False`` on the first mismatch.
    """
    manifest_path = Path(manifest_path)
    root = Path(root_dir)

    with manifest_path.open() as f:
        manifest = json.load(f)

    errors: list[str] = []
    for rel_path, info in manifest.get("files", {}).items():
        full_path = root / rel_path
        if not full_path.exists():
            errors.append(f"MISSING: {rel_path}")
            if not strict:
                break
            continue

        expected = info.get("sha256", "")
        if expected and not verify_sha256(full_path, expected):
            got = sha256_of(full_path)
            errors.append(
                f"SHA-256 MISMATCH: {rel_path} "
                f"(expected {expected[:16]}..., got {got[:16]}...)"
            )
            if not strict:
                break

    if errors:
        if strict:
            raise AudiokitError(
                "Integrity check failed:\n  " + "\n  ".join(errors)
            )
        return False
    return True


def _guess_format(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".onnx": "onnx",
        ".json": "xgboost-json",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".pt": "torch",
        ".bin": "binary",
        ".wav": "wav",
        ".flac": "flac",
        ".mp3": "mp3",
    }.get(ext, "unknown")


__all__ = [
    "create_manifest",
    "verify_integrity",
]
