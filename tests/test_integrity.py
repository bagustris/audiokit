"""Tests for audiokit.integrity (Priority 2 — model integrity verification)."""

import json

import pytest

from audiokit import verify_sha256, sha256_of, create_manifest, verify_integrity
from audiokit.errors import AudiokitError


def test_verify_sha256_ok(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello audiokit")
    # Compute the hash
    h = sha256_of(f)
    assert verify_sha256(f, h)


def test_verify_sha256_mismatch(tmp_path):
    f = tmp_path / "test.bin"
    f.write_bytes(b"hello audiokit")
    assert not verify_sha256(f, "0" * 64)


def test_verify_sha256_nonexistent(tmp_path):
    assert not verify_sha256(tmp_path / "nope.bin", "0" * 64)


def test_sha256_of_consistent(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"x" * 10000)
    h1 = sha256_of(f)
    h2 = sha256_of(f)
    assert h1 == h2


def test_create_manifest_single_file(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"fake onnx content")
    manifest = create_manifest(tmp_path, producing_tool="audiokit", producing_tool_version="0.2.0")
    assert "model.onnx" in manifest["files"]
    assert manifest["producing_tool"] == "audiokit"
    assert len(manifest["files"]["model.onnx"]["sha256"]) == 64


def test_create_manifest_skips_hidden(tmp_path):
    (tmp_path / ".hidden").write_bytes(b"secret")
    (tmp_path / "visible.bin").write_bytes(b"ok")
    manifest = create_manifest(tmp_path)
    assert ".hidden" not in manifest["files"]
    assert "visible.bin" in manifest["files"]


def test_verify_integrity_ok(tmp_path):
    f = tmp_path / "good.txt"
    f.write_bytes(b"hello")
    h = sha256_of(f)
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump({
            "files": {
                "good.txt": {"sha256": h, "source_url": "", "format": "txt", "size": 5}
            }
        }, mf)
    assert verify_integrity(manifest_path, tmp_path)


def test_verify_integrity_missing_file_raises(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump({
            "files": {
                "missing.txt": {"sha256": "0" * 64, "source_url": "", "format": "txt", "size": 5}
            }
        }, mf)
    with pytest.raises(AudiokitError, match="MISSING"):
        verify_integrity(manifest_path, tmp_path, strict=True)
