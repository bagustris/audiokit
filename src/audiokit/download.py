"""Model/file download with integrity verification and safe archive extraction.

Seeded from sherox's ``utils.py``. Uses only the standard library
(``urllib``, ``tarfile``, ``hashlib``) so it adds no third-party dependency.
Progress is written to stderr and can be silenced with ``progress=False``.
"""

import hashlib
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .errors import AudiokitError


def verify_sha256(path: "Path | str", expected: str) -> bool:
    """Return ``True`` if the file at *path* hashes to *expected* (hex)."""
    ok, _ = _sha256_check(path, expected)
    return ok


def sha256_of(path: "Path | str") -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    return _sha256_digest(Path(path))


def _sha256_check(path: "Path | str", expected: str) -> tuple[bool, str]:
    """Return ``(matches_expected, computed_digest)`` for *path*."""
    path = Path(path)
    if not path.exists():
        return False, ""
    digest = _sha256_digest(path)
    return digest == expected.lower(), digest


def _sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_file(
    url: str,
    dest: "Path | str",
    *,
    sha256: str = "",
    progress: bool = True,
) -> None:
    """Download ``url`` to ``dest``, resuming a partial file when possible.

    Supports ``http(s)://`` and ``file://`` URLs. If a partial file exists at
    ``dest``, an HTTP ``Range`` request is attempted; servers that ignore it
    (returning ``200`` instead of ``206``) trigger a clean restart.

    When *sha256* is provided the digest of the file is verified after
    download; ``AudiokitError`` is raised on mismatch.

    Raises ``AudiokitError`` on failure.
    """
    dest = Path(dest)
    existing_size = dest.stat().st_size if dest.exists() else 0
    if _existing_file_is_valid(dest, existing_size, sha256):
        return

    try:
        existing_size = _download_response(url, dest, existing_size, progress)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size > 0:
            _restart_download(url, dest, sha256, progress)
            return
        raise AudiokitError(f"Download failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AudiokitError(f"Download failed: {exc}") from exc

    _raise_if_hash_mismatch(dest, sha256)


def _existing_file_is_valid(dest: Path, existing_size: int, sha256: str) -> bool:
    return bool(sha256 and existing_size and verify_sha256(dest, sha256))


def _download_response(url: str, dest: Path, existing_size: int, progress: bool) -> int:
    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    with urllib.request.urlopen(req) as response:  # noqa: S310 - explicit scheme support
        existing_size = _normalise_existing_size(response, existing_size)
        total_size = _total_size(response)
        _write_response(response, dest, existing_size, total_size, progress)
        _finish_progress(progress, total_size)
    return existing_size


def _normalise_existing_size(response, existing_size: int) -> int:  # noqa: ANN001
    status = getattr(response, "status", None)
    if existing_size > 0 and (status is None or status != 206):
        return 0
    return existing_size


def _total_size(response) -> int:  # noqa: ANN001
    if "Content-Range" in response.headers:
        return int(response.headers["Content-Range"].split("/")[-1])
    return int(response.headers.get("Content-Length", 0))


def _write_response(response, dest: Path, existing_size: int, total_size: int, progress: bool) -> None:  # noqa: ANN001
    mode = "ab" if existing_size > 0 else "wb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open(mode) as f:
        downloaded = existing_size
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            _write_progress(downloaded, total_size, progress)


def _write_progress(downloaded: int, total_size: int, progress: bool) -> None:
    if not progress or total_size <= 0:
        return
    pct = min(100, downloaded * 100 // total_size)
    sys.stderr.write(f"\r  {pct}%")
    sys.stderr.flush()


def _finish_progress(progress: bool, total_size: int) -> None:
    if progress and total_size > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _restart_download(url: str, dest: Path, sha256: str, progress: bool) -> None:
    dest.unlink(missing_ok=True)
    download_file(url, dest, sha256=sha256, progress=progress)


def _raise_if_hash_mismatch(dest: Path, sha256: str) -> None:
    if not sha256:
        return
    ok, got = _sha256_check(dest, sha256)
    if ok:
        return
    raise AudiokitError(
        f"SHA-256 mismatch for {dest}: expected {sha256}, got {got}"
    )

def safe_tar_members(tf: tarfile.TarFile, dest_dir: "Path | str") -> Iterator[tarfile.TarInfo]:
    """Yield only members safe to extract into ``dest_dir``.

    Emulates the guarantees of ``filter="data"`` on Python < 3.12:
    rejects path traversal, device/special files, and links whose target
    escapes ``dest_dir``.
    """
    dest_dir = Path(dest_dir)
    dest_resolved = dest_dir.resolve()

    def _escapes(path: Path) -> bool:
        try:
            path.resolve().relative_to(dest_resolved)
        except ValueError:
            return True
        return False

    for member in tf.getmembers():
        if member.isdev():
            continue
        if _escapes(dest_dir / member.name):
            continue
        if member.issym():
            link_base = (dest_dir / member.name).parent
            if _escapes(link_base / member.linkname):
                continue
        elif member.islnk():
            if _escapes(dest_dir / member.linkname):
                continue
        yield member
