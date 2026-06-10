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
    path = Path(path)
    if not path.exists():
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest() == expected.lower()


def sha256_of(path: "Path | str") -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
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

    # If the existing file already satisfies the hash, skip download.
    if sha256 and existing_size and verify_sha256(dest, sha256):
        return

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    try:
        with urllib.request.urlopen(req) as response:  # noqa: S310 - explicit scheme support
            status = getattr(response, "status", None)
            if existing_size > 0 and (status is None or status != 206):
                existing_size = 0

            if "Content-Range" in response.headers:
                total_size = int(response.headers["Content-Range"].split("/")[-1])
            else:
                total_size = int(response.headers.get("Content-Length", 0))

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
                    if progress and total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        sys.stderr.write(f"\r  {pct}%")
                        sys.stderr.flush()
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size > 0:
            dest.unlink(missing_ok=True)
            download_file(url, dest, sha256=sha256, progress=progress)
            return
        raise AudiokitError(f"Download failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AudiokitError(f"Download failed: {exc}") from exc
    if progress and total_size > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if sha256 and not verify_sha256(dest, sha256):
        got = sha256_of(dest)
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
