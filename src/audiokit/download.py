"""Model/file download and safe archive extraction.

Seeded from sherox's ``utils.py``. Uses only the standard library
(``urllib``, ``tarfile``) so it adds no third-party dependency. Progress is
written to stderr and can be silenced with ``progress=False``.
"""

import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .errors import AudiokitError


def download_file(url: str, dest: "Path | str", progress: bool = True) -> None:
    """Download ``url`` to ``dest``, resuming a partial file when possible.

    Supports ``http(s)://`` and ``file://`` URLs. If a partial file exists at
    ``dest``, an HTTP ``Range`` request is attempted; servers that ignore it
    (returning ``200`` instead of ``206``) trigger a clean restart.

    Raises ``AudiokitError`` on failure.
    """
    dest = Path(dest)

    existing_size = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    try:
        with urllib.request.urlopen(req) as response:  # noqa: S310 - explicit scheme support
            status = getattr(response, "status", None)
            # Server ignored our Range request (full 200 instead of 206 partial).
            # Non-HTTP handlers such as file:// also ignore Range and do not
            # expose a status code, so restart rather than appending a full
            # response to the partial file.
            if existing_size > 0 and (status is None or status != 206):
                # Restart: existing_size = 0 below selects "wb", which truncates.
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
            # Requested range not satisfiable — discard partial and restart.
            dest.unlink(missing_ok=True)
            download_file(url, dest, progress=progress)
            return
        raise AudiokitError(f"Download failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AudiokitError(f"Download failed: {exc}") from exc
    if progress and total_size > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()


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
