import functools
import http.server
import tarfile
import threading

import pytest

from audiokit import download_file, safe_tar_members
from audiokit.errors import AudiokitError


@pytest.fixture
def http_server(tmp_path):
    """Serve ``tmp_path`` over HTTP on an ephemeral port. Yields (base_url, dir)."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_fresh(http_server, tmp_path):
    base_url, served_dir = http_server
    payload = b"hello audiokit" * 1000
    (served_dir / "blob.bin").write_bytes(payload)

    dest = tmp_path / "out" / "blob.bin"  # nested -> exercises parent mkdir
    download_file(f"{base_url}/blob.bin", dest, progress=False)

    assert dest.read_bytes() == payload


def test_download_restart_when_server_ignores_range(http_server, tmp_path):
    base_url, served_dir = http_server
    payload = b"0123456789" * 500
    (served_dir / "blob.bin").write_bytes(payload)

    # NOTE: dest must live outside the served directory, otherwise it *is* the
    # file the server returns and the test would download its own partial bytes.
    dest = tmp_path / "dl" / "blob.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"partial")  # pre-existing partial triggers a Range request

    # SimpleHTTPRequestHandler ignores Range (returns 200) -> code restarts cleanly.
    download_file(f"{base_url}/blob.bin", dest, progress=False)

    assert dest.read_bytes() == payload


def test_download_file_url_restarts_partial_destination(tmp_path):
    payload = b"file-url payload" * 100
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dest = tmp_path / "dest.bin"
    dest.write_bytes(b"partial")

    download_file(source.as_uri(), dest, progress=False)

    assert dest.read_bytes() == payload


def test_download_bad_url_raises():
    with pytest.raises(AudiokitError):
        download_file("http://127.0.0.1:1/nope.bin", "/tmp/audiokit_should_not_exist.bin",
                      progress=False)


def test_safe_tar_members_filters_traversal(tmp_path):
    tar_path = tmp_path / "a.tar"
    good = tmp_path / "good.txt"
    good.write_bytes(b"ok")

    with tarfile.open(tar_path, "w") as tf:
        tf.add(good, arcname="good.txt")
        # Inject a path-traversal member without touching the filesystem.
        evil = tarfile.TarInfo(name="../evil.txt")
        evil.size = 0
        tf.addfile(evil)

    with tarfile.open(tar_path, "r") as tf:
        names = [m.name for m in safe_tar_members(tf, tmp_path)]

    assert "good.txt" in names
    assert "../evil.txt" not in names


def test_safe_tar_members_filters_hardlink_target_traversal(tmp_path):
    tar_path = tmp_path / "hardlink.tar"

    with tarfile.open(tar_path, "w") as tf:
        hardlink = tarfile.TarInfo(name="dir/link")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "../outside.txt"
        tf.addfile(hardlink)

    with tarfile.open(tar_path, "r") as tf:
        names = [m.name for m in safe_tar_members(tf, tmp_path)]

    assert "dir/link" not in names
