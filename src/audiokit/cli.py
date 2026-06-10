"""Small CLI helpers shared by the consuming toolkits' command-line tools."""

import sys
from typing import Callable

import numpy as np

from .errors import AudiokitError


def render_mic_level(chunk, prefix: str = "  ") -> None:
    """Write a live RMS energy bar for ``chunk`` to stdout in place.

    Overwrites the current line with ``\\r``; the caller is responsible for
    clearing or advancing the line before printing other output.
    """
    energy = float(np.sqrt(np.mean(np.asarray(chunk, dtype=np.float64) ** 2)))
    bar = "#" * min(int(energy * 500), 40)
    sys.stdout.write(f"\r{prefix}mic: {bar:<40} {energy:.4f}")
    sys.stdout.flush()


def run_cli(impl: Callable[[], None]) -> None:
    """Run a CLI entrypoint, mapping library exceptions to exit codes.

    ``AudiokitError`` → exit 1; ``KeyboardInterrupt`` → exit 130.
    """
    try:
        impl()
    except AudiokitError:
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
