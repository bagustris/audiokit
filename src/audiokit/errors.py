"""Shared exception type for audiokit.

Kept in its own module so submodules can import it without a circular
dependency on the package ``__init__``.
"""


class AudiokitError(Exception):
    """Base error raised by audiokit for recoverable, user-facing failures."""
