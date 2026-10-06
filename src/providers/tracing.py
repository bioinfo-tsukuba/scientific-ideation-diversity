"""Langfuse tracing for LLM generation calls.

Reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and
LANGFUSE_BASE_URL (or LANGFUSE_HOST) from the environment.
All three are required; a missing variable raises RuntimeError.
"""

from __future__ import annotations

import os
import threading

from langfuse import Langfuse

_langfuse: Langfuse | None = None
_langfuse_lock = threading.Lock()
_initialized = False


def get_langfuse() -> Langfuse:
    """Return the global Langfuse client.

    Thread-safe; the client is created at most once.
    Raises RuntimeError if required environment variables are missing.
    """
    global _langfuse, _initialized

    if _initialized:
        assert _langfuse is not None
        return _langfuse

    with _langfuse_lock:
        if _initialized:
            assert _langfuse is not None
            return _langfuse

        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

        missing = [
            name
            for name, val in [
                ("LANGFUSE_SECRET_KEY", secret_key),
                ("LANGFUSE_PUBLIC_KEY", public_key),
                ("LANGFUSE_BASE_URL or LANGFUSE_HOST", host),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Langfuse tracing is required but the following environment variables are missing: {', '.join(missing)}"
            )

        _langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )

        _initialized = True
        return _langfuse


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    if _langfuse is not None:
        _langfuse.flush()
