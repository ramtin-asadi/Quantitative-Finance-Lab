"""Explicit local cache identities for expensive source-derived research results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(paths, settings) -> str:
    """Hash source file identities and caller-supplied model/transformation settings.

    Cache identity uses resolved paths, byte counts and nanosecond modification
    times. Include a calculation version in settings when changing algorithms.
    This helper creates no directories and writes no files.
    """
    files = []
    for path in paths:
        p = Path(path).resolve()
        info = p.stat()
        files.append((str(p), info.st_size, info.st_mtime_ns))
    payload = json.dumps({"files": files, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
