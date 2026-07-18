"""Supported process environment for publication runtime-asset entrypoints."""

from __future__ import annotations

import os


SUPPORTED_FINAL_RUNTIME_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def establish_supported_final_runtime_environment() -> None:
    """Remove user/Codex loader influence before final runtime verification."""

    os.environ["PATH"] = SUPPORTED_FINAL_RUNTIME_PATH
    os.unsetenv("LD_LIBRARY_PATH")
    os.environ.pop("LD_LIBRARY_PATH", None)


__all__ = [
    "SUPPORTED_FINAL_RUNTIME_PATH",
    "establish_supported_final_runtime_environment",
]
