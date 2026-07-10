"""Shared configuration paths for hedonic experiments."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DBLP_DIR = Path("~/Databases/Hedonic/Networks/DBLP")
DEFAULT_SYNTHETIC_DIR = Path(
    "~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020"
)

DBLP_DIR = Path(os.getenv("HEDONIC_DBLP_DIR", str(DEFAULT_DBLP_DIR)))
SYNTHETIC_DIR = Path(os.getenv("HEDONIC_SYNTHETIC_DIR", str(DEFAULT_SYNTHETIC_DIR)))


def reload_paths() -> tuple[Path, Path]:
    """Re-read path env vars (useful in tests after mutating the environment)."""
    global DBLP_DIR, SYNTHETIC_DIR
    DBLP_DIR = Path(os.getenv("HEDONIC_DBLP_DIR", str(DEFAULT_DBLP_DIR)))
    SYNTHETIC_DIR = Path(os.getenv("HEDONIC_SYNTHETIC_DIR", str(DEFAULT_SYNTHETIC_DIR)))
    return DBLP_DIR, SYNTHETIC_DIR
