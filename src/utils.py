from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  –  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def seconds_to_laptime(s: float) -> str:
    m   = int(s // 60)
    rem = s - m * 60
    return f"{m}:{rem:06.3f}"


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
