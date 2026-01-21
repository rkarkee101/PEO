from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )
