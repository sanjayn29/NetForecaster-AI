"""
src/utils/logger.py
-------------------
Centralized logging setup for NetForecaster AI.
Call ``get_logger(__name__)`` in any module.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name — use ``__name__`` in each module.
    level : str
        Logging level string: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str or Path, optional
        If provided, also write logs to this file (append mode).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Prevent log records from propagating to the root logger
    logger.propagate = False

    return logger


def setup_root_logger(level: str = "INFO", log_file: Optional[str | Path] = None) -> None:
    """Configure the root logger (useful for script entry points).

    Parameters
    ----------
    level : str
        Logging level.
    log_file : str or Path, optional
        Optional file path for persistent log output.
    """
    get_logger("netforecaster", level=level, log_file=log_file)
