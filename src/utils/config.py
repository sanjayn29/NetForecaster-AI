"""
src/utils/config.py
-------------------
Central configuration loader for NetForecaster AI.
Reads config.yaml from the project root and exposes it
as a nested attribute-accessible object.
"""

import os
import yaml
from pathlib import Path
from typing import Any


# ── Locate project root ────────────────────────────────────────────────────
# This file lives at  <project_root>/src/utils/config.py
# so root = 3 parents up from this file.
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


class _AttrDict(dict):
    """Dictionary subclass that exposes keys as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'") from None
        if isinstance(value, dict):
            return _AttrDict(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path | None = None) -> _AttrDict:
    """Load and return the project config as an attribute-accessible dict.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file.  Defaults to ``config.yaml`` at the
        project root.

    Returns
    -------
    _AttrDict
        Nested configuration object.  Values are accessible as attributes
        (``cfg.data.raw_dir``) or as dict keys (``cfg['data']['raw_dir']``).
    """
    config_path = Path(path) if path else CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected location: {CONFIG_FILE}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _to_attr(obj: Any) -> Any:
        if isinstance(obj, dict):
            return _AttrDict({k: _to_attr(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_attr(item) for item in obj]
        return obj

    return _to_attr(raw)


# Module-level singleton — import and use directly:
#   from src.utils.config import CFG
CFG = load_config()
