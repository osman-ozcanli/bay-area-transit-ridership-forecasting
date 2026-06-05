"""Central configuration loader for the BART project.

Loads `config.yaml` once and resolves environment-specific (local vs kaggle)
paths so the rest of the codebase never hardcodes any path or parameter.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the `src` directory that holds this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration.

    Args:
        config_path: Optional override path to a config file. Defaults to the
            project-root `config.yaml`.

    Returns:
        The parsed configuration as a dictionary (sozluk).
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Environment override: `BART_ENV` env-var wins over the file value.
    # Kaggle'da `os.environ["BART_ENV"]="kaggle"` demek config.yaml'i elle
    # duzenlemekten kurtarir.
    env_override = os.environ.get("BART_ENV")
    if env_override:
        cfg["environment"] = env_override
    return cfg


def get_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Resolve the active environment's paths into absolute `Path` objects.

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Mapping of path keys (raw_dir, sample_dir, ...) to absolute paths.
    """
    cfg = config or load_config()
    env = cfg["environment"]
    raw_paths = cfg["paths"][env]

    resolved: dict[str, Path] = {}
    for key, value in raw_paths.items():
        p = Path(value)
        # Local relative paths are anchored to the project root; absolute
        # (kaggle) paths are kept as-is.
        resolved[key] = p if p.is_absolute() else PROJECT_ROOT / p
    return resolved


if __name__ == "__main__":
    # Quick smoke test (hizli kontrol): print resolved config.
    _cfg = load_config()
    print(f"environment: {_cfg['environment']}")
    for _k, _v in get_paths(_cfg).items():
        print(f"  {_k}: {_v}")
