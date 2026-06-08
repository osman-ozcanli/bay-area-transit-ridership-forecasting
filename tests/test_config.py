"""Smoke tests for the configuration loader (`src/config.py`)."""
from __future__ import annotations

from typing import Any

from src.config import get_paths, load_config


def test_load_config_has_required_keys(cfg: dict[str, Any]) -> None:
    """Config exposes every top-level key the pipeline depends on."""
    for key in ["paths", "files", "target", "datetime_col",
                "features", "model", "eda"]:
        assert key in cfg, f"config.yaml '{key}' anahtarini icermeli"


def test_get_paths_resolves_absolute(cfg: dict[str, Any]) -> None:
    """Resolved paths are absolute (anchored to project root)."""
    paths = get_paths(cfg)
    assert paths["sample_dir"].is_absolute()
    assert paths["models_dir"].is_absolute()


def test_bart_env_override(monkeypatch) -> None:
    """`BART_ENV` env-var overrides the file's environment value."""
    load_config.cache_clear()  # lru_cache: taze yukleme zorla
    monkeypatch.setenv("BART_ENV", "kaggle")
    try:
        assert load_config()["environment"] == "kaggle"
    finally:
        load_config.cache_clear()  # sonraki testler icin yereli geri ver
