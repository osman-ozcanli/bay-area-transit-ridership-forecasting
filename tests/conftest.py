"""Shared pytest fixtures for the BART project test-suite.

Loads the local sample once per session (use_sample=true in config.yaml) and
exposes the raw + feature-built frames so individual tests stay fast.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config import load_config
from src.data.load import load_dataset
from src.features.build_features import build_features


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    """The project configuration (local environment, sample data)."""
    return load_config()


@pytest.fixture(scope="session")
def raw_df(cfg: dict[str, Any]) -> pd.DataFrame:
    """Raw merged ridership sample (output of `load_dataset`)."""
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def feat_df(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Feature-built frame (built once, reused across feature tests)."""
    return build_features(raw_df, cfg)
