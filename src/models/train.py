"""Model training for the BART ridership project.

Key methodological choices (analiz'deki kritik düzeltmeler):
    * **Temporal split** (analiz 3.2): 2016 -> train, 2017 -> test. Zaman
      serisinde random split future-leakage yaratir; yil bazli ayrim dogru.
    * **TimeSeriesSplit CV** (analiz 3.9): tek split yerine zaman-sirali CV ile
      varyans olculur.
    * **Categorical feature** (analiz 3.3): istasyonlar LightGBM'e `category`
      olarak verilir, cat.codes ordinal varsayimi yok.
    * **Model kayit** (analiz 3.8): final model `models/` altina yazilir.

All parameters come from `config.yaml`.
"""
from __future__ import annotations

import copy
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import get_paths, load_config


def temporal_split(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by calendar year: train_year -> train, test_year -> test.

    Args:
        df: Time-sorted, feature-built DataFrame.
        config: Loaded configuration.

    Returns:
        Tuple of (train_df, test_df), each time-sorted.
    """
    dt_col = config["datetime_col"]
    ts = config["temporal_split"]
    years = df[dt_col].dt.year
    train = df[years == ts["train_year"]].sort_values(dt_col).reset_index(drop=True)
    test = df[years == ts["test_year"]].sort_values(dt_col).reset_index(drop=True)
    return train, test


def get_feature_columns(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Build the feature and categorical column lists from config.

    Args:
        config: Loaded configuration.

    Returns:
        Tuple of (all_feature_columns, categorical_columns). Lag/rolling
        columns are derived from the feature settings.
    """
    fcfg = config["features"]
    target = config["target"]
    rw = fcfg["rolling_window"]
    lag_cols = [f"{target}_lag_{lag}" for lag in fcfg["lags"]]
    lag_cols.append(f"{target}_roll_mean_{rw}")

    categorical = list(fcfg["categorical"])
    features = list(fcfg["numeric"]) + categorical + lag_cols
    return features, categorical


def _build_params(config: dict[str, Any]) -> dict[str, Any]:
    """Assemble LightGBM params, injecting the device (cpu/gpu).

    Args:
        config: Loaded configuration.

    Returns:
        LightGBM parameter dict.
    """
    params = copy.deepcopy(config["model"]["params"])
    params["device_type"] = config["model"].get("device", "cpu")
    return params


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE and R^2.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict with mae, rmse, r2.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _lgb_train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    valid_set: lgb.Dataset,
    config: dict[str, Any],
) -> lgb.Booster:
    """Train a LightGBM booster with early stopping; GPU->CPU fallback.

    Args:
        params: LightGBM parameters (may request gpu).
        train_set: Training Dataset.
        valid_set: Validation Dataset for early stopping.
        config: Loaded configuration (rounds / logging).

    Returns:
        Trained Booster.
    """
    mcfg = config["model"]
    callbacks = [lgb.early_stopping(mcfg["early_stopping_rounds"], verbose=False)]
    if mcfg.get("log_period", 0):
        callbacks.append(lgb.log_evaluation(period=mcfg["log_period"]))

    try:
        return lgb.train(
            params,
            train_set,
            num_boost_round=mcfg["num_boost_round"],
            valid_sets=[valid_set],
            callbacks=callbacks,
        )
    except Exception as exc:  # noqa: BLE001 - GPU build yoksa CPU'ya dus
        if params.get("device_type") == "gpu":
            print(f"[WARN] GPU egitimi basarisiz, CPU'ya geciliyor: {exc}")
            params = {**params, "device_type": "cpu"}
            return lgb.train(
                params,
                train_set,
                num_boost_round=mcfg["num_boost_round"],
                valid_sets=[valid_set],
                callbacks=callbacks,
            )
        raise


def run_cv(
    train_df: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    """Time-aware cross-validation (TimeSeriesSplit) for MAE variance.

    Args:
        train_df: Time-sorted training DataFrame.
        config: Loaded configuration.

    Returns:
        Dict with per-fold MAE list, mean and std.
    """
    features, categorical = get_feature_columns(config)
    target = config["target"]
    params = _build_params(config)
    tscv = TimeSeriesSplit(n_splits=config["model"]["cv_splits"])

    x_all, y_all = train_df[features], train_df[target]
    fold_maes: list[float] = []
    for tr_idx, va_idx in tscv.split(x_all):
        tr = lgb.Dataset(
            x_all.iloc[tr_idx], label=y_all.iloc[tr_idx],
            categorical_feature=categorical, free_raw_data=False,
        )
        va = lgb.Dataset(
            x_all.iloc[va_idx], label=y_all.iloc[va_idx],
            reference=tr, categorical_feature=categorical, free_raw_data=False,
        )
        model = _lgb_train(params, tr, va, config)
        pred = model.predict(x_all.iloc[va_idx], num_iteration=model.best_iteration)
        fold_maes.append(float(mean_absolute_error(y_all.iloc[va_idx], pred)))

    return {
        "fold_maes": fold_maes,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
    }


def train_model(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Temporal-split train + holdout evaluation (+ optional CV).

    Args:
        df: Feature-built DataFrame (output of `build_features`).
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Tuple of (trained model, results dict with metrics / cv / split sizes).
    """
    cfg = config or load_config()
    features, categorical = get_feature_columns(cfg)
    target = cfg["target"]

    train_df, test_df = temporal_split(df, cfg)
    x_train, y_train = train_df[features], train_df[target]
    x_test, y_test = test_df[features], test_df[target]

    params = _build_params(cfg)
    train_set = lgb.Dataset(
        x_train, label=y_train, categorical_feature=categorical, free_raw_data=False
    )
    valid_set = lgb.Dataset(
        x_test, label=y_test, reference=train_set,
        categorical_feature=categorical, free_raw_data=False,
    )

    model = _lgb_train(params, train_set, valid_set, cfg)
    preds = model.predict(x_test, num_iteration=model.best_iteration)
    metrics = _evaluate(y_test, preds)

    results: dict[str, Any] = {
        "metrics": metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best_iteration": int(model.best_iteration or 0),
        "device": params.get("device_type"),
        "features": features,
    }
    if cfg["model"].get("run_cv"):
        results["cv"] = run_cv(train_df, cfg)
    return model, results


def save_model(model: lgb.Booster, config: dict[str, Any] | None = None) -> str:
    """Persist the trained model to the configured models directory.

    Args:
        model: Trained LightGBM Booster.
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        The path the model was written to.
    """
    cfg = config or load_config()
    models_dir = get_paths(cfg)["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / cfg["files"]["final_model"]
    model.save_model(str(out_path))
    return str(out_path)


def _report(results: dict[str, Any]) -> None:
    """Pretty-print training results."""
    m = results["metrics"]
    print("=== Training results ===")
    print(f"device        : {results['device']}")
    print(f"train / test  : {results['n_train']:,} / {results['n_test']:,}")
    print(f"best_iteration: {results['best_iteration']}")
    print(f"holdout  MAE  : {m['mae']:.4f}")
    print(f"holdout  RMSE : {m['rmse']:.4f}")
    print(f"holdout  R2   : {m['r2']:.4f}")
    if "cv" in results:
        cv = results["cv"]
        folds = ", ".join(f"{x:.4f}" for x in cv["fold_maes"])
        print(f"CV MAE folds  : [{folds}]")
        print(f"CV MAE        : {cv['mean_mae']:.4f} +/- {cv['std_mae']:.4f}")


def run_pipeline(config: dict[str, Any] | None = None) -> tuple[lgb.Booster, dict[str, Any]]:
    """End-to-end: load -> features -> train -> save (local/`python -m`).

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Tuple of (trained model, results dict).
    """
    from src.data.load import load_dataset
    from src.features.build_features import build_features

    cfg = config or load_config()
    df = build_features(load_dataset(cfg), cfg)
    model, results = train_model(df, cfg)
    path = save_model(model, cfg)
    _report(results)
    print(f"model saved   : {path}")
    return model, results


if __name__ == "__main__":
    run_pipeline()
