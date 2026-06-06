"""Evaluation & interpretation for the BART ridership model.

Loads the trained model (no re-training) and produces:
    * Holdout metrics (MAE / RMSE / R^2) on the 2017 temporal test set.
    * Feature importance table (gain & split) + a written interpretation
      (analiz 3.7: "grafik var, yorum yok" eksikligini kapatir).
    * Residual + predicted-vs-actual figures saved under reports/figures/.

Model is read from the repo's `models/` dir (PROJECT_ROOT-relative), so it
works the same locally and on Kaggle (clone getirir).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless: figürleri dosyaya yaz, ekran açma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT, get_paths, load_config
from src.models.train import get_feature_columns, temporal_split


def get_model_path(config: dict[str, Any]) -> Path:
    """Resolve the committed model path (repo `models/` dir).

    Args:
        config: Loaded configuration.

    Returns:
        Path to the model file (works locally and in the Kaggle clone).
    """
    return PROJECT_ROOT / "models" / config["files"]["final_model"]


def load_model(config: dict[str, Any] | None = None) -> lgb.Booster:
    """Load the trained LightGBM model from disk (no training).

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        The loaded LightGBM Booster.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    cfg = config or load_config()
    path = get_model_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadi: {path}. Once egitip (train.py) repoya koy."
        )
    return lgb.Booster(model_file=str(path))


def evaluate_holdout(
    model: lgb.Booster, df: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate the model on the 2017 temporal holdout.

    Args:
        model: Trained Booster.
        df: Feature-built DataFrame.
        config: Loaded configuration.

    Returns:
        Tuple of (metrics dict, y_true, y_pred).
    """
    features, _ = get_feature_columns(config)
    target = config["target"]
    _, test = temporal_split(df, config)

    y_true = test[target].to_numpy()
    y_pred = model.predict(test[features], num_iteration=model.best_iteration)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(test)),
    }
    return metrics, y_true, y_pred


def feature_importance_df(model: lgb.Booster) -> pd.DataFrame:
    """Build a sorted feature-importance table (gain + split).

    Args:
        model: Trained Booster.

    Returns:
        DataFrame sorted by gain, with a `gain_pct` column.
    """
    names = model.feature_name()
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    imp = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp: pd.DataFrame, figures_dir: Path) -> Path:
    """Save a horizontal bar chart of feature importance (gain %).

    Args:
        imp: Importance DataFrame from `feature_importance_df`.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "feature_importance.png"
    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"][::-1], imp["gain_pct"][::-1], color="#2b8cbe")
    plt.xlabel("Importance (gain %)")
    plt.title("Feature Importance — BART Throughput Model")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path, sample: int = 50_000
) -> Path:
    """Save residual distribution + predicted-vs-actual figures.

    Args:
        y_true: Ground-truth holdout values.
        y_pred: Model predictions.
        figures_dir: Output directory.
        sample: Max points to scatter (büyük veride hız için örnekler).

    Returns:
        Path to the saved figure.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "residuals.png"
    resid = y_true - y_pred

    rng = np.random.default_rng(42)
    idx = (
        rng.choice(len(y_true), size=sample, replace=False)
        if len(y_true) > sample
        else np.arange(len(y_true))
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(resid, bins=60, color="#756bb1")
    axes[0].set_title("Residual Distribution (y_true - y_pred)")
    axes[0].set_xlabel("Residual")

    axes[1].scatter(y_true[idx], y_pred[idx], s=3, alpha=0.2, color="#2b8cbe")
    lim = max(y_true[idx].max(), y_pred[idx].max())
    axes[1].plot([0, lim], [0, lim], "r--", linewidth=1)
    axes[1].set_title("Predicted vs Actual (holdout sample)")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def interpret_importance(imp: pd.DataFrame) -> str:
    """Produce a short written interpretation of the importance table.

    Args:
        imp: Importance DataFrame from `feature_importance_df`.

    Returns:
        Human-readable interpretation string (analiz 3.7).
    """
    top = imp.iloc[0]
    lag_mask = imp["feature"].str.contains("lag|roll", case=False)
    lag_share = imp.loc[lag_mask, "gain_pct"].sum()
    return (
        f"En güçlü sürücü '{top['feature']}' (gain %{top['gain_pct']:.1f}). "
        f"Lag/rolling feature'lar toplam gain'in %{lag_share:.1f}'ini açıklıyor "
        f"-> talep güçlü şekilde zaman-otokorelasyonlu (yakın geçmiş yakın geleceği "
        f"belirliyor). Kalan sinyal ağırlıkla Hour/Origin/Destination/Period "
        f"(ne zaman + nerede) — EDA'daki yoğun saat & istasyon bulgularıyla tutarlı."
    )


def run_evaluation(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """End-to-end evaluation: load model -> metrics -> importance -> figures.

    Args:
        config: Optional pre-loaded config; loaded from disk if omitted.

    Returns:
        Tuple of (holdout metrics, importance DataFrame).
    """
    from src.data.load import load_dataset
    from src.features.build_features import build_features

    cfg = config or load_config()
    model = load_model(cfg)
    df = build_features(load_dataset(cfg), cfg)

    metrics, y_true, y_pred = evaluate_holdout(model, df, cfg)
    imp = feature_importance_df(model)
    figs_dir = get_paths(cfg)["figures_dir"]
    fi_path = plot_feature_importance(imp, figs_dir)
    res_path = plot_residuals(y_true, y_pred, figs_dir)

    print("=== Holdout (2017) evaluation ===")
    print(f"n_test : {metrics['n_test']:,}")
    print(f"MAE    : {metrics['mae']:.4f}")
    print(f"RMSE   : {metrics['rmse']:.4f}")
    print(f"R2     : {metrics['r2']:.4f}")
    print("\n=== Feature importance (gain %) ===")
    print(imp[["feature", "gain_pct", "split"]].to_string(index=False))
    print("\n=== Yorum ===")
    print(interpret_importance(imp))
    print(f"\nfigures: {fi_path}\n         {res_path}")
    return metrics, imp


if __name__ == "__main__":
    run_evaluation()
