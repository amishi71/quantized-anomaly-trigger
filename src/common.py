"""Shared helpers: config, seeding, data loading, scoring, evaluation."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def load_processed(cfg: dict):
    d = ROOT / cfg["data"]["processed_dir"]
    npz_path = d / "dataset.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run: python -m src.prepare_dataset")
    z = np.load(npz_path)
    data = {k: z[k] for k in z.files}
    meta = json.loads((d / "meta.json").read_text())
    return data, meta


def anomaly_arrays(data: dict) -> dict:
    return {k[len("X_anom_"):]: v for k, v in data.items() if k.startswith("X_anom_")}


def reconstruct(model, X, batch_size: int = 8192):
    return model.predict(X, batch_size=batch_size, verbose=0)


def scores(model, X) -> np.ndarray:
    """Per-jet anomaly score = mean squared reconstruction error."""
    return np.mean((X - reconstruct(model, X)) ** 2, axis=1)


def per_feature_mse(model, X) -> np.ndarray:
    return np.mean((X - reconstruct(model, X)) ** 2, axis=0)


def evaluate_model(model, data: dict, fprs):
    """Return (metrics dict, per-feature MSE on QCD test, (qcd scores, {class: scores}))."""
    from .roc_auc import evaluate_scores

    s_qcd = scores(model, data["X_test_qcd"])
    s_an = {k: scores(model, v) for k, v in anomaly_arrays(data).items()}
    metrics = evaluate_scores(s_qcd, s_an, fprs)
    pf = per_feature_mse(model, data["X_test_qcd"])
    # inputs are standardised (variance 1): MSE > 1 is worse than predicting the mean
    metrics["n_feat_worse_than_mean"] = int((pf > 1.0).sum())
    return metrics, pf, (s_qcd, s_an)
