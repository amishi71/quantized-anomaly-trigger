"""Phase 1: load the 16-feature jet dataset, build the QCD / anomaly split, preprocess.

Normal = g + q (QCD). Train/val/test split is applied to QCD only; W, Z, top are
evaluation-only. Clip bounds and scaler are fitted on QCD train only.

    python -m src.prepare_dataset
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .common import ROOT, load_config
from .preprocessing import Preprocessor

CLASSES = ["g", "q", "w", "z", "t"]
# 16 high-level features as I recall them; verify against your download.
FEATURES = [
    "zlogz", "c1_b0_mmdt", "c1_b1_mmdt", "c1_b2_mmdt", "c2_b1_mmdt",
    "c2_b2_mmdt", "d2_b1_mmdt", "d2_b2_mmdt", "d2_a1_b1_mmdt",
    "d2_a1_b2_mmdt", "m2_b1_mmdt", "m2_b2_mmdt", "n2_b1_mmdt",
    "n2_b2_mmdt", "mass_mmdt", "multiplicity",
]


def _norm_label(v) -> str:
    s = str(v).strip().lower()
    if s.startswith("j_"):
        s = s[2:]
    if s.isdigit() and int(s) < len(CLASSES):
        s = CLASSES[int(s)]
    return s


def split_frame(df: pd.DataFrame):
    """DataFrame -> (X [N,16] float, y [N] str in CLASSES, feature_names)."""
    lower = {c.lower(): c for c in df.columns}
    onehot = [lower.get(f"j_{c}") for c in CLASSES]
    if all(onehot):
        y = np.array(CLASSES)[df[onehot].to_numpy().argmax(axis=1)]
        feats = df.drop(columns=onehot)
    elif "class" in lower:
        y = np.array([_norm_label(v) for v in df[lower["class"]]])
        feats = df.drop(columns=[lower["class"]])
    else:
        raise ValueError(f"Cannot find label columns. Columns: {list(df.columns)[:30]}")
    feats = feats.rename(columns={c: str(c).lower().removeprefix("j_") for c in feats.columns})
    if all(f in feats.columns for f in FEATURES):
        feats = feats[FEATURES]
    else:
        feats = feats.select_dtypes("number")
    if feats.shape[1] != 16:
        raise ValueError(f"Expected 16 features, got {feats.shape[1]}: {list(feats.columns)}")
    return feats.to_numpy(dtype=np.float64), y, list(feats.columns)


def load_raw(cfg: dict):
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    for p in sorted(raw_dir.glob("*")) if raw_dir.exists() else []:
        if p.suffix == ".csv":
            return split_frame(pd.read_csv(p))
        if p.suffix == ".parquet":
            return split_frame(pd.read_parquet(p))
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            y = z["y"]
            if y.ndim == 2:
                y = np.array(CLASSES)[y.argmax(axis=1)]
            names = list(z["feature_names"]) if "feature_names" in z.files else [f"f{i}" for i in range(z["X"].shape[1])]
            return z["X"].astype(np.float64), np.array([_norm_label(v) for v in y]), names
    from sklearn.datasets import fetch_openml

    try:
        bunch = fetch_openml(name=cfg["data"]["openml_name"], as_frame=True, parser="auto")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not fetch the dataset from OpenML. Download the 16-feature hls4ml jet "
            f"dataset (Zenodo) and put a csv/parquet/npz in {raw_dir}. Original error: {e}"
        ) from e
    return split_frame(bunch.frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["data"]
    seed = cfg["seed"]

    X, y, names = load_raw(cfg)
    ok = np.isfinite(X).all(axis=1)
    if (~ok).any():
        print(f"Dropping {(~ok).sum()} rows with NaN/inf")
    X, y = X[ok], y[ok]

    counts = {c: int((y == c).sum()) for c in CLASSES}
    print("Class counts:", counts)
    for c in dc["normal_classes"] + dc["anomaly_classes"]:
        assert counts.get(c, 0) > 0, f"class {c} missing"

    skew = pd.DataFrame(X, columns=names).skew()
    heavy = skew[skew.abs() > 3]
    print("Heavy-tailed features (|skew|>3):", dict(heavy.round(1)) if len(heavy) else "none")

    # QCD split
    rng = np.random.default_rng(seed)
    qcd_idx = np.flatnonzero(np.isin(y, dc["normal_classes"]))
    rng.shuffle(qcd_idx)
    n = len(qcd_idx)
    n_tr, n_val = int(dc["split"][0] * n), int(dc["split"][1] * n)
    tr, va, te = qcd_idx[:n_tr], qcd_idx[n_tr:n_tr + n_val], qcd_idx[n_tr + n_val:]

    log_idx = [names.index(f) for f in dc["log_features"] if f in names]
    if len(log_idx) != len(dc["log_features"]):
        print("WARNING: some log_features not found in feature names; skipped.")
    pre = Preprocessor(dc["clip_percentiles"], log_idx).fit(X[tr])

    out = {"X_train": pre.transform(X[tr]), "X_val": pre.transform(X[va]),
           "X_test_qcd": pre.transform(X[te])}
    for c in dc["anomaly_classes"]:
        out[f"X_anom_{c}"] = pre.transform(X[y == c])

    pdir = ROOT / dc["processed_dir"]
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "dataset.npz", **out)
    joblib.dump(pre, pdir / "preprocessor.joblib")
    meta = {"feature_names": names, "class_counts": counts,
            "split_sizes": {"train": len(tr), "val": len(va), "test_qcd": len(te)},
            "log_features": [names[i] for i in log_idx], "seed": seed}
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2))

    # raw feature histograms (QCD train) to spot heavy tails
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 4, figsize=(14, 10))
    for i, ax in enumerate(axs.ravel()):
        ax.hist(X[tr][:, i], bins=80, log=True)
        ax.set_title(names[i], fontsize=8)
    fig.tight_layout()
    (ROOT / "results").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "feature_hist_raw.png", dpi=110)

    print({k: v.shape for k, v in out.items()})
    print(f"Saved to {pdir}")


if __name__ == "__main__":
    main()
