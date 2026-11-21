"""Is reconstruction MSE the right score? Alternative scores from the SAME trained models, plus the
trigger-regime view (TPR at low false-positive rate), with the anomaly samples split into dev / holdout halves.

Scores, declared in advance (not tuned):
  mse       mean squared reconstruction error (what the project uses so far)
  mse_std   squared error per feature divided by that feature's mean squared error on QCD validation jets
  latent    Mahalanobis distance of the latent code from the QCD-validation latent distribution
  norm      mean(x^2) of the input: the trivial detector (no model involved)
Rule: choose a score on `dev`, report it on `holdout`. Negatives (QCD test) are shared by both halves.
These are single-seed snapshots (the models saved by Phases 2 to 4), so only large differences mean anything.

    python -m src.score_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .autoencoder import latent_index, parse_arch
from .common import ROOT, anomaly_arrays, load_config, load_processed
from .model_io import load_qmodel, weights_path
from .roc_auc import auc, tpr_at_fpr

FPRS = (1e-2, 1e-3)


def fit_reference(X_val, Xh_val, Z_val) -> dict:
    """Statistics from QCD validation jets only (never from test or anomaly data)."""
    cov = np.cov(Z_val, rowvar=False) + 1e-6 * np.eye(Z_val.shape[1])
    return {"e_ref": np.mean((X_val - Xh_val) ** 2, axis=0) + 1e-12,
            "mu": Z_val.mean(axis=0), "cov_inv": np.linalg.inv(cov)}


def compute_scores(X, Xh, Z, ref) -> dict:
    err = (X - Xh) ** 2
    d = Z - ref["mu"]
    return {"mse": err.mean(axis=1),
            "mse_std": (err / ref["e_ref"]).mean(axis=1),
            "latent": np.einsum("ij,jk,ik->i", d, ref["cov_inv"], d),
            "norm": (X ** 2).mean(axis=1)}


def split_masks(anoms: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {k: rng.random(len(v)) < 0.5 for k, v in anoms.items()}  # True = dev


def split_metrics(neg, pos_by_class: dict) -> dict:
    out = {f"auc_{k}": auc(neg, v) for k, v in pos_by_class.items()}
    pooled = np.concatenate(list(pos_by_class.values()))
    out["auc_pooled"] = auc(neg, pooled)
    for f in FPRS:
        out[f"tpr_fpr{f:g}_pooled"] = tpr_at_fpr(neg, pooled, f)
    return out


def _rows_for(label: dict, neg_scores: dict, anom_scores: dict, masks: dict, score_names) -> list:
    rows = []
    for s in score_names:
        for split, pick in (("dev", lambda m: m), ("holdout", lambda m: ~m)):
            pos = {k: anom_scores[k][s][pick(masks[k])] for k in anom_scores}
            rows.append({**label, "score": s, "split": split, **split_metrics(neg_scores[s], pos)})
    return rows


def main() -> None:
    cfg = load_config()
    qc = cfg["quant"]
    data, _ = load_processed(cfg)
    anoms = anomaly_arrays(data)
    masks = split_masks(anoms, cfg["seed"])
    Xq, Xv = data["X_test_qcd"], data["X_val"]

    import tensorflow as tf

    def predict_pair(model, latent_name, X):
        probe = tf.keras.Model(model.input, [model.output, model.get_layer(latent_name).output])
        Xh, Z = probe.predict(X, batch_size=8192, verbose=0)
        return Xh, Z

    rows = []
    # trivial detector: no model
    neg0 = {"norm": (Xq ** 2).mean(axis=1)}
    an0 = {k: {"norm": (v ** 2).mean(axis=1)} for k, v in anoms.items()}
    rows += _rows_for({"arch": "(none)", "bits": 0, "sparsity": 0.0}, neg0, an0, masks, ["norm"])

    def run(label, model, latent_name):
        ref = fit_reference(Xv, *predict_pair(model, latent_name, Xv))
        negs = compute_scores(Xq, *predict_pair(model, latent_name, Xq), ref)
        ans = {k: compute_scores(v, *predict_pair(model, latent_name, v), ref) for k, v in anoms.items()}
        return _rows_for(label, negs, ans, masks, ["mse", "mse_std", "latent"])

    for arch in cfg["archs"]:
        lat = latent_index(parse_arch(arch))
        p = ROOT / "models" / "fp32" / f"{arch}.h5"
        if p.exists():
            rows += run({"arch": arch, "bits": 32, "sparsity": 0.0},
                        tf.keras.models.load_model(p, compile=False), f"fc{lat + 1}")
        for bits in qc["bits"]:
            for s in cfg["prune"]["sparsities"]:
                if weights_path(arch, bits, s).exists():
                    rows += run({"arch": arch, "bits": bits, "sparsity": s},
                                load_qmodel(arch, bits, cfg, s), "q_latent")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "score_study.csv"
    df.to_csv(out, index=False)

    cols = ["auc_w", "auc_z", "auc_t", "auc_pooled", "tpr_fpr0.01_pooled", "tpr_fpr0.001_pooled"]
    triv = df[(df.arch == "(none)")].set_index("split")
    print("TRIVIAL detector (score = mean x^2):")
    print(triv[cols].round(3).to_string())
    print("\nFP32 models: score x split (dev = choose here, holdout = report here)")
    fp = df[(df.bits == 32)].sort_values(["arch", "score", "split"])
    print(fp[["arch", "score", "split", *cols]].round(3).to_string(index=False))

    print("\nCompressed models (QKeras + pruned), holdout split: how many beat the trivial detector?")
    comp = df[(df.bits > 0) & (df.bits < 32) & (df.split == "holdout")]
    t_ho = triv.loc["holdout"]
    summ = comp.groupby("score").agg(n=("arch", "size"), mean_auc=("auc_pooled", "mean"),
                                     beat_auc=("auc_pooled", lambda s: int((s > t_ho["auc_pooled"]).sum())),
                                     beat_tpr_1e3=("tpr_fpr0.001_pooled",
                                                   lambda s: int((s > t_ho["tpr_fpr0.001_pooled"]).sum())))
    print(summ.round(3).to_string())
    print(f"\n(trivial on holdout: pooled AUC {t_ho['auc_pooled']:.3f}, TPR@1e-3 {t_ho['tpr_fpr0.001_pooled']:.3f})")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
