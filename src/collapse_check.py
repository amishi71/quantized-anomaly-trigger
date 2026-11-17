"""Are the models really reconstructing, or collapsing toward a constant (zero) output?

Inputs are standardised, so an all-zero reconstruction has per-jet error mean(x^2). Scoring
jets by mean(x^2) is therefore a free "trivial detector" any autoencoder has to beat.
For every saved model (FP32, quantized, pruned; seed-42 snapshots from Phases 2-4) this reports:
  r2            = 1 - MSE_model / mean(x^2) on QCD test (about 0 means no reconstruction)
  pct_out_zero  = % of output values that are exactly 0
  auc_*         = AUCs, and d_* = AUC minus the trivial detector's AUC (same class)

    python -m src.collapse_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ROOT, anomaly_arrays, evaluate_model, load_config, load_processed, reconstruct
from .model_io import load_qmodel, weights_path
from .roc_auc import auc


def main() -> None:
    cfg = load_config()
    fprs, qc = cfg["eval"]["fprs"], cfg["quant"]
    data, _ = load_processed(cfg)
    X = data["X_test_qcd"]
    neg = (X ** 2).mean(axis=1)
    anoms = anomaly_arrays(data)
    triv = {k: auc(neg, (v ** 2).mean(axis=1)) for k, v in anoms.items()}
    triv["pooled"] = auc(neg, np.concatenate([(v ** 2).mean(axis=1) for v in anoms.values()]))
    print("Trivial detector (score = mean x^2): " + ", ".join(f"{k} {v:.3f}" for k, v in triv.items()))

    import tensorflow as tf

    def row(label, model):
        m, _, _ = evaluate_model(model, data, fprs)
        xh = reconstruct(model, X)
        r = {**label,
             "r2": 1 - float(np.mean((X - xh) ** 2) / np.mean(X ** 2)),
             "pct_out_zero": float(100 * np.mean(xh == 0))}
        for k in ["pooled", *anoms]:
            r[f"auc_{k}"] = m[f"auc_{k}"]
            r[f"d_{k}"] = m[f"auc_{k}"] - triv[k]
        return r

    rows = []
    for arch in cfg["archs"]:
        p = ROOT / "models" / "fp32" / f"{arch}.h5"
        if p.exists():
            rows.append(row({"arch": arch, "bits": 32, "sparsity": 0.0},
                            tf.keras.models.load_model(p, compile=False)))
        for bits in qc["bits"]:
            for s in cfg["prune"]["sparsities"]:
                if weights_path(arch, bits, s).exists():
                    rows.append(row({"arch": arch, "bits": bits, "sparsity": s},
                                    load_qmodel(arch, bits, cfg, s)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "collapse_check.csv", index=False)
    print("\nr2 ~ 0 and pct_out_zero ~ 100 mean the model outputs (nearly) nothing; "
          "d_* < 0 means worse than the trivial detector.\n")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
