"""Phase 2: train FP32 autoencoders on QCD only, evaluate per anomaly class.

    python -m src.train                       # the three baseline architectures
    python -m src.train --archs 16-8-4-3-4-8-16   # e.g. a fallback latent
"""
from __future__ import annotations

import argparse

import pandas as pd

from .autoencoder import build_ae, expected_params, parse_arch
from .common import ROOT, evaluate_model, load_config, load_processed, scores, set_seeds
from .roc_auc import plot_roc
from .threshold_scan import scan


def fit_autoencoder(model, data, epochs, batch_size, lr, patience, verbose=2):
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True)
    return model.fit(data["X_train"], data["X_train"],
                     validation_data=(data["X_val"], data["X_val"]),
                     epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, fprs = cfg["train"], cfg["eval"]["fprs"]
    archs = args.archs or cfg["archs"]

    data, meta = load_processed(cfg)
    res, models_dir = ROOT / "results", ROOT / "models" / "fp32"
    res.mkdir(exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows, pf_rows, scan_rows = [], [], []
    for arch in archs:
        set_seeds(cfg["seed"])
        widths = parse_arch(arch)
        model = build_ae(widths)
        assert model.count_params() == expected_params(widths)
        hist = fit_autoencoder(model, data, tc["epochs"], tc["batch_size"], tc["lr"], tc["patience"])
        model.save(models_dir / f"{arch}.h5")

        metrics, pf, (s_qcd, s_an) = evaluate_model(model, data, fprs)
        rows.append({"arch": arch, "n_params": model.count_params(),
                     "epochs_run": len(hist.history["loss"]),
                     "best_val_loss": min(hist.history["val_loss"]), **metrics})
        pf_rows.append({"arch": arch, **dict(zip(meta["feature_names"], pf))})
        sc = scan(scores(model, data["X_val"]), s_qcd, s_an, fprs)
        sc.insert(0, "arch", arch)
        scan_rows.append(sc)
        plot_roc(s_qcd, s_an, res / f"roc_fp32_{arch}.png", title=f"FP32 {arch}")

        target = cfg["eval"]["baseline_auc_target"]
        msg = "OK" if metrics["auc_pooled"] >= target else f"BELOW {target} target (record honestly)"
        print(f"[{arch}] pooled AUC {metrics['auc_pooled']:.3f} -> {msg}")

    pd.DataFrame(rows).to_csv(res / "fp32_baseline.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(res / "fp32_per_feature_mse.csv", index=False)
    pd.concat(scan_rows).to_csv(res / "threshold_scan_fp32.csv", index=False)
    print(pd.DataFrame(rows).filter(regex="arch|n_params|auc_").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
