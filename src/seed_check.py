"""Seed-spread check: how much do AUC numbers move from seed to seed, and how much of the
FP32 -> quantized change is just "20 more epochs of training"?

For each seed: train FP32 -> fine-tune QKeras twins (as in src.quantize) -> also continue
training the FP32 model for the same number of epochs with no quantization (control).
Nothing in models/ is overwritten.

    python -m src.seed_check --arch 16-3-16 --seeds 5 --bits 8 4
"""
from __future__ import annotations

import argparse

import pandas as pd

from .autoencoder import build_ae, parse_arch
from .common import ROOT, evaluate_model, load_config, load_processed, set_seeds
from .quantize import build_qae, transfer_weights
from .train import fit_autoencoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bits", nargs="*", type=int, default=[8, 4])
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, qc, fprs = cfg["train"], cfg["quant"], cfg["eval"]["fprs"]
    arch = args.arch or cfg["archs"][-1]
    widths = parse_arch(arch)
    data, _ = load_processed(cfg)
    seeds = [cfg["seed"] + i for i in range(args.seeds)]

    rows = []

    def record(seed, variant, metrics):
        rows.append({"seed": seed, "variant": variant,
                     **{k: v for k, v in metrics.items() if k.startswith("auc_")}})

    for seed in seeds:
        set_seeds(seed)
        fp = build_ae(widths)
        fit_autoencoder(fp, data, tc["epochs"], tc["batch_size"], tc["lr"], tc["patience"], verbose=0)
        m, _, _ = evaluate_model(fp, data, fprs)
        record(seed, "fp32", m)

        for bits in args.bits:
            set_seeds(seed)
            qm = build_qae(widths, bits, qc)
            transfer_weights(fp, qm, bits, qc["w_int"])
            fit_autoencoder(qm, data, qc["finetune_epochs"], tc["batch_size"], qc["lr"],
                            qc["patience"], verbose=0)
            m, _, _ = evaluate_model(qm, data, fprs)
            record(seed, f"q{bits}b", m)

        # control: same extra training, no quantization (done last: it modifies fp in place)
        set_seeds(seed)
        fit_autoencoder(fp, data, qc["finetune_epochs"], tc["batch_size"], qc["lr"],
                        qc["patience"], verbose=0)
        m, _, _ = evaluate_model(fp, data, fprs)
        record(seed, "fp32_extra_training", m)
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith("auc_")]
    summary = df.groupby("variant", sort=False)[cols].agg(["mean", "std"])
    base = df[df.variant == "fp32"].set_index("seed")[cols]
    drops = []
    for v in df.variant.unique():
        if v == "fp32":
            continue
        d = base - df[df.variant == v].set_index("seed")[cols]  # positive = worse than FP32
        drops.append({"variant": v, **{f"{c}_drop_mean": d[c].mean() for c in cols},
                      **{f"{c}_drop_std": d[c].std() for c in cols}})
    drops = pd.DataFrame(drops)

    out = ROOT / "results"
    df.to_csv(out / f"seed_check_{arch}.csv", index=False)
    summary.to_csv(out / f"seed_check_{arch}_summary.csv")
    drops.to_csv(out / f"seed_check_{arch}_paired_drops.csv", index=False)
    print(f"\n{arch}: AUC mean/std over {len(seeds)} seeds")
    print(summary.round(3).to_string())
    print("\nPaired drop vs FP32 (positive = worse), mean and std over seeds")
    print(drops.filter(regex="variant|pooled").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
