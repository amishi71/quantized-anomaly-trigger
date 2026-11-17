"""Phase 7: multi-seed sweep over architecture x bits x sparsity.

For each (seed, arch): train FP32 -> QKeras fine-tune at each weight bit-width -> prune each
quantized model to each sparsity (0 = the quantized model itself) -> evaluate, plus resource
proxy (mode C) and collapse diagnostics (r2, share of exactly-zero outputs, trivial-detector AUC).
One row per (seed, arch, bits, sparsity) is appended to results/sweep_runs.csv after each
(seed, arch) group finishes, so the run is resumable: re-running skips finished groups.
Delete/move results/sweep_runs.csv if you change the grid or the quantizer settings.

Pruning here always starts from the quantized model (sequential order).

    python -m src.sweep --archs 16-3-16 --seeds 1      # pilot / timing
    python -m src.sweep --seeds 5                      # full sweep
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .autoencoder import build_ae, expected_params, parse_arch
from .common import (ROOT, anomaly_arrays, evaluate_model, load_config, load_processed,
                     reconstruct, set_seeds)
from .estimate_resources import proxy_row
from .prune import prune_and_finetune
from .quantize import build_qae, transfer_weights
from .roc_auc import auc
from .train import fit_autoencoder

OUT = ROOT / "results" / "sweep_runs.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--sparsities", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, qc, pc, fprs = cfg["train"], cfg["quant"], cfg["prune"], cfg["eval"]["fprs"]
    archs = args.archs or cfg["archs"]
    bits_list = args.bits or qc["bits"]
    sparsities = args.sparsities if args.sparsities is not None else pc["sparsities"]
    seeds = [cfg["seed"] + i for i in range(args.seeds)]
    baseline_params = expected_params(parse_arch(cfg["archs"][0]))

    import tensorflow as tf

    data, _ = load_processed(cfg)
    (ROOT / "results").mkdir(exist_ok=True)

    # trivial detector: score = mean(x^2) (inputs are standardised; equals the error of an all-zero output)
    X = data["X_test_qcd"]
    anoms = anomaly_arrays(data)
    neg = (X ** 2).mean(axis=1)
    triv = {k: auc(neg, (v ** 2).mean(axis=1)) for k, v in anoms.items()}
    triv["pooled"] = auc(neg, np.concatenate([(v ** 2).mean(axis=1) for v in anoms.values()]))
    mse0 = float(np.mean(X ** 2))

    def recon_stats(model) -> dict:
        xh = reconstruct(model, X)
        return {"r2": 1 - float(np.mean((X - xh) ** 2)) / mse0,
                "pct_out_zero": float(100 * np.mean(xh == 0))}

    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(zip(prev["arch"], prev["seed"]))
        print(f"Resuming: {len(done)} (arch, seed) groups already in {OUT.name}")

    todo = [(s, a) for s in seeds for a in archs if (a, s) not in done]
    print(f"{len(todo)} groups to run")
    t_start = time.time()
    for n, (seed, arch) in enumerate(todo, 1):
        t0 = time.time()
        widths = parse_arch(arch)
        set_seeds(seed)
        fp = build_ae(widths)
        fit_autoencoder(fp, data, tc["epochs"], tc["batch_size"], tc["lr"], tc["patience"], verbose=0)
        fp_m, _, _ = evaluate_model(fp, data, fprs)
        r2_fp32 = recon_stats(fp)["r2"]

        rows = []
        for bits in bits_list:
            set_seeds(seed)
            qm = build_qae(widths, bits, qc)
            transfer_weights(fp, qm, bits, qc["w_int"])
            fit_autoencoder(qm, data, qc["finetune_epochs"], tc["batch_size"], qc["lr"],
                            qc["patience"], verbose=0)
            for s in sparsities:
                if s == 0:
                    model = qm
                else:
                    set_seeds(seed)
                    model = prune_and_finetune(qm, widths, bits, qc, data, s, pc["epochs"],
                                               pc["hold_epochs"], tc["batch_size"], pc["lr"],
                                               pc["frequency"])
                m, _, _ = evaluate_model(model, data, fprs)
                px = proxy_row(model, bits, qc)
                row = {"seed": seed, "arch": arch, "bits": bits, "act_bits": qc.get("act_bits") or bits,
                       "sparsity": s, "n_params_dense": expected_params(widths), **px,
                       "param_reduction_vs_baseline": 1 - px["nnz_params"] / baseline_params,
                       "bits_reduction_vs_fp32_baseline": 1 - px["total_weight_bits"] / (baseline_params * 32),
                       "n_feat_worse_than_mean": m["n_feat_worse_than_mean"],
                       **recon_stats(model), "r2_fp32": r2_fp32}
                row.update({k: v for k, v in m.items() if k.startswith("auc_")})
                row.update({k: v for k, v in m.items() if k.startswith("tpr_") and k.endswith("_pooled")})
                row.update({f"{k}_fp32": v for k, v in fp_m.items() if k.startswith("auc_")})
                row.update({f"auc_trivial_{k}": v for k, v in triv.items()})
                rows.append(row)

        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        tf.keras.backend.clear_session()
        dt = time.time() - t0
        eta = (time.time() - t_start) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] seed {seed} {arch}: {len(rows)} rows in {dt / 60:.1f} min "
              f"(ETA {eta / 60:.0f} min)")
    print("Done. Next: python -m src.aggregate")


if __name__ == "__main__":
    main()
