"""Phase 7: aggregate the multi-seed sweep into compression_table.csv + Pareto plot.

A run is COLLAPSED if its reconstruction r2 is below `eval.collapse_r2_abs` (it explains almost none
of the input) or more than `eval.collapse_zero_pct` % of its outputs are exactly zero. A collapsed
model scores about the trivial detector's AUC, so its AUC says nothing about compression: configs
with any collapsed seed are excluded from the Pareto front.
A run is DEGRADED if it is not collapsed but its r2 is below `eval.collapse_r2_frac` x the FP32
model's r2. Degraded configs stay in the table and on the front (hollow markers); a second front
(`pareto_full_quality`) uses only configs with no degraded seed.

    python -m src.aggregate                      # cost = total weight bits
    python -m src.aggregate --cost bops
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import ROOT, load_config


def pareto_flags(df: pd.DataFrame, cost: str, metric: str) -> pd.Series:
    """True where no cheaper-or-equal config has an equal-or-better metric."""
    best, flags = -np.inf, pd.Series(False, index=df.index)
    for idx, r in df.sort_values([cost, metric], ascending=[True, False]).iterrows():
        if r[metric] > best + 1e-12:
            flags[idx], best = True, r[metric]
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cost", default="total_weight_bits",
                    choices=["total_weight_bits", "bops", "n_mult", "nnz_params"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    target = cfg["quant"]["auc_drop_target"]
    zero_pct = cfg["eval"].get("collapse_zero_pct", 50)
    r2_abs = cfg["eval"].get("collapse_r2_abs", 0.2)
    r2_frac = cfg["eval"].get("collapse_r2_frac", 0.9)
    classes = ["pooled", *cfg["data"]["anomaly_classes"]]

    src = ROOT / "results" / "sweep_runs.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found. Run: python -m src.sweep")
    df = pd.read_csv(src)
    if "r2" not in df.columns:
        raise SystemExit("sweep_runs.csv has no collapse columns (r2, pct_out_zero). It is from an "
                         "older sweep: move it away and re-run python -m src.sweep.")
    for k in classes:
        df[f"drop_{k}"] = df[f"auc_{k}_fp32"] - df[f"auc_{k}"]   # positive = worse than FP32
    df["collapsed"] = (df["r2"] < r2_abs) | (df["pct_out_zero"] > zero_pct)
    df["degraded"] = (~df["collapsed"]) & (df["r2"] < r2_frac * df["r2_fp32"])

    rows = []
    for (arch, bits, s), g in df.groupby(["arch", "bits", "sparsity"], sort=False):
        n = g["seed"].nunique()
        r = {"arch": arch, "bits": bits, "act_bits": g["act_bits"].iloc[0], "sparsity": s, "n_seeds": n}
        for k in classes:
            r[f"auc_{k}_mean"], r[f"auc_{k}_std"] = g[f"auc_{k}"].mean(), g[f"auc_{k}"].std()
            r[f"d_trivial_{k}_mean"] = r[f"auc_{k}_mean"] - g[f"auc_trivial_{k}"].iloc[0]
            d = g[f"drop_{k}"]
            r[f"drop_{k}_mean"], r[f"drop_{k}_std"] = d.mean(), d.std()
            r[f"drop_{k}_sem"] = d.std() / np.sqrt(n) if n > 1 else np.nan
        for c in ["nnz_params", "total_weight_bits", "bops", "n_mult", "kernel_sparsity",
                  "param_reduction_vs_baseline", "bits_reduction_vs_fp32_baseline",
                  "r2", "r2_fp32", "pct_out_zero"]:
            r[c + ("_mean" if c in ("r2", "r2_fp32", "pct_out_zero") else "")] = g[c].mean()
        r["auc_fp32_pooled_mean"] = g["auc_pooled_fp32"].mean()
        r["n_params_dense"] = g["n_params_dense"].iloc[0]
        r["collapse_rate"] = float(g["collapsed"].mean())
        r["degraded_rate"] = float(g["degraded"].mean())
        r["valid"] = bool(r["collapse_rate"] == 0)
        r["full_quality"] = bool(r["valid"] and r["degraded_rate"] == 0)
        r["meets_target_pooled_mean"] = bool(r["drop_pooled_mean"] < target)
        r["meets_target_all_classes_mean"] = all(r[f"drop_{k}_mean"] < target for k in classes)
        r["meets_target_conservative"] = bool(r["drop_pooled_mean"] + 2 * r["drop_pooled_sem"] < target) \
            if n > 1 else False
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab["pareto"] = False
    tab["pareto_full_quality"] = False
    for col, mask in (("pareto", tab["valid"]), ("pareto_full_quality", tab["full_quality"])):
        if mask.any():
            tab.loc[mask, col] = pareto_flags(tab[mask], args.cost, "auc_pooled_mean")
    tab.to_csv(ROOT / "results" / "compression_table.csv", index=False)

    n_min = tab["n_seeds"].min()
    print(f"{len(tab)} configs, {n_min}-{tab['n_seeds'].max()} seeds each. Pareto cost = {args.cost}")
    print(f"Collapsed in >=1 seed: {int((~tab.valid).sum())} of {len(tab)} configs "
          f"(rule: r2 < {r2_abs} or > {zero_pct}% zero outputs)")
    print(f"Not collapsed but degraded in >=1 seed (r2 < {r2_frac} x FP32 r2): "
          f"{int((tab.valid & ~tab.full_quality).sum())}; full quality: {int(tab.full_quality.sum())}")
    print(f"Non-collapsed configs beating the trivial detector on pooled AUC: "
          f"{int((tab.valid & (tab.d_trivial_pooled_mean > 0)).sum())} of {int(tab.valid.sum())}\n")
    cols = ["arch", "bits", "sparsity", "nnz_params", "total_weight_bits", "r2_mean", "collapse_rate",
            "degraded_rate", "auc_pooled_mean", "auc_pooled_std", "d_trivial_pooled_mean",
            *[f"d_trivial_{k}_mean" for k in classes[1:]]]
    print("All configs (d_trivial_* = AUC minus the trivial detector's; negative = worse):")
    print(tab.sort_values(args.cost)[cols].round(3).to_string(index=False))
    show = ["arch", "bits", "sparsity", "n_seeds", "nnz_params", "total_weight_bits", "r2_mean",
            "degraded_rate", "auc_pooled_mean", "auc_pooled_std", "d_trivial_pooled_mean",
            "drop_pooled_mean", "drop_pooled_sem"]
    print("\nPareto front among non-collapsed configs (AUC is a test-set mean; picking the best of "
          "many noisy configs overstates it):")
    print(tab[tab.pareto].sort_values(args.cost)[show].round(3).to_string(index=False))
    ok = tab[tab.valid & (tab.param_reduction_vs_baseline >= 0.6) & tab.meets_target_pooled_mean]
    print(f"\nNon-collapsed configs with >=60% fewer non-zero params AND mean pooled drop < {target}: "
          f"{len(ok)} ({int(ok.meets_target_conservative.sum())} also pass with a 2-sem margin)")
    print(f"Pareto points: {int(tab.pareto.sum())} non-collapsed, "
          f"{int(tab.pareto_full_quality.sum())} full-quality only (target >= 3)")
    if n_min < 3:
        print("WARNING: fewer than 3 seeds for some configs; std/sem are not meaningful.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    marks = {0.0: "o", 0.5: "s", 0.75: "^"}
    archs = list(tab.arch.unique())
    for i, a in enumerate(archs):
        for s, mk in marks.items():
            for deg in (False, True):
                t = tab[(tab.arch == a) & (tab.sparsity == s) & tab.valid
                        & ((tab.degraded_rate > 0) == deg)]
                if len(t):
                    ax.errorbar(t[args.cost], t.auc_pooled_mean, yerr=t.auc_pooled_std.fillna(0),
                                fmt=mk, color=f"C{i}", mfc="none" if deg else f"C{i}", alpha=0.8,
                                capsize=2, label=None if deg else f"{a}, s={s:g}")
        if args.cost == "total_weight_bits":
            t = tab[tab.arch == a].iloc[0]
            ax.plot(t.n_params_dense * 32, t.auc_fp32_pooled_mean, "x", color=f"C{i}", ms=9)
    bad = tab[~tab.valid]
    if len(bad):
        ax.plot(bad[args.cost], bad.auc_pooled_mean, "x", color="0.6", ms=6, label="collapsed")
    triv = df["auc_trivial_pooled"].iloc[0]
    ax.plot([], [], "o", mfc="none", color="k", label="hollow = degraded r2")
    ax.axhline(triv, color="r", ls=":", lw=1, label=f"trivial detector ({triv:.3f})")
    front = tab[tab.pareto].sort_values(args.cost)
    ax.step(front[args.cost], front.auc_pooled_mean, where="post", color="k", lw=0.8)
    for _, r in front.iterrows():
        ax.annotate(f"{int(r.bits)}b", (r[args.cost], r.auc_pooled_mean), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(args.cost + (" (x = FP32 reference)" if args.cost == "total_weight_bits" else ""))
    ax.set_ylabel("pooled AUC (mean, bars = std over seeds)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "pareto.png", dpi=140)
    print("\nWrote results/compression_table.csv and results/pareto.png")


if __name__ == "__main__":
    main()
