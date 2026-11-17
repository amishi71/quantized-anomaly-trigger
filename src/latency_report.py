"""Phase 6: collect latency numbers from the resource-estimate CSVs.

Mode A/B give cycle counts. Mode C has no cycle estimate (only layer depth), so it is
never compared against the cycle target; real cycles need synthesis or a predictor.

    python -m src.latency_report
"""
from __future__ import annotations

import pandas as pd

from .common import ROOT, load_config


def main() -> None:
    cfg = load_config()
    target = cfg["estimate"]["latency_target_cycles"]
    frames = []
    for mode in ("A", "B", "C"):
        p = ROOT / "results" / f"resource_estimates_{mode}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No results/resource_estimates_*.csv found. Run src.estimate_resources.")
    df = pd.concat(frames, ignore_index=True)
    for col in ("latency_best_cycles", "latency_worst_cycles", "clock_est_ns"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else float("nan")
    df["latency_ns"] = df["latency_worst_cycles"].fillna(df["latency_best_cycles"]) * df["clock_est_ns"]
    keep = ["arch", "bits", "sparsity", "estimate_mode", "label", "latency_best_cycles",
            "latency_worst_cycles", "clock_est_ns", "latency_ns", "layer_depth"]
    out = df[keep]
    out.to_csv(ROOT / "results" / "latency_report.csv", index=False)
    print(out.round(2).to_string(index=False))

    cyc = out.dropna(subset=["latency_best_cycles"])
    if cyc.empty:
        print("\nNo cycle estimates yet (mode C only has layer depth). Latency target "
              f"(<{target} cycles) not assessed.")
        return
    smallest = df.loc[df["total_weight_bits"].idxmin()]
    val = smallest["latency_worst_cycles"] if pd.notna(smallest.get("latency_worst_cycles")) else smallest["latency_best_cycles"]
    print(f"\nSmallest config {smallest['arch']} {smallest['bits']}b s={smallest['sparsity']}: "
          f"{val} cycles ({smallest['label']}) -> target <{target}: {'MET' if val < target else 'NOT MET'}")


if __name__ == "__main__":
    main()
