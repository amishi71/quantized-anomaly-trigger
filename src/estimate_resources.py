"""Phase 6: resource estimates. Pick a mode and LABEL it in the README.

  C  analytic proxy (always works): non-zero multipliers, total weight bits, BOPs, layer depth
  B  rule4ml pre-synthesis predictor (labelled "predicted"; API must be verified, see below)
  A  real synthesis via hls_model.build(synth=True) (needs Vitis/Vivado HLS on PATH)

    python -m src.estimate_resources --mode C            # pilot subset
    python -m src.estimate_resources --mode C --grid     # all 27 configs
    python -m src.estimate_resources --mode A --archs 16-3-16 --bits 8 --sparsities 0
"""
from __future__ import annotations

import argparse
import itertools
import shutil

import numpy as np
import pandas as pd

from .common import ROOT, load_config
from .model_io import config_name, load_qmodel, quantized_param_counts

LABEL = {"C": "proxy", "B": "predicted", "A": "synthesized"}


def proxy_row(model, bits: int, qcfg: dict) -> dict:
    """Mode C. Under hls4ml's latency strategy (reuse factor 1) every non-zero weight is one
    multiplier; zero weights are removed. Power-of-two weights could become shifts: not modelled.
    BOPs follow Baskin et al. (UNIQ): per MAC  b_w*b_a + b_a + b_w + log2(fan_in)  (verify)."""
    c = quantized_param_counts(model)
    bops = 0.0
    for i, l in enumerate(c["layers"]):
        b_a = qcfg["input_bits"] if i == 0 else (qcfg.get("act_bits") or bits)
        bops += l["nnz"] * (bits * b_a + b_a + bits + np.log2(max(l["fan_in"], 2)))
    return {"n_mult": c["nnz_kernel"], "nnz_params": c["nnz_params"],
            "total_weight_bits": c["nnz_params"] * bits, "bops": round(bops, 1),
            "kernel_sparsity": c["kernel_sparsity"], "layer_depth": len(c["layers"])}


def synth_row(hls_model) -> dict:
    """Mode A. Key names follow hls4ml's CSynthesisReport as I recall; verify on your version."""
    if not (shutil.which("vitis_hls") or shutil.which("vivado_hls")):
        raise RuntimeError("Mode A needs vitis_hls/vivado_hls on PATH. Use mode C (or B) instead.")
    rep = hls_model.build(csim=False, synth=True, cosim=False, validation=False,
                          export=False, vsynth=False)
    cs = rep.get("CSynthesisReport", {}) if isinstance(rep, dict) else {}
    pick = lambda *ks: next((cs[k] for k in ks if k in cs), None)  # noqa: E731
    return {"lut": pick("LUT"), "ff": pick("FF"), "dsp": pick("DSP", "DSP48E"),
            "bram_18k": pick("BRAM_18K"), "latency_best_cycles": pick("BestLatency"),
            "latency_worst_cycles": pick("WorstLatency"), "interval_cycles": pick("IntervalMin"),
            "clock_est_ns": pick("EstimatedClockPeriod")}


def rule4ml_row(qmodel, hls_config: dict) -> dict:
    """Mode B. UNVERIFIED: written from memory of the rule4ml README. Check the API for your
    installed version; any failure here raises so you don't silently get empty numbers."""
    try:
        from rule4ml.models.estimators import MultiTaskPredictor

        df = MultiTaskPredictor.get_default().predict(qmodel, hls_config)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"rule4ml call failed ({e}). Check its docs; use mode C meanwhile.") from e
    flat = df.iloc[0].to_dict() if hasattr(df, "iloc") else dict(df)
    find = lambda key: next((v for k, v in flat.items() if key in str(k).lower()), None)  # noqa: E731
    return {"lut": find("lut"), "ff": find("ff"), "dsp": find("dsp"), "bram_18k": find("bram"),
            "latency_best_cycles": find("cycle")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=list(LABEL), default="C")
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--sparsities", nargs="*", type=float, default=None)
    ap.add_argument("--grid", action="store_true", help="all 27 configs instead of the pilot subset")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["estimate"]
    if args.archs:
        archs = args.archs
    else:
        archs = cfg["archs"] if args.grid else [cfg["archs"][0], cfg["archs"][-1]]
    bits_list = args.bits or (cfg["quant"]["bits"] if args.grid else ec["pilot_bits"])
    sps = args.sparsities if args.sparsities is not None else (
        cfg["prune"]["sparsities"] if args.grid else ec["pilot_sparsities"])

    if args.mode == "A" and not (shutil.which("vitis_hls") or shutil.which("vivado_hls")):
        raise SystemExit("Mode A needs vitis_hls/vivado_hls on PATH. Use --mode C or B.")

    rows = []
    for arch, bits, s in itertools.product(archs, bits_list, sps):
        qm = load_qmodel(arch, bits, cfg, s)
        row = {"arch": arch, "bits": bits, "sparsity": s, "estimate_mode": args.mode,
               "label": LABEL[args.mode], **proxy_row(qm, bits, cfg["quant"])}
        if args.mode in ("A", "B"):
            from .convert import convert_model

            hls_model, hcfg = convert_model(qm, ROOT / "hls_prj" / config_name(arch, bits, s), cfg["hls"])
            row.update(synth_row(hls_model) if args.mode == "A" else rule4ml_row(qm, hcfg))
        rows.append(row)
        print(f"[{arch} | {bits}b | s={s:.2f}] mult {row['n_mult']} bits {row['total_weight_bits']} "
              f"BOPs {row['bops']}")

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"resource_estimates_{args.mode}.csv", index=False)

    # sanity: zeroed weights should reduce multipliers (check, don't assume)
    for (arch, bits), g in df.groupby(["arch", "bits"]):
        g = g.sort_values("sparsity")
        if not g["n_mult"].is_monotonic_decreasing:
            print(f"WARNING: multipliers do not decrease with sparsity for {arch} {bits}b: "
                  f"{dict(zip(g['sparsity'], g['n_mult']))}")


if __name__ == "__main__":
    main()
