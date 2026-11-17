"""Phase 5: QKeras -> hls4ml conversion, C-simulation (compile) and Keras-vs-hls4ml check.

No Vivado/Vitis needed: compile() builds the C-sim library (needs a C++ compiler, e.g. g++).
Weight/bias/activation precision is inferred from the QKeras quantizers. Two things hls4ml does
NOT carry over by itself are fixed in apply_precision_fixes():
  * accumulators default to ap_fixed<16,6>, which wraps around at +-32
  * linear QActivation layers (q_latent, q_output) are dropped, so their rounding/saturation
    is lost; the preceding Dense layer's result type is set to the quantizer's type instead

    python -m src.convert                       # first bit-width, no sparsity, all archs
    python -m src.convert --grid                # full arch x bits x sparsity grid
    python -m src.convert --archs 16-3-16 --bits 8 --sparsities 0.5
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from .common import ROOT, anomaly_arrays, load_config, load_processed
from .model_io import config_name, load_qmodel
from .roc_auc import auc

LINEAR_QUANTIZERS = ("q_latent", "q_output")


def _prec_dict(hcfg: dict, layer_name: str) -> dict:
    entry = hcfg["LayerName"][layer_name]
    p = entry.get("Precision")
    if not isinstance(p, dict):
        p = {"result": p} if p else {}
        entry["Precision"] = p
    return p


def apply_precision_fixes(qmodel, hcfg: dict, hls_cfg: dict) -> dict:
    accum = hls_cfg.get("accum_precision", "fixed<32,16>")
    layers = list(qmodel.layers)
    for i, l in enumerate(layers):
        kind = l.__class__.__name__
        if kind == "QDense":
            _prec_dict(hcfg, l.name)["accum"] = accum
        elif kind == "QActivation" and l.name in LINEAR_QUANTIZERS:
            q = l.activation  # quantized_bits instance
            prec = f"fixed<{int(q.bits)},{int(q.integer) + 1},RND_CONV,SAT,0>"
            _prec_dict(hcfg, layers[i - 1].name)["result"] = prec
    return hcfg


def convert_model(qmodel, out_dir, hls_cfg: dict, trace: bool = False):
    """Return (hls_model, hls_config dict). Per-layer ('name') granularity."""
    import hls4ml

    hcfg = hls4ml.utils.config_from_keras_model(qmodel, granularity="name")
    hcfg["Model"]["Strategy"] = hls_cfg.get("strategy", "Latency")
    hcfg["Model"]["ReuseFactor"] = hls_cfg.get("reuse_factor", 1)
    apply_precision_fixes(qmodel, hcfg, hls_cfg)
    if trace:
        for lname in hcfg["LayerName"]:
            hcfg["LayerName"][lname]["Trace"] = True
    hls_model = hls4ml.converters.convert_from_keras_model(
        qmodel, hls_config=hcfg, output_dir=str(out_dir),
        backend=hls_cfg["backend"], part=hls_cfg["part"],
        io_type=hls_cfg.get("io_type", "io_parallel"))
    return hls_model, hcfg


def describe_precisions(hls_model) -> list:
    """Best-effort per-layer precision dump (hls4ml internals differ between versions)."""
    out = []
    for layer in hls_model.get_layers():
        entry = {"layer": layer.name, "class": layer.__class__.__name__}
        for group in ("weights", "variables"):
            try:
                for k, v in getattr(layer, group).items():
                    entry[f"{group}.{k}"] = str(v.type.precision)
            except Exception:  # noqa: BLE001
                pass
        out.append(entry)
    return out


def agreement(y_ref: np.ndarray, y_hls: np.ndarray, atol: float, frac_required: float) -> dict:
    y_hls = np.asarray(y_hls).reshape(y_ref.shape)
    d = np.abs(y_ref - y_hls)
    frac = float((d <= atol + 1e-9).mean())
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "p99_abs_diff": float(np.percentile(d, 99)),
            "frac_within_atol": frac, "atol": atol, "agrees": frac >= frac_required}


def check_config(arch, bits, sparsity, cfg, data, require_weights=True, out_root=None):
    qc, hc = cfg["quant"], cfg["hls"]
    qmodel = load_qmodel(arch, bits, cfg, sparsity, require_weights=require_weights)
    name = config_name(arch, bits, sparsity)
    out_dir = (out_root or ROOT / "hls_prj") / name
    hls_model, hcfg = convert_model(qmodel, out_dir, hc)
    (ROOT / "results" / "hls_configs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "hls_configs" / f"{name}.json").write_text(
        json.dumps({"hls_config": hcfg, "precisions": describe_precisions(hls_model)},
                   indent=2, default=str))
    hls_model.compile()

    rng = np.random.default_rng(cfg["seed"])
    n = hc["n_check"]

    def sample(X, m):
        return X[rng.choice(len(X), size=min(m, len(X)), replace=False)]

    qcd = sample(data["X_test_qcd"], n)
    anoms = {k: sample(v, n // 2) for k, v in anomaly_arrays(data).items()}

    def both(X):
        yk = qmodel.predict(X, batch_size=8192, verbose=0)
        yh = np.asarray(hls_model.predict(np.ascontiguousarray(X))).reshape(yk.shape)
        return yk, yh

    yk_q, yh_q = both(qcd)
    lsb = 2.0 ** -(qc["out_bits"] - qc["out_int"] - 1)
    agr = agreement(yk_q, yh_q, hc["agree_atol_lsb"] * lsb, hc["agree_frac"])

    mse = lambda X, Y: np.mean((X - Y) ** 2, axis=1)  # noqa: E731
    sk_q, sh_q = mse(qcd, yk_q), mse(qcd, yh_q)
    pooled_k, pooled_h = [], []
    for X in anoms.values():
        yk, yh = both(X)
        pooled_k.append(mse(X, yk))
        pooled_h.append(mse(X, yh))
    auc_k = auc(sk_q, np.concatenate(pooled_k))
    auc_h = auc(sh_q, np.concatenate(pooled_h))
    return {"arch": arch, "bits": bits, "sparsity": sparsity, "backend": hc["backend"],
            **agr, "auc_pooled_keras": auc_k, "auc_pooled_hls4ml": auc_h,
            "auc_diff": auc_k - auc_h}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--sparsities", nargs="*", type=float, default=None)
    ap.add_argument("--grid", action="store_true", help="full arch x bits x sparsity grid")
    args = ap.parse_args()
    cfg = load_config(args.config)
    archs = args.archs or cfg["archs"]
    bits_list = args.bits or (cfg["quant"]["bits"] if args.grid else cfg["quant"]["bits"][:1])
    sps = args.sparsities if args.sparsities is not None else (
        cfg["prune"]["sparsities"] if args.grid else [0.0])
    data, _ = load_processed(cfg)
    (ROOT / "results").mkdir(exist_ok=True)

    rows = []
    for arch, bits, s in itertools.product(archs, bits_list, sps):
        r = check_config(arch, bits, s, cfg, data)
        rows.append(r)
        print(f"[{arch} | {bits}b | s={s:.2f}] max|diff| {r['max_abs_diff']:.4f} "
              f"p99 {r['p99_abs_diff']:.4f} within-tol {r['frac_within_atol']:.2%} "
              f"agrees={r['agrees']} AUC keras {r['auc_pooled_keras']:.3f} vs hls4ml "
              f"{r['auc_pooled_hls4ml']:.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "hls_agreement.csv", index=False)


if __name__ == "__main__":
    main()
