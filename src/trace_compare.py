"""Layer-by-layer Keras vs hls4ml comparison: where do the outputs first diverge?

Uses hls4ml's per-layer tracing (Trace=True for every layer) and a Keras probe model.

    python -m src.trace_compare --arch 16-8-4-3-4-8-16 --bits 8 --sparsity 0
    python -m src.trace_compare --arch 16-8-4-3-4-8-16 --bits 8 --sparsity 0.75   # an agreeing config
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .common import ROOT, load_config, load_processed
from .convert import convert_model
from .model_io import config_name, load_qmodel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    hc = cfg["hls"]
    arch = args.arch or cfg["archs"][0]

    import tensorflow as tf

    data, _ = load_processed(cfg)
    X = np.ascontiguousarray(data["X_test_qcd"][: args.n])
    qm = load_qmodel(arch, args.bits, cfg, args.sparsity)

    name = config_name(arch, args.bits, args.sparsity)
    hls_model, _ = convert_model(qm, ROOT / "hls_prj" / f"trace_{name}", hc, trace=True)
    hls_model.compile()
    _, hls_trace = hls_model.trace(X)

    names = [l.name for l in qm.layers if l.name in hls_trace]
    if not names:
        raise SystemExit(f"No matching layer names. hls trace keys: {list(hls_trace)}")
    probe = tf.keras.Model(qm.input, [qm.get_layer(n).output for n in names])
    outs = probe.predict(X, batch_size=8192, verbose=0)

    rows = []
    for n, k in zip(names, outs):
        h = np.asarray(hls_trace[n]).reshape(k.shape)
        d = np.abs(k - h)
        rows.append({"layer": n, "max_abs_diff": float(d.max()),
                     "pct_elements_differ": float(100 * (d > 1e-4).mean()),
                     "pct_jets_differ": float(100 * (d > 1e-4).any(axis=1).mean()),
                     "max_abs_value_keras": float(np.abs(k).max())})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"trace_{name}.csv", index=False)
    print(f"{arch} {args.bits}b s={args.sparsity}: {args.n} QCD jets")
    print(df.round(4).to_string(index=False))
    bad = df[df.max_abs_diff > 1e-3]
    print("\nFirst layer with a difference:", bad.iloc[0]["layer"] if len(bad) else "none")


if __name__ == "__main__":
    main()
