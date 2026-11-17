"""Diagnostic: how often do values exceed each quantizer's range?

QKeras saturates (clips) at the quantizer range; fixed-point hardware may wrap around instead.
Reports, per QActivation layer, the percentage of QCD-test values outside the representable
range (relu layers: >= 2**act_int; latent / input / output: |x| >= 2**int).

    python -m src.range_check
    python -m src.range_check --archs 16-8-4-3-4-8-16 --bits 8 --sparsities 0 0.75
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from .common import ROOT, load_config, load_processed
from .model_io import load_qmodel, weights_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--sparsities", nargs="*", type=float, default=None)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qc = cfg["quant"]
    archs = args.archs or cfg["archs"]
    bits_list = args.bits or qc["bits"]
    sps = args.sparsities if args.sparsities is not None else cfg["prune"]["sparsities"]

    import tensorflow as tf
    from qkeras import QActivation

    data, _ = load_processed(cfg)
    X = data["X_test_qcd"][: args.n]
    bound = {"q_input": 2.0 ** qc["input_int"], "q_latent": 2.0 ** qc["latent_int"],
             "q_output": 2.0 ** qc["out_int"]}

    rows = []
    for arch, bits, s in itertools.product(archs, bits_list, sps):
        if not weights_path(arch, bits, s).exists():
            continue
        qm = load_qmodel(arch, bits, cfg, s)
        acts = [l for l in qm.layers if isinstance(l, QActivation)]
        probe = tf.keras.Model(qm.input, [l.input for l in acts])
        outs = probe.predict(X, batch_size=8192, verbose=0)
        row = {"arch": arch, "bits": bits, "sparsity": s}
        for l, o in zip(acts, outs):
            if l.name.startswith("relu"):
                pct = float((o >= 2.0 ** qc["act_int"]).mean() * 100)
            else:
                pct = float((np.abs(o) >= bound[l.name]).mean() * 100)
            row[l.name] = round(pct, 2)
            row[l.name + "_max"] = float(np.abs(o).max())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "range_check.csv", index=False)
    show = ["arch", "bits", "sparsity"] + [c for c in df.columns
                                           if c.startswith(("relu", "q_")) and not c.endswith("_max")]
    print("% of values at/over the quantizer range (QCD test jets):")
    print(df[show].fillna("-").to_string(index=False))


if __name__ == "__main__":
    main()
