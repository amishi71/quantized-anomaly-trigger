"""Phase 4: magnitude (unstructured) pruning on top of the QKeras models.

Pruning is unstructured magnitude pruning (tfmot prune_low_magnitude, polynomial schedule
to the target sparsity). Quantizers stay in the loop, so the pruned model is still
quantization-aware. Kernels are pruned; biases are not.

    python -m src.prune                      # all archs x bits x sparsities in the config
    python -m src.prune --order-check        # sequential vs joint, on one config
"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from .autoencoder import expected_params, parse_arch
from .common import ROOT, evaluate_model, load_config, load_processed, set_seeds
from .model_io import load_qmodel, quantized_param_counts, weights_path
from .quantize import build_qae, transfer_weights


def prune_and_finetune(src, widths, bits, qcfg, data, sparsity, epochs, hold_epochs,
                       batch_size, lr, frequency, verbose=0):
    """Copy weights from `src` (a QKeras model) into a pruned twin, train, strip wrappers."""
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    sp = tfmot.sparsity.keras
    steps = math.ceil(len(data["X_train"]) / batch_size)
    end_step = max(1, steps * max(epochs - hold_epochs, 1))
    sched = sp.PolynomialDecay(initial_sparsity=0.0, final_sparsity=sparsity,
                               begin_step=0, end_step=end_step, frequency=frequency)
    pm = build_qae(widths, bits, qcfg,
                   dense_wrapper=lambda l: sp.prune_low_magnitude(l, pruning_schedule=sched))
    src_layers = [l for l in src.layers if l.get_weights()]
    wrapped = [l for l in pm.layers if type(l).__name__ == "PruneLowMagnitude"]
    assert len(src_layers) == len(wrapped), "layer mismatch when copying into pruned model"
    for a, b in zip(src_layers, wrapped):
        b.layer.set_weights(a.get_weights())
    pm.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    pm.fit(data["X_train"], data["X_train"], validation_data=(data["X_val"], data["X_val"]),
           epochs=epochs, batch_size=batch_size, callbacks=[sp.UpdatePruningStep()],
           verbose=verbose)
    return sp.strip_pruning(pm)


def raw_kernel_sparsity(model) -> float:
    """Fraction of exactly-zero kernel entries stored in the model (before quantizer rounding)."""
    from qkeras import QDense

    z = n = 0
    for l in model.layers:
        if isinstance(l, QDense):
            k = l.get_weights()[0]
            z += int((k == 0).sum())
            n += k.size
    return z / max(n, 1)


def prune_one(fp, qbase, arch, bits, sparsity, start, data, cfg):
    widths, pc, qc = parse_arch(arch), cfg["prune"], cfg["quant"]
    set_seeds(cfg["seed"])
    if start == "quantized":
        src = qbase
    else:  # joint: start from FP32 weights, quantization + pruning learned together
        src = build_qae(widths, bits, qc)
        transfer_weights(fp, src, bits, qc["w_int"])
    return prune_and_finetune(src, widths, bits, qc, data, sparsity, pc["epochs"],
                              pc["hold_epochs"], cfg["train"]["batch_size"], pc["lr"],
                              pc["frequency"])


def order_check(cfg, data, arch, bits, sparsity):
    import tensorflow as tf

    fprs = cfg["eval"]["fprs"]
    fp = tf.keras.models.load_model(ROOT / "models" / "fp32" / f"{arch}.h5", compile=False)
    qbase = load_qmodel(arch, bits, cfg)
    rows = []
    for start in ("quantized", "fp32"):
        m = prune_one(fp, qbase, arch, bits, sparsity, start, data, cfg)
        met, _, _ = evaluate_model(m, data, fprs)
        rows.append({"arch": arch, "bits": bits, "sparsity": sparsity, "start": start,
                     "auc_pooled": met["auc_pooled"],
                     "raw_kernel_sparsity": raw_kernel_sparsity(m)})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "prune_order_check.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("Set prune.start in configs/default.yaml to the better order, then run the full sweep.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    ap.add_argument("--sparsities", nargs="*", type=float, default=None)
    ap.add_argument("--start", choices=["quantized", "fp32"], default=None)
    ap.add_argument("--order-check", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc, qc, fprs = cfg["prune"], cfg["quant"], cfg["eval"]["fprs"]
    archs, bits_list = args.archs or cfg["archs"], args.bits or qc["bits"]
    sparsities = args.sparsities if args.sparsities is not None else pc["sparsities"]
    start = args.start or pc["start"]

    import tensorflow as tf

    data, meta = load_processed(cfg)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models" / "pruned").mkdir(parents=True, exist_ok=True)

    if args.order_check:
        order_check(cfg, data, archs[0], bits_list[0], 0.5)
        return

    baseline_params = expected_params(parse_arch(cfg["archs"][0]))
    classes = ["pooled", *cfg["data"]["anomaly_classes"]]
    rows = []
    for arch in archs:
        fp = tf.keras.models.load_model(ROOT / "models" / "fp32" / f"{arch}.h5", compile=False)
        fp_m, _, _ = evaluate_model(fp, data, fprs)
        for bits in bits_list:
            qbase = load_qmodel(arch, bits, cfg)
            q_m, _, _ = evaluate_model(qbase, data, fprs)
            for s in sparsities:
                if s == 0:
                    model, m = qbase, q_m
                else:
                    model = prune_one(fp, qbase, arch, bits, s, start, data, cfg)
                    m, _, _ = evaluate_model(model, data, fprs)
                    model.save_weights(weights_path(arch, bits, s))
                c = quantized_param_counts(model)
                row = {"arch": arch, "bits": bits, "sparsity_target": s, "start": start if s else "n/a",
                       "n_params_dense": model.count_params(),
                       "raw_kernel_sparsity": raw_kernel_sparsity(model),
                       "kernel_sparsity_quantized": c["kernel_sparsity"],
                       "nnz_kernel": c["nnz_kernel"], "nnz_params": c["nnz_params"],
                       "total_weight_bits": c["nnz_params"] * bits,
                       "param_reduction_vs_baseline": 1 - c["nnz_params"] / baseline_params,
                       "bits_reduction_vs_fp32_baseline": 1 - c["nnz_params"] * bits / (baseline_params * 32),
                       "n_feat_worse_than_mean": m["n_feat_worse_than_mean"]}
                for k in classes:
                    row[f"auc_{k}"] = m[f"auc_{k}"]
                    row[f"auc_drop_vs_fp32_{k}"] = fp_m[f"auc_{k}"] - m[f"auc_{k}"]
                    row[f"auc_drop_vs_quant_{k}"] = q_m[f"auc_{k}"] - m[f"auc_{k}"]
                for f in fprs:
                    row[f"tpr_fpr{f:g}_pooled"] = m[f"tpr_fpr{f:g}_pooled"]
                row["meets_drop_target"] = all(
                    row[f"auc_drop_vs_fp32_{k}"] < qc["auc_drop_target"] for k in classes)
                rows.append(row)
                print(f"[{arch} | {bits}b | s={s:.2f}] AUC {m['auc_pooled']:.3f} "
                      f"nnz {c['nnz_params']} (raw sparsity {row['raw_kernel_sparsity']:.0%}, "
                      f"quantized {c['kernel_sparsity']:.0%}) bits {row['total_weight_bits']}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "pruning_table.csv", index=False)


if __name__ == "__main__":
    main()
