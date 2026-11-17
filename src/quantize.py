"""Phase 3: QKeras quantization (8/6/4-bit) of the FP32 baselines.

Workflow per (arch, bits): load FP32 -> build QKeras twin -> copy weights ->
evaluate post-training quantization (PTQ) -> fine-tune -> evaluate.

    python -m src.quantize
"""
from __future__ import annotations

import argparse

import pandas as pd

from .autoencoder import latent_index, parse_arch
from .common import ROOT, evaluate_model, load_config, load_processed, set_seeds
from .train import fit_autoencoder


def build_qae(widths, bits: int, q: dict, name: str | None = None, dense_wrapper=None):
    """QKeras twin of build_ae. alpha=1 everywhere for hls4ml compatibility.

    dense_wrapper: optional callable applied to each QDense (used by prune.py to attach pruning).
    """
    import tensorflow as tf
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    def qb(b, i):
        return quantized_bits(b, i, alpha=1)

    def qdense(units, lname):
        layer = QDense(units, kernel_quantizer=qb(bits, q["w_int"]),
                       bias_quantizer=qb(bits, q["w_int"]),
                       kernel_initializer="lecun_uniform", name=lname)
        return dense_wrapper(layer) if dense_wrapper else layer

    inp = tf.keras.layers.Input(shape=(widths[0],), name="inp")
    x = QActivation(qb(q["input_bits"], q["input_int"]), name="q_input")(inp)
    lat = latent_index(widths)
    act_bits = q.get("act_bits") or bits  # activations/latent may use a fixed width
    for i, w in enumerate(widths[1:-1]):
        x = qdense(w, f"fc{i + 1}")(x)
        if i == lat:  # linear latent -> signed quantizer
            x = QActivation(qb(act_bits, q["latent_int"]), name="q_latent")(x)
        else:
            x = QActivation(quantized_relu(act_bits, q["act_int"]), name=f"relu{i + 1}")(x)
    x = qdense(widths[-1], "output")(x)
    x = QActivation(qb(q["out_bits"], q["out_int"]), name="q_output")(x)
    return tf.keras.Model(inp, x, name=name or f"qae_{bits}b_" + "_".join(map(str, widths)))


def transfer_weights(fp, qmodel, bits: int, w_int: int) -> float:
    """Copy FP32 weights into the QKeras model; return fraction outside the quantizer range."""
    fpl = [l for l in fp.layers if l.get_weights()]
    ql = [l for l in qmodel.layers if l.get_weights()]
    assert len(fpl) == len(ql), "layer mismatch between FP32 and QKeras models"
    lim, step = 2.0 ** w_int, 2.0 ** -(bits - w_int - 1)
    n_clip = n_tot = 0
    for a, b in zip(fpl, ql):
        ws = a.get_weights()
        b.set_weights(ws)
        for w in ws:
            n_clip += int(((w < -lim) | (w > lim - step)).sum())
            n_tot += w.size
    return n_clip / max(n_tot, 1)


def _plot_auc_vs_bits(df: pd.DataFrame, fp: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (arch, g) in enumerate(df.groupby("arch")):
        g = g.sort_values("bits")
        c = f"C{i}"
        ax.plot(g["bits"], g["auc_pooled"], "o-", color=c, label=arch)
        ax.axhline(fp[arch], color=c, ls="--", lw=0.8)
    ax.set_xlabel("weight bits")
    ax.set_ylabel("pooled AUC (dashed = FP32)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--bits", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    qc, fprs = cfg["quant"], cfg["eval"]["fprs"]
    archs, bits_list = args.archs or cfg["archs"], args.bits or qc["bits"]

    import tensorflow as tf

    data, meta = load_processed(cfg)
    res, qdir = ROOT / "results", ROOT / "models" / "qkeras"
    qdir.mkdir(parents=True, exist_ok=True)
    classes = [k for k in ("pooled", *[c for c in cfg["data"]["anomaly_classes"]])]

    rows, pf_rows, fp_pooled = [], [], {}
    for arch in archs:
        widths = parse_arch(arch)
        fp = tf.keras.models.load_model(ROOT / "models" / "fp32" / f"{arch}.h5", compile=False)
        fp_m, _, _ = evaluate_model(fp, data, fprs)
        fp_pooled[arch] = fp_m["auc_pooled"]
        for bits in bits_list:
            set_seeds(cfg["seed"])
            qm = build_qae(widths, bits, qc)
            clipped = transfer_weights(fp, qm, bits, qc["w_int"])
            ptq, _, _ = evaluate_model(qm, data, fprs)
            fit_autoencoder(qm, data, qc["finetune_epochs"], cfg["train"]["batch_size"],
                            qc["lr"], qc["patience"], verbose=0)
            m, pf, _ = evaluate_model(qm, data, fprs)
            qm.save_weights(qdir / f"{arch}_b{bits}.h5")

            row = {"arch": arch, "bits": bits, "n_params": qm.count_params(),
                   "frac_fp32_weights_clipped": clipped,
                   "n_feat_worse_than_mean": m["n_feat_worse_than_mean"]}
            for k in classes:
                row[f"auc_fp32_{k}"] = fp_m[f"auc_{k}"]
                row[f"auc_ptq_{k}"] = ptq[f"auc_{k}"]
                row[f"auc_{k}"] = m[f"auc_{k}"]
                row[f"auc_drop_{k}"] = fp_m[f"auc_{k}"] - m[f"auc_{k}"]
            for f in fprs:
                row[f"tpr_fpr{f:g}_pooled"] = m[f"tpr_fpr{f:g}_pooled"]
            row["meets_drop_target"] = all(
                row[f"auc_drop_{k}"] < qc["auc_drop_target"] for k in classes)
            rows.append(row)
            pf_rows.append({"arch": arch, "bits": bits, **dict(zip(meta["feature_names"], pf))})
            print(f"[{arch} | {bits}b] pooled AUC fp32 {fp_m['auc_pooled']:.3f} "
                  f"ptq {ptq['auc_pooled']:.3f} qat {m['auc_pooled']:.3f} "
                  f"clipped {clipped:.1%} target_met={row['meets_drop_target']}")

    df = pd.DataFrame(rows)
    df.to_csv(res / "quantization_table.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(res / "quantization_per_feature_mse.csv", index=False)
    _plot_auc_vs_bits(df, fp_pooled, res / "auc_vs_bits.png")


if __name__ == "__main__":
    main()
