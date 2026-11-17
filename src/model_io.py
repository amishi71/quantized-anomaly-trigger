"""Paths, loading and parameter counting for the QKeras / pruned models."""
from __future__ import annotations

from .autoencoder import parse_arch
from .common import ROOT


def pct(sparsity: float) -> int:
    return int(round(sparsity * 100))


def weights_path(arch: str, bits: int, sparsity: float = 0.0):
    if sparsity == 0:
        return ROOT / "models" / "qkeras" / f"{arch}_b{bits}.h5"
    return ROOT / "models" / "pruned" / f"{arch}_b{bits}_s{pct(sparsity)}.h5"


def config_name(arch: str, bits: int, sparsity: float = 0.0) -> str:
    return f"{arch.replace('-', '_')}_b{bits}_s{pct(sparsity)}"


def load_qmodel(arch: str, bits: int, cfg: dict, sparsity: float = 0.0, require_weights: bool = True):
    from .quantize import build_qae

    qm = build_qae(parse_arch(arch), bits, cfg["quant"], name=config_name(arch, bits, sparsity))
    p = weights_path(arch, bits, sparsity)
    if p.exists():
        qm.load_weights(p)
    elif require_weights:
        raise FileNotFoundError(f"{p} not found. Run the earlier phase (quantize / prune) first.")
    return qm


def quantized_param_counts(model) -> dict:
    """Count non-zero weights AFTER applying each layer's quantizer (what hardware sees)."""
    import tensorflow as tf
    from qkeras import QDense

    layers, nk = [], 0
    out = {"nnz_kernel": 0, "total_kernel": 0, "nnz_bias": 0, "total_bias": 0}
    for l in model.layers:
        if not isinstance(l, QDense):
            continue
        kq, bq = l.get_quantizers()
        k, b = l.get_weights()
        kv = kq(tf.constant(k, dtype=tf.float32)).numpy()
        bv = bq(tf.constant(b, dtype=tf.float32)).numpy()
        nnz = int((kv != 0).sum())
        out["nnz_kernel"] += nnz
        out["total_kernel"] += k.size
        out["nnz_bias"] += int((bv != 0).sum())
        out["total_bias"] += b.size
        layers.append({"name": l.name, "nnz": nnz, "fan_in": int(k.shape[0]), "fan_out": int(k.shape[1])})
    out["layers"] = layers
    out["nnz_params"] = out["nnz_kernel"] + out["nnz_bias"]
    out["kernel_sparsity"] = 1.0 - out["nnz_kernel"] / max(out["total_kernel"], 1)
    return out
