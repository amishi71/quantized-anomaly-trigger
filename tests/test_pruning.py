import numpy as np
import tensorflow_model_optimization as tfmot

from src.autoencoder import parse_arch
from src.common import load_config
from src.model_io import quantized_param_counts
from src.prune import prune_and_finetune, raw_kernel_sparsity
from src.quantize import build_qae

CFG = load_config()


def test_pruning_reaches_target_and_strips_wrappers():
    rng = np.random.default_rng(0)
    data = {"X_train": rng.normal(size=(256, 16)).astype("float32"),
            "X_val": rng.normal(size=(64, 16)).astype("float32")}
    widths = parse_arch("16-8-4-2-4-8-16")
    src = build_qae(widths, 8, CFG["quant"])
    m = prune_and_finetune(src, widths, 8, CFG["quant"], data, sparsity=0.5, epochs=4,
                           hold_epochs=1, batch_size=64, lr=1e-3, frequency=1)
    assert not any(type(l).__name__ == "PruneLowMagnitude" for l in m.layers)
    s = raw_kernel_sparsity(m)
    assert 0.45 <= s <= 0.6, s
    c = quantized_param_counts(m)
    assert c["nnz_kernel"] <= 0.55 * c["total_kernel"]
