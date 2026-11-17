import numpy as np

from src.autoencoder import parse_arch
from src.common import load_config
from src.convert import agreement
from src.estimate_resources import proxy_row
from src.quantize import build_qae

CFG = load_config()


def test_zeroed_weights_reduce_multipliers_and_bops():
    m = build_qae(parse_arch("16-8-4-2-4-8-16"), 8, CFG["quant"])
    dense = proxy_row(m, 8, CFG["quant"])
    rng = np.random.default_rng(0)
    for l in m.layers:
        if l.get_weights():
            k, b = l.get_weights()
            k = k * (rng.random(k.shape) > 0.75)
            l.set_weights([k, b])
    sparse = proxy_row(m, 8, CFG["quant"])
    assert sparse["n_mult"] < dense["n_mult"]
    assert sparse["bops"] < dense["bops"]
    assert sparse["layer_depth"] == dense["layer_depth"] == 6


def test_bops_grow_with_bits():
    a = proxy_row(build_qae(parse_arch("16-3-16"), 4, CFG["quant"]), 4, CFG["quant"])
    b = proxy_row(build_qae(parse_arch("16-3-16"), 8, CFG["quant"]), 8, CFG["quant"])
    assert b["bops"] > a["bops"] and b["total_weight_bits"] > a["total_weight_bits"]


def test_agreement_logic():
    y = np.zeros((100, 4))
    assert agreement(y, y + 0.01, atol=0.03, frac_required=0.99)["agrees"]
    assert not agreement(y, y + 0.1, atol=0.03, frac_required=0.99)["agrees"]
