"""Needs hls4ml and a C++ compiler, but NOT Vivado/Vitis (no synthesis).

Guards the conversion path: convert, compile, predict, and Keras-vs-hls4ml agreement. The strict
"99% of outputs within 1 output LSB" flag is what `python -m src.convert` reports for trained models;
random (untrained) weights land right at that line, so this test asserts the failure modes seen before
the precision fixes instead: large element differences (3 to 8.6) and AUC mismatches (up to 0.14).
"""
import numpy as np
import tensorflow as tf

from src.common import load_config
from src.convert import check_config

CFG = load_config()


def test_convert_compile_predict_and_agree(tmp_path):
    tf.keras.utils.set_random_seed(0)  # untrained weights: make the run repeatable
    rng = np.random.default_rng(0)
    data = {"X_test_qcd": rng.normal(size=(600, 16)).astype("float32"),
            "X_anom_w": (rng.normal(size=(300, 16)) * 1.5).astype("float32")}
    cfg = {**CFG, "hls": {**CFG["hls"], "n_check": 400}}
    r = check_config("16-3-16", 8, 0.0, cfg, data, require_weights=False, out_root=tmp_path)
    assert r["max_abs_diff"] < 1.0, r
    assert r["frac_within_atol"] > 0.95, r
    assert abs(r["auc_diff"]) < 0.01, r
