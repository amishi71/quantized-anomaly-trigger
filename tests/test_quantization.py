import numpy as np
import pytest
import tensorflow as tf
from qkeras import QDense

from src.autoencoder import parse_arch
from src.common import load_config
from src.quantize import build_qae

Q = load_config()["quant"]


@pytest.mark.parametrize("bits", [8, 6, 4])
def test_weights_on_quantizer_grid(bits):
    m = build_qae(parse_arch("16-4-2-4-16"), bits, Q)
    rng = np.random.default_rng(0)
    frac = bits - Q["w_int"] - 1
    step, lim = 2.0 ** -frac, 2.0 ** Q["w_int"]
    n = 0
    for layer in m.layers:
        if not isinstance(layer, QDense):
            continue
        k, b = layer.get_weights()
        layer.set_weights([rng.normal(0, 3, k.shape), rng.normal(0, 3, b.shape)])  # force clipping
        kq_fn, bq_fn = layer.get_quantizers()
        for q_fn, w in ((kq_fn, layer.get_weights()[0]), (bq_fn, layer.get_weights()[1])):
            wq = q_fn(tf.constant(w, dtype=tf.float32)).numpy()
            assert np.allclose(wq / step, np.round(wq / step), atol=1e-5)
            assert wq.min() >= -lim - 1e-6 and wq.max() <= lim - step + 1e-6
            assert len(np.unique(wq)) <= 2 ** bits
        n += 1
    assert n == 4


def test_output_on_output_grid():
    m = build_qae(parse_arch("16-3-16"), 6, Q)
    x = np.random.default_rng(1).normal(0, 2, (256, 16)).astype("float32")
    y = m.predict(x, verbose=0)
    step = 2.0 ** -(Q["out_bits"] - Q["out_int"] - 1)
    assert np.allclose(y / step, np.round(y / step), atol=1e-4)
    assert y.min() >= -(2.0 ** Q["out_int"]) - 1e-6
