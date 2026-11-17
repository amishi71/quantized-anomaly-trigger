"""FP32 autoencoder factory. Architectures are strings like '16-8-4-2-4-8-16'."""
from __future__ import annotations

from typing import List


def parse_arch(arch: str) -> List[int]:
    widths = [int(x) for x in arch.split("-")]
    if len(widths) < 3 or widths[0] != widths[-1]:
        raise ValueError(f"Bad architecture '{arch}': need >=3 widths, input == output")
    return widths


def expected_params(widths: List[int]) -> int:
    return sum(i * o + o for i, o in zip(widths[:-1], widths[1:]))


def latent_index(widths: List[int]) -> int:
    """Index (within the hidden layers) of the bottleneck: the middle hidden layer.
    It is linear (no ReLU) so no latent unit can die."""
    return len(widths[1:-1]) // 2


def build_ae(widths: List[int], name: str | None = None):
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = layers.Input(shape=(widths[0],), name="inp")
    x = inp
    lat = latent_index(widths)
    for i, w in enumerate(widths[1:-1]):
        x = layers.Dense(w, activation=None if i == lat else "relu",
                         kernel_initializer="lecun_uniform", name=f"fc{i + 1}")(x)
    out = layers.Dense(widths[-1], kernel_initializer="lecun_uniform", name="output")(x)
    return tf.keras.Model(inp, out, name=name or "ae_" + "_".join(map(str, widths)))
