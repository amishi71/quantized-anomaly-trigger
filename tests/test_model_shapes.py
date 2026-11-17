import pytest

from src.autoencoder import build_ae, expected_params, parse_arch

EXPECTED = {"16-8-4-2-4-8-16": 378, "16-4-2-4-16": 170, "16-3-16": 115}


@pytest.mark.parametrize("arch,n", EXPECTED.items())
def test_shapes_and_params(arch, n):
    widths = parse_arch(arch)
    assert expected_params(widths) == n
    m = build_ae(widths)
    assert m.input_shape == (None, 16)
    assert m.output_shape == (None, 16)
    assert m.count_params() == n


def test_output_layer_is_linear():
    m = build_ae(parse_arch("16-4-2-4-16"))
    assert m.get_layer("output").activation.__name__ == "linear"


def test_bad_arch_rejected():
    with pytest.raises(ValueError):
        parse_arch("16-4-8")


def test_latent_layer_is_linear():
    for arch, name in [("16-8-4-2-4-8-16", "fc3"), ("16-4-2-4-16", "fc2"), ("16-3-16", "fc1")]:
        m = build_ae(parse_arch(arch))
        assert m.get_layer(name).activation.__name__ == "linear", arch
