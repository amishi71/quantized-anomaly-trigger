import numpy as np

from src.preprocessing import Preprocessor


def test_fit_on_train_standardises_and_clips():
    rng = np.random.default_rng(0)
    X = rng.lognormal(0, 1.5, (5000, 4))
    X[0, 0] = 1e9  # outlier must be clipped
    p = Preprocessor((0.1, 99.9), log_idx=[1]).fit(X)
    Z = p.transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-4)
    assert np.allclose(Z.std(0), 1, atol=1e-3)
    assert np.isfinite(Z).all() and np.abs(Z).max() < 50


def test_no_leakage_from_other_splits():
    rng = np.random.default_rng(1)
    tr, other = rng.normal(0, 1, (1000, 3)), rng.normal(100, 5, (1000, 3))
    p = Preprocessor().fit(tr)
    assert np.allclose(p.mean, tr.mean(0), atol=0.05)
    assert p.transform(other).max() < 1e3  # transform works, stats unchanged
