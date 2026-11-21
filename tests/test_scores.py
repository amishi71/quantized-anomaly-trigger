import numpy as np

from src.score_study import compute_scores, fit_reference, split_masks, split_metrics


def test_scores_shapes_and_norm_is_model_free():
    rng = np.random.default_rng(0)
    Xv, X = rng.normal(size=(500, 16)), rng.normal(size=(300, 16))
    mk = lambda A: (0.8 * A, A[:, :3])  # fake reconstruction and latent  # noqa: E731
    ref = fit_reference(Xv, *mk(Xv))
    s = compute_scores(X, *mk(X), ref)
    assert set(s) == {"mse", "mse_std", "latent", "norm"}
    assert all(v.shape == (300,) for v in s.values())
    assert np.allclose(s["norm"], (X ** 2).mean(axis=1))
    assert np.all(s["latent"] >= 0)


def test_dev_holdout_partition_and_metrics():
    rng = np.random.default_rng(1)
    anoms = {"w": rng.normal(size=(1000, 16)), "z": rng.normal(size=(800, 16))}
    m = split_masks(anoms, seed=42)
    assert all(0.4 < m[k].mean() < 0.6 for k in m)
    neg = rng.normal(size=5000)
    pos = {k: rng.normal(1.0, 1.0, size=500) for k in anoms}
    r = split_metrics(neg, pos)
    assert 0.65 < r["auc_pooled"] < 0.85 and 0 <= r["tpr_fpr0.001_pooled"] <= r["tpr_fpr0.01_pooled"] <= 1
