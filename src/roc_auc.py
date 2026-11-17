"""ROC / AUC / TPR@FPR utilities. Convention: higher score = more anomalous."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos]))


def tpr_at_fpr(neg: np.ndarray, pos: np.ndarray, fpr: float) -> float:
    thr = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thr))


def evaluate_scores(s_qcd: np.ndarray, s_anom: dict, fprs) -> dict:
    """Per-class and pooled AUC, plus TPR at fixed FPRs. Keys: auc_<cls>, tpr_fpr<f>_<cls>."""
    out = {}
    pooled = np.concatenate(list(s_anom.values()))
    groups = {**s_anom, "pooled": pooled}
    for k, s in groups.items():
        out[f"auc_{k}"] = auc(s_qcd, s)
        for f in fprs:
            out[f"tpr_fpr{f:g}_{k}"] = tpr_at_fpr(s_qcd, s, f)
    return out


def plot_roc(s_qcd: np.ndarray, s_anom: dict, path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k, s in s_anom.items():
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(s_qcd)), np.ones(len(s))], np.r_[s_qcd, s])
        ax.plot(fpr, tpr, label=f"{k} (AUC {auc(s_qcd, s):.3f})")
    ax.plot([1e-4, 1], [1e-4, 1], "k:", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("QCD false-positive rate")
    ax.set_ylabel("Anomaly true-positive rate")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
