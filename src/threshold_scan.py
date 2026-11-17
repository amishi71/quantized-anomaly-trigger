"""Pick thresholds on QCD-val scores for target FPRs; report test FPR and signal efficiency."""
from __future__ import annotations

import numpy as np
import pandas as pd


def scan(val_scores, test_qcd_scores, anom_scores: dict, fprs) -> pd.DataFrame:
    rows = []
    for f in fprs:
        thr = float(np.quantile(val_scores, 1.0 - f))
        row = {"target_fpr": f, "threshold": thr,
               "achieved_test_fpr": float(np.mean(test_qcd_scores > thr))}
        for k, s in anom_scores.items():
            row[f"eff_{k}"] = float(np.mean(s > thr))
        rows.append(row)
    return pd.DataFrame(rows)
