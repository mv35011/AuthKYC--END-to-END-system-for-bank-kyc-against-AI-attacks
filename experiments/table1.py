import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_experiment_metrics(y_true, y_probs):
    """Calculates the exact metrics required for Exp 1: Table 3."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # False Negative Rate is 1 - True Positive Rate
    fnr = 1 - tpr

    # 1. Equal Error Rate (EER): The point where FAR (fpr) == FRR (fnr)
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]

    # 2. FAR at 1% FRR
    # Find the index where FRR is closest to 0.01 (1%)
    frr_1_percent_idx = np.nanargmin(np.absolute((fnr - 0.01)))
    far_at_1_frr = fpr[frr_1_percent_idx]

    return {
        "AUC": roc_auc,
        "EER": eer,
        "FAR@1%FRR": far_at_1_frr
    }