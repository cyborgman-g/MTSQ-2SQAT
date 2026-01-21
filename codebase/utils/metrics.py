import numpy as np
from typing import List

def compute_eer(
    labels: List[int],
    scores: List[float]
) -> float:
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    labels = np.array(labels)
    scores = np.array(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer * 100

def compute_min_dcf(
    labels: List[int],
    scores: List[float],
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0
) -> float:
    labels = np.array(labels)
    scores = np.array(scores)

    sorted_indices = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_indices]

    n_target = np.sum(labels == 1)
    n_nontarget = np.sum(labels == 0)

    if n_target == 0 or n_nontarget == 0:
        return 0.0

    p_miss = np.zeros(len(labels) + 1)
    p_fa = np.zeros(len(labels) + 1)

    p_miss[0] = 0
    p_fa[0] = 1

    for i in range(len(labels)):
        if labels_sorted[i] == 1:
            p_miss[i+1] = p_miss[i] + 1/n_target
            p_fa[i+1] = p_fa[i]
        else:
            p_miss[i+1] = p_miss[i]
            p_fa[i+1] = p_fa[i] - 1/n_nontarget

    dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return float(np.min(dcf) / c_def)
