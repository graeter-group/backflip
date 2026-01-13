# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import numpy as np
from scipy.stats import pearsonr, spearmanr


def rmse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pearson_corr(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.size < 2:
        return float('nan')
    return float(pearsonr(pred, target)[0])


def spearman_corr(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.size < 2:
        return float('nan')
    return float(spearmanr(pred, target)[0])
