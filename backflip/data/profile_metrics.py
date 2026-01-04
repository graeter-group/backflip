# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.


import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict

def get_metrics(predictions:Dict[str, List[np.ndarray]], targets:Dict[str, List[np.ndarray]], profile_types:List[str]) -> Dict[str, Dict[str, float]]:
    '''
    Expects targets and predictions to be dictionaries with {target_type: list of numpy arrays with shape (*batch,num_res).}
    '''
    metrics = {}
    for profile_type in profile_types:
        if profile_type == "None":
            continue
        assert all([profile_type in d.keys() for d in [predictions, targets]])
        metrics[profile_type] = get_profile_metrics(targets[profile_type], predictions[profile_type])

    return metrics

def get_profile_metrics(target:List[np.ndarray], prediction:List[np.ndarray]) -> Dict[str, float]:
    '''
    Expects target and prediction to be lists of 1d numpy arrays, each of which representing prediction and target for a single protein.
    Assumes that the individual arrays are shape (num_residues,).
    '''
    assert isinstance(target, list) and isinstance(prediction, list), f"Expected target and prediction to be lists, got {type(target)} and {type(prediction)}"
    assert len(target) == len(prediction), f"Expected target and prediction to have the same length, got {len(target)} and {len(prediction)}"
    assert all([len(t) == len(p) for t, p in zip(target, prediction)]), f"Expected target and prediction to have the same length, got {[len(t) for t in target]} and {[len(p) for p in prediction]}"
    assert all([isinstance(t, np.ndarray) and isinstance(p, np.ndarray) for t, p in zip(target, prediction)]), f"Expected target and prediction to be lists of numpy arrays, got {set([type(t) for t in target])} and {set([type(p) for p in prediction])}"
    # assert the entries are floats:
    assert all([(t.dtype == np.float32 or t.dtype == np.float64) and (p.dtype == np.float32 or p.dtype == np.float64) for t, p in zip(target, prediction)]), f"Expected target and prediction to be lists of numpy arrays of dtype float32, got {[t.dtype for t in target]} and {[p.dtype for p in prediction]}"

    if any([np.isnan(t).any() for t in target]):
        raise ValueError("Target contains NaN values.")
    if any([np.isnan(p).any() for p in prediction]):
        raise ValueError("Prediction contains NaN values.")
    
    pearson_rs = [pearsonr(t, p)[0] for t, p in zip(target, prediction)]
    per_target_pearson_R = np.median(pearson_rs)

    per_target_mae = np.median([mean_absolute_error(t, p) for t, p in zip(target, prediction)])

    per_target_avg_r = np.mean(pearson_rs)
    per_target_avg_mae = np.mean([mean_absolute_error(t, p) for t, p in zip(target, prediction)])

    orig_target = target.copy()
    target = np.concatenate(target)
    prediction = np.concatenate(prediction)

    pearson_R, _ = pearsonr(target, prediction)
    mae = mean_absolute_error(target, prediction)

    # benchmark against a naive baseline that always predicts the same value (mean of the target)
    # mean flexibility across all proteins
    mean_flexibility = np.median([np.mean(t) for t in orig_target])
    mean_prediction_baseline = [np.full_like(t, mean_flexibility) for t in orig_target]
    # compute mae for the naive baseline
    per_target_baseline_mae = np.median([mean_absolute_error(t, mean_prediction_baseline[i]) for i, t in enumerate(orig_target)])
    global_mae_baseline = mean_absolute_error(target, np.full_like(target, mean_flexibility))
    per_target_avg_mae_baseline = np.mean([mean_absolute_error(t, mean_prediction_baseline[i]) for i, t in enumerate(orig_target)])

    epoch_metrics = {
        "per-target-avg-r": per_target_avg_r,
        "per-target-avg-mae": per_target_avg_mae,
        "num-proteins": len(orig_target),
        "per-target-median-r": per_target_pearson_R,
        "per-target-median-mae": per_target_mae,
        "global-r": pearson_R,
        "global-mae": mae,
        "mean-prediction-per-target-avg-mae": per_target_avg_mae_baseline,
        "mean-prediction-per-target-mae": per_target_baseline_mae,
        "mean-prediction-global-mae": global_mae_baseline,
    }

    return epoch_metrics