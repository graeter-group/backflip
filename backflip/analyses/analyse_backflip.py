"""
Compares predicted per-residue covariances against ground truth and reports
RMSF, DCCM, and covariance-ellipsoid overlap metrics.

Example:
    python analyze_inference_with_ellipsoids.py \\
        --inference_folder /path/to/inference_results/npz \\
        --gt_npz_folder /path/to/ground_truth_dataset \\
        --output_csv metrics.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from backflip.data.flexibility_utils import batched_rmsf_from_covar

def compute_dccm_from_scalar_covar(C: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
      cov = trace-pooled NxN scalar coupling
      rmsd = sqrt(diag(cov))
      cov[cov == 0] = eps
      dccm = cov / (rmsd[:,None] * rmsd[None,:])

    C: (..., N, N) allowed, but must be square.
    """
    if C.shape[-1] != C.shape[-2]:
        raise ValueError(f"C must be square, got {C.shape}")

    cov = C.clone()
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    rmsd = torch.sqrt(diag)
    cov = torch.where(cov == 0, cov.new_tensor(eps), cov)
    denom = rmsd.unsqueeze(-1) * rmsd.unsqueeze(-2)
    dccm = cov / denom
    return dccm

def compute_kl_symvar(cov_true: torch.Tensor, cov_pred: torch.Tensor) -> torch.Tensor:
    """
    cov0, cov1: (..., 3, 3) SPD covariance matrices
    Returns scalar KL_symvar averaged over residues.
    """
    assert cov_true.shape == cov_pred.shape
    assert cov_true.ndim == 3 and cov_true.shape[-2:] == (3, 3)

    d = cov_true.shape[-1]
    cov_true_inv = torch.linalg.inv(cov_true)
    cov_pred_inv = torch.linalg.inv(cov_pred)

    t1 = torch.diagonal(cov_pred_inv @ cov_true, dim1=-2, dim2=-1).sum(-1)
    t2 = torch.diagonal(cov_true_inv @ cov_pred, dim1=-2, dim2=-1).sum(-1)

    kl_symvar_per_residue = 0.25 * (t1 + t2 - 2 * d)
    kl_symvar = kl_symvar_per_residue.mean()
    return kl_symvar.numpy()

def _sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:, None])) @ np.linalg.inv(P)
    return out

def _calc_rmwd(cov_true, cov_pred):
    emd_mean = (np.square(cov_true - cov_pred).sum(-1) ** 0.5) * 10
    try:
        emd_var = (
            np.trace(
                cov_true + cov_pred - 2 * _sqrtm(cov_true @ cov_pred),
                axis1=1,
                axis2=2,
            )
            ** 0.5
        ) * 10
    except Exception:
        emd_var = np.trace(cov_true) ** 0.5 * 10

    rmwd_trans = np.square(emd_mean).mean() ** 0.5
    rmwd_var = np.square(emd_var).mean() ** 0.5
    rmwd = np.sqrt(rmwd_trans ** 2 + rmwd_var ** 2)

    return {
        "rmwd": rmwd,
        "rmwd_trans": rmwd_trans,
        "rmwd_var": rmwd_var,
    }

def _symmetrize(A):
    return 0.5 * (A + np.swapaxes(A, -1, -2))

def project_to_spd(A, eps=1e-6):
    A = _symmetrize(np.asarray(A, dtype=float))
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return _symmetrize((vecs * vals[..., None, :]) @ np.swapaxes(vecs, -1, -2))

def normalize_covariance_shape(Sigma):
    det = np.linalg.det(Sigma)
    return Sigma / det[..., None, None] ** (1.0 / 3.0)

def fibonacci_sphere(n_dirs):
    i = np.arange(n_dirs)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n_dirs
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.stack([x, y, z], axis=-1)

def centered_covariance_ellipsoid_iou(
    Sigma_pred,
    Sigma_true,
    tau=1.0,
    n_dirs=8192,
    shape_normalized=False,
    eps=1e-6,
):
    """
    Estimates the IoU and Dice overlap between the Mahalanobis-distance-tau
    ellipsoids of two per-residue covariance matrices, by sampling directions
    on the unit sphere and comparing the ellipsoid radii along each direction.
    """
    Sigma_pred = project_to_spd(Sigma_pred, eps=eps)
    Sigma_true = project_to_spd(Sigma_true, eps=eps)

    batch_shape = np.broadcast_shapes(
        Sigma_pred.shape[:-2],
        Sigma_true.shape[:-2],
    )

    Sigma_pred = np.broadcast_to(Sigma_pred, batch_shape + (3, 3))
    Sigma_true = np.broadcast_to(Sigma_true, batch_shape + (3, 3))

    if shape_normalized:
        Sigma_pred = normalize_covariance_shape(Sigma_pred)
        Sigma_true = normalize_covariance_shape(Sigma_true)

    Prec_pred = np.linalg.inv(Sigma_pred)
    Prec_true = np.linalg.inv(Sigma_true)

    U = fibonacci_sphere(n_dirs)

    q_pred = np.einsum("di,...ij,dj->...d", U, Prec_pred, U)
    q_true = np.einsum("di,...ij,dj->...d", U, Prec_true, U)

    r_pred = np.sqrt(tau / q_pred)
    r_true = np.sqrt(tau / q_true)

    r_intersection = np.minimum(r_pred, r_true)
    r_union = np.maximum(r_pred, r_true)

    intersection_measure = np.mean(r_intersection ** 3, axis=-1)
    union_measure = np.mean(r_union ** 3, axis=-1)
    iou = intersection_measure / union_measure

    pred_measure = np.mean(r_pred ** 3, axis=-1)
    true_measure = np.mean(r_true ** 3, axis=-1)
    dice = 2.0 * intersection_measure / (pred_measure + true_measure)

    pred_volume = (4.0 / 3.0) * np.pi * (tau ** 1.5) * np.sqrt(np.linalg.det(Sigma_pred))
    true_volume = (4.0 / 3.0) * np.pi * (tau ** 1.5) * np.sqrt(np.linalg.det(Sigma_true))

    return {
        "iou": iou,
        "dice": dice,
        "volume_ratio": pred_volume / true_volume,
    }

def summarize_ellipsoid_metrics(per_res_cov_pred, per_res_cov_gt, n_dirs=8192, tau=1.0):
    ellipsoid_abs = centered_covariance_ellipsoid_iou(
        per_res_cov_pred,
        per_res_cov_gt,
        tau=tau,
        n_dirs=n_dirs,
        shape_normalized=False,
    )
    ellipsoid_shape = centered_covariance_ellipsoid_iou(
        per_res_cov_pred,
        per_res_cov_gt,
        tau=tau,
        n_dirs=n_dirs,
        shape_normalized=True,
    )

    return {
        "ellipsoid_iou_abs_mean": float(np.mean(ellipsoid_abs["iou"])),
        "ellipsoid_dice_abs_mean": float(np.mean(ellipsoid_abs["dice"])),
        "ellipsoid_log_volume_ratio_abs_mean": float(np.mean(np.abs(np.log(ellipsoid_abs["volume_ratio"])))),
        "ellipsoid_iou_shape_mean": float(np.mean(ellipsoid_shape["iou"])),
        "ellipsoid_dice_shape_mean": float(np.mean(ellipsoid_shape["dice"])),
    }

# The pairwise-coupling edge output was renamed from 'CA_covariance' to
# 'pairwise_couplings'; older dataset/inference npz files on disk may still
# use the old key, so accept either.
PAIRWISE_COUPLING_KEYS = ("pairwise_couplings", "CA_covariance")

def _load_first_present(npz_data, keys):
    for key in keys:
        if key in npz_data:
            return npz_data[key]
    raise KeyError(f"None of {keys} found in npz file (available keys: {list(npz_data.keys())})")

def quick_compute_metrics_covar_pred_with_ellipsoids(
    inference_folder: str,
    gt_npz_folder: str,
    n_dirs: int = 8192,
    tau: float = 1.0,
) -> pd.DataFrame:
    """
    Compares each predicted `*_pred.npz` in inference_folder against its
    ground-truth counterpart in npz_folder, and returns a per-protein
    DataFrame of RMSF/DCCM/ellipsoid overlap metrics.
    """
    metrics = {
        "npz_file": [],
        "corr_pred_rmsf_to_md": [],
        "mae_pred_rmsf_to_md": [],
        "corr_pred_rmsf_to_gt": [],
        "mae_pred_rmsf_to_gt": [],
        "rmwd_var_pred": [],
        "kl_symvar_pred": [],
        "dccm_ca_corr_pred": [],
        "dccm_ca_mae_pred": [],
        "ellipsoid_iou_abs_mean": [],
        "ellipsoid_dice_abs_mean": [],
        "ellipsoid_log_volume_ratio_abs_mean": [],
        "ellipsoid_iou_shape_mean": [],
        "ellipsoid_dice_shape_mean": [],
    }

    npzs_inference = [f for f in os.listdir(inference_folder) if f.endswith(".npz")]

    for npz_file in tqdm(npzs_inference, desc="Computing metrics"):
        npz_gt_path = os.path.join(gt_npz_folder, npz_file.replace("_pred.npz", ".npz"))
        npz_pred_path = os.path.join(inference_folder, npz_file)
        
        try:
            data_gt = np.load(npz_gt_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_gt_path}: {e}")
            continue
        
        data_pred = np.load(npz_pred_path, allow_pickle=True)

        per_res_cov_gt = torch.tensor(data_gt["per_res_covariance"], dtype=torch.float32)
        CA_cov_gt = torch.tensor(_load_first_present(data_gt, PAIRWISE_COUPLING_KEYS), dtype=torch.float32)
        md_rmsf = data_gt["global_rmsf"]
        
        per_res_cov_pred = torch.tensor(data_pred["per_res_covariance"], dtype=torch.float32)
        CA_cov_pred = torch.tensor(_load_first_present(data_pred, PAIRWISE_COUPLING_KEYS), dtype=torch.float32)

        if per_res_cov_pred.shape != per_res_cov_gt.shape:
            print(f"Shape mismatch for {npz_file}: predicted {per_res_cov_pred.shape}, ground truth {per_res_cov_gt.shape}. Skipping.")
            continue
        
        rmsf_pred = batched_rmsf_from_covar(per_res_cov_pred)
        rmsf_gt = batched_rmsf_from_covar(per_res_cov_gt)

        rmwd_var_pred_bbflow = _calc_rmwd(per_res_cov_gt.numpy(), per_res_cov_pred.numpy())["rmwd_var"]
        kl_symvar_pred = compute_kl_symvar(per_res_cov_gt, per_res_cov_pred)

        dccm_pred = compute_dccm_from_scalar_covar(CA_cov_pred * 100)
        dccm_gt = compute_dccm_from_scalar_covar(CA_cov_gt * 100)
        dccm_corr_pred = np.corrcoef(
            dccm_pred.flatten().cpu().numpy(),
            dccm_gt.flatten().cpu().numpy(),
        )[0, 1]
        dccm_mae = np.abs(
            dccm_pred.flatten().cpu().numpy() - dccm_gt.flatten().cpu().numpy()
        ).mean()

        ellipsoid_metrics = summarize_ellipsoid_metrics(
            per_res_cov_pred.cpu().numpy(),
            per_res_cov_gt.cpu().numpy(),
            n_dirs=n_dirs,
            tau=tau,
        )

        metrics["npz_file"].append(npz_file)
        metrics["corr_pred_rmsf_to_md"].append(np.corrcoef(rmsf_pred, md_rmsf)[0, 1])
        metrics["mae_pred_rmsf_to_md"].append(np.mean(np.abs(rmsf_pred - md_rmsf)))
        metrics["corr_pred_rmsf_to_gt"].append(np.corrcoef(rmsf_pred, rmsf_gt)[0, 1])
        metrics["mae_pred_rmsf_to_gt"].append(np.mean(np.abs(rmsf_pred - rmsf_gt)))
        metrics["rmwd_var_pred"].append(rmwd_var_pred_bbflow)
        metrics["kl_symvar_pred"].append(kl_symvar_pred)
        metrics["dccm_ca_corr_pred"].append(dccm_corr_pred)
        metrics["dccm_ca_mae_pred"].append(dccm_mae)

        for key, value in ellipsoid_metrics.items():
            metrics[key].append(value)

    return pd.DataFrame(metrics)

def report_metrics_with_ellipsoids(metrics_df: pd.DataFrame, name: str = "inference"):
    print(f"--- Metrics report for {name} ---")
    print(f"RMSF correlation to MD: {metrics_df['corr_pred_rmsf_to_md'].median():.3f}")
    print(f"RMSF MAE to MD: {metrics_df['mae_pred_rmsf_to_md'].median():.3f}")
    print(f"RMSF correlation to GT: {metrics_df['corr_pred_rmsf_to_gt'].median():.3f}")
    print(f"RMSF MAE to GT: {metrics_df['mae_pred_rmsf_to_gt'].median():.3f}")
    print(f"RMWD variance component: {np.median([metrics_df['rmwd_var_pred']]):.3f}")
    print(f"KL divergence of variances: {np.median([metrics_df['kl_symvar_pred']]):.3f}")
    print(f"DCCM CA Pearson r: {metrics_df['dccm_ca_corr_pred'].median():.3f}")
    print(f"DCCM CA MAE: {metrics_df['dccm_ca_mae_pred'].median():.3f}")
    print(f"Ellipsoid IoU absolute volume: {metrics_df['ellipsoid_iou_abs_mean'].median():.3f}")
    print(f"Ellipsoid Dice absolute volume: {metrics_df['ellipsoid_dice_abs_mean'].median():.3f}")
    print(f"Ellipsoid |log volume ratio| absolute: {metrics_df['ellipsoid_log_volume_ratio_abs_mean'].median():.3f}")
    print(f"Ellipsoid IoU shape-normalized: {metrics_df['ellipsoid_iou_shape_mean'].median():.3f}")
    print(f"Ellipsoid Dice shape-normalized: {metrics_df['ellipsoid_dice_shape_mean'].median():.3f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-residue covariance metrics (RMSF, DCCM, and covariance-ellipsoid "
            "IoU/Dice overlap) comparing BackFlip predictions against ground truth, and "
            "report the medians across all matched proteins."
        )
    )
    parser.add_argument(
        "--inference_folder",
        type=str,
        required=True,
        help="Folder of predicted '*_pred.npz' files, as written by BackFlip.predict's npz output.",
    )
    parser.add_argument(
        "--gt_npz_folder",
        type=str,
        required=True,
        help="Folder of ground-truth .npz dataset files, as written by scripts/dataset/generate_dataset.py.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Label used in the printed report. Defaults to the basename of --inference_folder.",
    )
    parser.add_argument(
        "--n_dirs",
        type=int,
        default=8192,
        help="Number of directions sampled on the unit sphere when estimating ellipsoid IoU/Dice overlap.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Squared Mahalanobis-distance threshold defining the covariance ellipsoid surface.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save the per-protein metrics table as a CSV file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    name = args.name or Path(args.inference_folder).name

    metrics_df = quick_compute_metrics_covar_pred_with_ellipsoids(
        args.inference_folder,
        args.gt_npz_folder,
        n_dirs=args.n_dirs,
        tau=args.tau,
    )
    report_metrics_with_ellipsoids(metrics_df, name=name)

    if args.output_csv is not None:
        metrics_df.to_csv(args.output_csv, index=False)
        print(f"Saved per-protein metrics to {args.output_csv}")

if __name__ == "__main__":
    main()
