"""Utility functions for experiments."""
import logging
import torch
import os
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pandas as pd
from pathlib import Path
import re
import math

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def rename_csv_paths(csv:str, save:bool=True):
    df = pd.read_csv(csv)
    new_sample_paths = []
    for idx, row in df.iterrows():
        sample_name = row['pdb_name']
        new_path = Path(csv).parent / f"{sample_name}.npz"
        new_sample_paths.append(new_path)
    df['processed_path'] = new_sample_paths
    if save:
        new_csv_path = Path(csv).parent / f"{Path(csv).stem}.csv"
        df.to_csv(new_csv_path, index=False)
        print(f"Saved renamed CSV to {new_csv_path}")
    return df

def terminal_mask_hec_torch(
    global_rmsf: torch.Tensor,   # (B,N,1), float32
    dssp: torch.Tensor,          # (B,N,1), int32 with {0:H, 1:E, 2:C}
    gap_tolerance: int = 1,
    min_len: int = 5,
    coil_frac_min: float = 0.7,
    tau: float = 2.0,
    pi_max: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns mask M in {0,1} with shape (B,N,1). 1=keep in loss, 0=downweight terminals.
    """
    assert global_rmsf.ndim == 3 and dssp.ndim == 3 and global_rmsf.shape[:2] == dssp.shape[:2]
    B, N, _ = global_rmsf.shape
    device = global_rmsf.device

    M = torch.ones((B, N), dtype=torch.float32, device=device)

    for b in range(B):
        x = global_rmsf[b, :, 0]                          # (N,)
        lab = dssp[b, :, 0].to(torch.int32)               # (N,)
        coil = (lab == 2)                                  # True for 'C'
        non_coil = (~coil).to(torch.int32)

        L = int(N)
        if L == 0:
            continue

        # robust interior stats
        w = max(10, math.ceil(0.05 * L))
        if L < 2 * w + 1:
            w = max(1, L // 4)
        i0, i1 = w, L - w
        if i1 <= i0:
            x_int = x
        else:
            x_int = x[i0:i1]
        med = x_int.median()
        mad = (x_int - med).abs().median()
        sigma = (1.4826 * mad).clamp_min(eps)

        # longest coil-dominated prefix with up to g non-coils
        cum_nc = non_coil.cumsum(dim=0)                    # counts in [0..i]
        pos = torch.nonzero(cum_nc > gap_tolerance, as_tuple=False)
        lenN = int(pos[0].item()) if pos.numel() > 0 else L
        SN = torch.arange(0, lenN, device=device)

        # longest coil-dominated suffix with up to g non-coils
        cum_nc_rev = non_coil.flip(0).cumsum(dim=0)
        pos_rev = torch.nonzero(cum_nc_rev > gap_tolerance, as_tuple=False)
        lenC = int(pos_rev[0].item()) if pos_rev.numel() > 0 else L
        SC = (L - lenC) + torch.arange(0, lenC, device=device) if lenC > 0 else torch.arange(0, 0, device=device)

        def segment_ok(S: torch.Tensor):
            if S.numel() < min_len:
                return False, torch.tensor(0.0, device=device)
            purity = coil[S].float().mean()
            if purity.item() < coil_frac_min:
                return False, purity
            z = (x[S].mean() - med) / sigma
            return bool(z.item() >= tau), z

        passN, zN = segment_ok(SN)
        passC, zC = segment_ok(SC)

        # cap total masked length
        cap = int(math.floor(pi_max * L))
        total_len = (SN.numel() if passN else 0) + (SC.numel() if passC else 0)
        if passN and passC and total_len > cap:
            if zN.item() >= zC.item():
                passC = False
            else:
                passN = False

        if passN:
            M[b, SN] = 0.0
        if passC:
            M[b, SC] = 0.0

    return M.unsqueeze(-1)  # (B,N,1)

def eigen_decomposition(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = C.dtype
    # increased precision for eigen-decomposition to improve numerical stability, especially for small eigenvalues
    C = C.to(torch.float64)
    # numerical symmetrization to improve stability of eigen-decomposition, especially for small eigenvalues
    C = 0.5 * (C + C.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=eps)
    return evals.to(orig_dtype), evecs.to(orig_dtype)

def log_C(evals:torch.Tensor, evecs:torch.Tensor) -> torch.Tensor:
    log_evals = torch.log(evals)
	# log(C) = U diag(log(lambda)) U^T
    logC = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)
    return logC