import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backflip.data.profile_metrics import get_metrics
from backflip.deployment.inference_class import BackFlip


def _to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser(description="Run BackFlip inference on a CSV of processed structures and report metrics.")
    parser.add_argument("csv_path", type=Path, help="CSV file with a processed_path column pointing to .npz/.pkl files.")
    parser.add_argument("--tag", help="Model tag to load (default backflip-1.0 unless ckpt_path is provided).", default="backflip-1.0")
    parser.add_argument("--ckpt-path", type=Path, help="Path to a specific checkpoint (.ckpt). If provided, overrides --tag.")
    parser.add_argument("--device", default="cuda", help="Device to run on (e.g., cpu or cuda). Defaults to cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum batch size per forward pass. Defaults to auto-estimation.")
    parser.add_argument("--cuda-memory-gb", type=int, default=8, help="Approximate available GPU memory in GB for batch size estimation. Defaults to 8.")
    parser.add_argument("--path-batchsize", type=int, default=1000, help="How many paths to load per predict call. Defaults to 1000.")
    args = parser.parse_args()

    if args.ckpt_path is not None:
        bf = BackFlip(ckpt_path=args.ckpt_path, device=args.device)
    else:
        bf = BackFlip.from_tag(tag=args.tag, device=args.device)

    preds, gts = bf.predict_from_csv(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        cuda_memory_GB=args.cuda_memory_gb,
        stop_grad=True,
        path_batchsize=args.path_batchsize,
    )

    assert len(preds) == len(gts), "Number of predictions and ground-truth entries should match."

    pred_dict = {}
    target_dict = {}
    for pred, gt in zip(preds, gts):
        for key, val in pred.items():
            # Only score keys that have ground truth available
            if key not in gt:
                continue
            pred_arr = _to_numpy(val).squeeze().astype(np.float32)
            gt_arr = _to_numpy(gt[key]).squeeze().astype(np.float32)
            pred_dict.setdefault(key, []).append(pred_arr)
            target_dict.setdefault(key, []).append(gt_arr)

    if not pred_dict:
        raise ValueError("No overlapping prediction/ground-truth keys found (expected global_rmsf/local_flex).")

    metrics = get_metrics(pred_dict, target_dict, list(pred_dict.keys()))
    metrics_df = pd.DataFrame(metrics)
    # for simplicity, only print per-target-avg-r per-target-avg-mae num-proteins:
    metrics_df = metrics_df.loc[["per-target-avg-r", "per-target-avg-mae", "num-proteins"]]
    print(metrics_df.to_string())


if __name__ == "__main__":
    main()
