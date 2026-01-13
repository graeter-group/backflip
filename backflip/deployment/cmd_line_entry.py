# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

"""Command line interface for BackFlip."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backflip.deployment.inference_class import BackFlip
from backflip.deployment.utils import save_prediction


def _as_numpy(arr: Any) -> np.ndarray:
    """Convert torch tensors or arrays to a numpy array on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def backflip_predict_cli() -> None:
    """Predict flexibility for a single PDB file."""
    parser = argparse.ArgumentParser(description="Predict flexibility for one PDB file.")
    parser.add_argument("pdb_path", type=Path, help="Path to a PDB file.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output file (.txt, .npy, .pdb, or .cif). If omitted, overwrites the B-factor of the input PDB.",
    )
    parser.add_argument(
        "--tag",
        help="Model tag to load (default backflip-1.0 unless ckpt_path is provided).",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="Path to a specific checkpoint (.ckpt). If provided, overrides --tag.",
    )
    parser.add_argument(
        "--rmsf-type",
        choices=["global_rmsf", "local_flex"],
        default="global_rmsf",
        help="Which flexibility type to predict. Defaults to global_rmsf. global_rmsf: C alpha RMSF as calculated from alignment of MD trajectories. local_flex: Local flexibility score as defined in the paper, calculated from local alignment of residue neighborhoods.",
    )
    args = parser.parse_args()

    if not args.pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {args.pdb_path}")

    if args.ckpt_path is not None:
        bf = BackFlip(ckpt_path=args.ckpt_path, device="cpu", progress_bar=False)
    else:
        bf = BackFlip.from_tag(tag=args.tag or "backflip-1.0", device="cpu", progress_bar=False)
    bf.rmsf_type = args.rmsf_type

    prediction = bf.predict_from_pdb(pdb_path=args.pdb_path)

    if args.output is None:
        save_prediction(
            input_path=args.pdb_path,
            prediction=prediction,
            output_folder=args.pdb_path.parent,
            overwrite=True,
            rmsf_type=args.rmsf_type,
        )
        return

    output_ext = args.output.suffix.lower()
    if output_ext not in {".txt", ".npy", ".pdb", ".cif"}:
        raise ValueError(f"--output must end with .txt, .npy, .pdb, or .cif. Got {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if output_ext in {".txt", ".npy"}:
        profile = _as_numpy(prediction[args.rmsf_type])
        if output_ext == ".txt":
            np.savetxt(args.output, profile)
        else:
            np.save(args.output, profile)
        return

    save_prediction(
        input_path=args.pdb_path,
        prediction=prediction,
        output_folder=args.output.parent,
        overwrite=True,
        rmsf_type=args.rmsf_type,
    )
    produced_file = args.output.parent / args.pdb_path.name
    if produced_file != args.output:
        if args.output.exists():
            args.output.unlink()
        produced_file.rename(args.output)


def backflip_annotate_cli() -> None:
    """Annotate a folder of PDB files with predicted flexibility."""
    parser = argparse.ArgumentParser(description="Annotate PDB files with BackFlip predictions.")
    parser.add_argument("input_path", type=Path, help="Path to a PDB file or folder of PDB files.")
    parser.add_argument(
        "--output-folder",
        "-o",
        type=Path,
        help="Where to write outputs. Defaults to an inference_results folder next to the inputs.",
    )
    parser.add_argument(
        "--tag",
        help="Model tag to load (default backflip-1.0 unless ckpt_path is provided).",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="Path to a specific checkpoint (.ckpt). If provided, overrides --tag.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (e.g., cuda or cpu). Defaults to cuda.",
    )
    parser.add_argument(
        "--rmsf-type",
        choices=["global_rmsf", "local_flex"],
        default="global_rmsf",
        help="Which prediction to write into outputs. Defaults to global_rmsf.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of writing to a new folder.",
    )
    parser.add_argument(
        "--cuda-memory-gb",
        type=int,
        default=8,
        help="Approximate available GPU memory in GB for batch size estimation. Defaults to 8.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum batch size per forward pass. Defaults to auto-estimation.",
    )
    parser.add_argument(
        "--path-batchsize",
        type=int,
        default=1000,
        help="How many paths to load per predict call. Defaults to 1000.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    if args.ckpt_path is not None:
        bf = BackFlip(ckpt_path=args.ckpt_path, device=args.device, progress_bar=not args.no_progress_bar)
    else:
        bf = BackFlip.from_tag(tag=args.tag or "backflip-1.0", device=args.device, progress_bar=not args.no_progress_bar)
    bf.rmsf_type = args.rmsf_type

    bf.predict(
        input_path=args.input_path,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        cuda_memory_GB=args.cuda_memory_gb,
        stop_grad=True,
        path_batchsize=args.path_batchsize,
        overwrite=args.overwrite,
    )


__all__ = ["backflip_predict_cli", "backflip_annotate_cli"]
