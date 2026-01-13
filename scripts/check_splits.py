#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd


def _load_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".jsonl", ".json"]:
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported manifest format: {path}")


def _infer_id_col(df, id_col):
    if id_col in df.columns:
        return id_col
    for fallback in ["protein_id", "pdb_name", "processed_path", "wt_structure_path"]:
        if fallback in df.columns:
            return fallback
    raise ValueError(f"Could not infer id column from: {list(df.columns)}")


def _split_from_manifest(df):
    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column in manifest when using --manifest.")
    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = df[df["split"] == split].reset_index(drop=True)
    return splits


def _summary(df, id_col, cluster_col):
    ids = df[id_col].astype(str).tolist()
    summary = {
        "num_examples": int(len(df)),
        "num_unique_ids": int(len(set(ids))),
    }
    if cluster_col is not None and cluster_col in df.columns:
        clusters = df[cluster_col].astype(str).tolist()
        summary["num_unique_clusters"] = int(len(set(clusters)))
    return summary


def _overlap(a, b):
    return sorted(set(a).intersection(set(b)))


def main():
    parser = argparse.ArgumentParser(description="Check DDG split integrity.")
    parser.add_argument("--manifest", type=str, help="Manifest CSV/JSONL with a split column.")
    parser.add_argument("--train", type=str, help="Train manifest path.")
    parser.add_argument("--val", type=str, help="Val manifest path.")
    parser.add_argument("--test", type=str, help="Test manifest path.")
    parser.add_argument("--id-col", type=str, default="protein_id", help="Column for protein ids.")
    parser.add_argument("--cluster-col", type=str, default=None, help="Optional cluster column.")
    parser.add_argument("--out", type=str, default="split_summary.json", help="Output JSON path.")
    args = parser.parse_args()

    if args.manifest is None and not (args.train and args.val and args.test):
        raise ValueError("Provide --manifest or all of --train/--val/--test.")

    if args.manifest is not None:
        df = _load_manifest(args.manifest)
        splits = _split_from_manifest(df)
    else:
        splits = {
            "train": _load_manifest(args.train),
            "val": _load_manifest(args.val),
            "test": _load_manifest(args.test),
        }

    id_col = _infer_id_col(pd.concat(list(splits.values()), ignore_index=True), args.id_col)
    cluster_col = args.cluster_col

    ids = {k: splits[k][id_col].astype(str).tolist() for k in splits}
    clusters = {}
    if cluster_col is not None and cluster_col in splits["train"].columns:
        clusters = {k: splits[k][cluster_col].astype(str).tolist() for k in splits}

    overlap_ids = {
        "train_val": _overlap(ids["train"], ids["val"]),
        "train_test": _overlap(ids["train"], ids["test"]),
        "val_test": _overlap(ids["val"], ids["test"]),
    }

    overlap_clusters = {}
    if clusters:
        overlap_clusters = {
            "train_val": _overlap(clusters["train"], clusters["val"]),
            "train_test": _overlap(clusters["train"], clusters["test"]),
            "val_test": _overlap(clusters["val"], clusters["test"]),
        }

    summary = {
        "id_col": id_col,
        "cluster_col": cluster_col,
        "splits": {
            split: _summary(splits[split], id_col, cluster_col) for split in splits
        },
        "overlap_ids": {k: len(v) for k, v in overlap_ids.items()},
        "overlap_clusters": {k: len(v) for k, v in overlap_clusters.items()} if overlap_clusters else {},
        "overlap_id_examples": {
            k: v[:10] for k, v in overlap_ids.items() if len(v) > 0
        },
        "overlap_cluster_examples": {
            k: v[:10] for k, v in overlap_clusters.items() if len(v) > 0
        } if overlap_clusters else {},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    has_overlap = any(len(v) > 0 for v in overlap_ids.values())
    if overlap_clusters:
        has_overlap = has_overlap or any(len(v) > 0 for v in overlap_clusters.values())
    if has_overlap:
        raise SystemExit("Split overlap detected. See summary JSON for details.")


if __name__ == "__main__":
    main()
