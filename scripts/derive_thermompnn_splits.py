#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path


def _normalize_ids(ids):
    return [str(x) for x in ids]


def _find_split_keys(keys):
    train_keys = [k for k in keys if "train" in k.lower()]
    val_keys = [k for k in keys if any(x in k.lower() for x in ["val", "valid", "dev"])]
    test_keys = [k for k in keys if "test" in k.lower()]
    if not train_keys or not val_keys or not test_keys:
        return None
    return train_keys[0], val_keys[0], test_keys[0]


def _infer_splits(obj):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        lower_keys = {str(k).lower(): k for k in keys}
        if any(k in lower_keys for k in ["train", "val", "valid", "test"]):
            train_key = lower_keys.get("train")
            val_key = lower_keys.get("val") or lower_keys.get("valid")
            test_key = lower_keys.get("test")
            if train_key and val_key and test_key:
                return "dict_split_lists", {"train": obj[train_key], "val": obj[val_key], "test": obj[test_key]}, (train_key, val_key, test_key)
        inferred = _find_split_keys([str(k) for k in keys])
        if inferred:
            train_key, val_key, test_key = inferred
            return "dict_split_lists", {"train": obj[train_key], "val": obj[val_key], "test": obj[test_key]}, (train_key, val_key, test_key)
        # dict of id -> split
        if all(isinstance(v, str) for v in obj.values()):
            split_map = {}
            for k, v in obj.items():
                split_map.setdefault(v, []).append(k)
            inferred = _find_split_keys(list(split_map.keys()))
            if inferred:
                train_key, val_key, test_key = inferred
                return "dict_id_to_split", {"train": split_map[train_key], "val": split_map[val_key], "test": split_map[test_key]}, (train_key, val_key, test_key)
        # nested dict
        return "dict_unknown", None, tuple(keys)
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        return "tuple_splits", {"train": obj[0], "val": obj[1], "test": obj[2]}, ("0", "1", "2")
    return "unknown", None, None


def _write_list(path, ids):
    with open(path, "w") as f:
        for _id in ids:
            f.write(f"{_id}\n")


def _write_full_tsv(path, splits):
    with open(path, "w") as f:
        for split, ids in splits.items():
            for _id in ids:
                f.write(f"{_id}\t{split}\n")


def _write_summary(path, src, fmt, keys, splits):
    with open(path, "w") as f:
        f.write(f"source: {src}\n")
        f.write(f"format: {fmt}\n")
        f.write(f"detected_keys: {keys}\n")
        for split, ids in splits.items():
            f.write(f"{split}_count: {len(ids)}\n")
            f.write(f"{split}_head: {ids[:20]}\n")


def main():
    parser = argparse.ArgumentParser(description="Derive ThermoMPNN split files.")
    parser.add_argument("--pkl", required=True, help="Path to split pickle.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--prefix", required=True, help="Output prefix (e.g., mega).")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    fmt, splits, keys = _infer_splits(obj)
    if splits is None:
        raise ValueError(f"Could not infer split format. Detected format={fmt}, keys={keys}")

    splits = {k: _normalize_ids(v) for k, v in splits.items()}

    summary_path = outdir / f"{args.prefix}_splits_summary.txt"
    train_path = outdir / f"{args.prefix}_train_ids.txt"
    val_path = outdir / f"{args.prefix}_val_ids.txt"
    test_path = outdir / f"{args.prefix}_test_ids.txt"
    tsv_path = outdir / f"{args.prefix}_splits_full.tsv"

    _write_summary(summary_path, str(pkl_path), fmt, keys, splits)
    _write_list(train_path, splits["train"])
    _write_list(val_path, splits["val"])
    _write_list(test_path, splits["test"])
    _write_full_tsv(tsv_path, splits)

    # overlap checks
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    overlap_path = outdir / f"{args.prefix}_overlap_check.txt"
    with open(overlap_path, "w") as f:
        f.write(f"train&val: {len(train_set & val_set)}\n")
        f.write(f"train&test: {len(train_set & test_set)}\n")
        f.write(f"val&test: {len(val_set & test_set)}\n")
        if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
            f.write("overlap_examples:\n")
            f.write(f"train&val: {list((train_set & val_set))[:20]}\n")
            f.write(f"train&test: {list((train_set & test_set))[:20]}\n")
            f.write(f"val&test: {list((val_set & test_set))[:20]}\n")


if __name__ == "__main__":
    main()
