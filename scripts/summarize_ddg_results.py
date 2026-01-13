#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "test_metrics.json"
    preds_path = run_dir / "test_preds.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test_metrics.json in {run_dir}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing test_preds.csv in {run_dir}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    num_test = len(pd.read_csv(preds_path))
    return metrics, num_test


def main():
    parser = argparse.ArgumentParser(description="Summarize DDG benchmark results.")
    parser.add_argument("--ipa", required=True, help="Path to IPA run directory.")
    parser.add_argument("--noipa", required=True, help="Path to No-IPA run directory.")
    parser.add_argument("--out", required=True, help="Output TSV path.")
    args = parser.parse_args()

    rows = []
    for label, run_dir in [("ipa", args.ipa), ("noipa", args.noipa)]:
        metrics, num_test = load_metrics(run_dir)
        rows.append({
            "model": label,
            "RMSE": metrics.get("rmse"),
            "Pearson": metrics.get("pearson"),
            "Spearman": metrics.get("spearman"),
            "num_test_examples": num_test,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
