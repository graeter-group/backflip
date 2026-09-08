"""
Build a joint ATLAS + mdCATH train/val/test split using structural clustering.

Prerequisite (run this before invoking the script):
Cluster the ATLAS+mdCATH equilibrium PDBs with foldseek to produce a
`res_cluster.tsv` cluster assignment tsv:

    foldseek easy-cluster <atlas_mdcath_eq_pdbs_dir> res tmp \
        -c 0.9 --tmscore-threshold 0.5 --alignment-type 1

This writes `res_cluster.tsv` (plus other `res_*`/`tmp` artifacts) into
the directory the command was run from. Point --cluster-tsv at that file
(or copy it into --atlas-mdcath-eq-pdbs, its default location).

Any protein cluster that mixes ATLAS proteins from more than one split, or
that only contains mdCATH proteins, is handled separately: ATLAS splits take
priority for clusters containing ATLAS proteins, and mdCATH-only clusters are
randomly assigned to train/val/test.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_foldseek_cluster_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["cluster_id", "pdb_name"])
    df["cluster_size"] = df.groupby("cluster_id")["pdb_name"].transform("count")
    return df


def load_full_atlas_split(alphaflow_train_split, alphaflow_val_split, alphaflow_test_split):
    train_df = pd.read_csv(alphaflow_train_split)
    val_df = pd.read_csv(alphaflow_val_split)
    test_df = pd.read_csv(alphaflow_test_split)

    full_atlas_split = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_atlas_split["split"] = np.repeat(
        ["train", "val", "test"], [len(train_df), len(val_df), len(test_df)]
    )
    return full_atlas_split


def assign_joint_sample_table_from_atlas(
    full_atlas_split: pd.DataFrame,
    cluster_split_atlas_mdcath_df: pd.DataFrame,
):
    """
    Returns a sample-level dataframe with one row per pdb_name across ATLAS + mdCATH.

    full_atlas_split columns:
        - pdb_name
        - split in {'train', 'val', 'test'}

    cluster_split_atlas_mdcath_df columns:
        - cluster_id
        - pdb_name
    """
    atlas = full_atlas_split[["pdb_name", "split"]].copy()
    atlas = atlas.rename(columns={"split": "atlas_split"})

    clusters = cluster_split_atlas_mdcath_df[["cluster_id", "pdb_name"]].copy()

    atlas_names = set(atlas["pdb_name"])
    clusters["dataset"] = np.where(clusters["pdb_name"].isin(atlas_names), "atlas", "mdcath")

    merged = clusters.merge(atlas, on="pdb_name", how="left")

    cluster_rows = []
    for cluster_id, g in merged.groupby("cluster_id", sort=False):
        atlas_members = sorted(g.loc[g["dataset"] == "atlas", "pdb_name"].tolist())
        mdcath_members = sorted(g.loc[g["dataset"] == "mdcath", "pdb_name"].tolist())
        atlas_splits = sorted(g.loc[g["dataset"] == "atlas", "atlas_split"].dropna().unique().tolist())

        has_atlas = len(atlas_members) > 0
        conflict = len(atlas_splits) > 1

        if not has_atlas:
            cluster_status = "mdcath_only"
            assigned_split = None
        elif conflict:
            cluster_status = "conflict"
            assigned_split = None
        else:
            cluster_status = "assigned_from_atlas"
            assigned_split = atlas_splits[0]

        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "n_members": len(g),
                "n_atlas": len(atlas_members),
                "n_mdcath": len(mdcath_members),
                "atlas_members": ",".join(atlas_members),
                "mdcath_members": ",".join(mdcath_members),
                "atlas_splits_present": ",".join(atlas_splits),
                "conflict": conflict,
                "cluster_status": cluster_status,
                "assigned_split": assigned_split,
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows)

    out = merged.merge(cluster_summary, on="cluster_id", how="left")

    # Preserve ATLAS split as source of truth
    out["final_split"] = np.where(
        out["dataset"] == "atlas",
        out["atlas_split"],
        out["assigned_split"],
    )

    return out, cluster_summary


def assign_random_mdcath_only_splits(
    out: pd.DataFrame,
    cluster_col: str = "cluster_id",
    seed: int = 123,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8

    out = out.copy()

    mask = (
        (out["dataset"] == "mdcath") &
        (out["final_split"].isna()) &
        (out["conflict"] == False)
    )

    mdcath_no_split = out.loc[mask].copy()
    cluster_ids = mdcath_no_split[cluster_col].drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)

    n = len(cluster_ids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_clusters = set(cluster_ids[:n_train])
    val_clusters = set(cluster_ids[n_train:n_train + n_val])

    def assign(cluster_id):
        if cluster_id in train_clusters:
            return "train"
        if cluster_id in val_clusters:
            return "val"
        return "test"

    cluster_to_split = {cid: assign(cid) for cid in cluster_ids}

    out.loc[mask, "final_split"] = out.loc[mask, cluster_col].map(cluster_to_split)

    return out


def subset_metadata_by_split(full_ds, atlas_df, mdcath_df, split: str):
    split_df = full_ds[full_ds["final_split"] == split]
    atlas_pdbs = set(split_df.loc[split_df["dataset"] == "atlas", "pdb_name"])
    mdcath_pdbs = set(split_df.loc[split_df["dataset"] == "mdcath", "pdb_name"])

    atlas_subset = atlas_df[atlas_df["pdb_name"].isin(atlas_pdbs)].copy()
    mdcath_subset = mdcath_df[mdcath_df["pdb_name"].isin(mdcath_pdbs)].copy()

    return atlas_subset, mdcath_subset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a joint ATLAS + mdCATH train/val/test split using structural clustering."
    )
    parser.add_argument(
        "--atlas-mdcath-eq-pdbs",
        type=str,
        required=True,
        help="Directory of equilibrium PDBs shared between ATLAS and mdCATH that was foldseek-clustered.",
    )
    parser.add_argument(
        "--cluster-tsv",
        type=str,
        default=None,
        help="Path to the foldseek res_cluster.tsv file. Defaults to <atlas-mdcath-eq-pdbs>/res_cluster.tsv.",
    )
    parser.add_argument(
        "--alphaflow-train-split",
        type=str,
        required=True,
        help="Path to the alphaflow_train.csv file (ATLAS train split).",
    )
    parser.add_argument(
        "--alphaflow-val-split",
        type=str,
        required=True,
        help="Path to the alphaflow_val.csv file (ATLAS val split).",
    )
    parser.add_argument(
        "--alphaflow-test-split",
        type=str,
        required=True,
        help="Path to the alphaflow_test.csv file (ATLAS test split).",
    )
    parser.add_argument(
        "--mdcath-meta",
        type=str,
        required=True,
        help="Path to the mdCATH metadata.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the joint split and per-split metadata CSVs to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used to assign mdCATH-only clusters to train/val/test (default: 123).",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction for mdCATH-only clusters (default: 0.8).")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Val fraction for mdCATH-only clusters (default: 0.1).")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction for mdCATH-only clusters (default: 0.1).")
    return parser.parse_args()


def main():
    args = parse_args()

    cluster_tsv = args.cluster_tsv or os.path.join(args.atlas_mdcath_eq_pdbs, "res_cluster.tsv")
    cluster_split_atlas_mdcath_df = load_foldseek_cluster_tsv(cluster_tsv)

    full_atlas_split = load_full_atlas_split(
        args.alphaflow_train_split,
        args.alphaflow_val_split,
        args.alphaflow_test_split,
    )
    mdcath_df = pd.read_csv(args.mdcath_meta)

    out, _cluster_summary = assign_joint_sample_table_from_atlas(
        full_atlas_split=full_atlas_split,
        cluster_split_atlas_mdcath_df=cluster_split_atlas_mdcath_df,
    )

    full_ds = assign_random_mdcath_only_splits(
        out,
        cluster_col="cluster_id",
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_ds.to_csv(output_dir / "joint_atlas_mdcath_split.csv", index=False)

    for split in ("train", "val", "test"):
        atlas_subset, mdcath_subset = subset_metadata_by_split(
            full_ds, full_atlas_split, mdcath_df, split=split
        )
        concat_df = pd.concat([atlas_subset, mdcath_subset], ignore_index=True)
        concat_df.to_csv(output_dir / f"joint_atlas_mdcath_split_{split}.csv", index=False)

if __name__ == "__main__":
    main()
