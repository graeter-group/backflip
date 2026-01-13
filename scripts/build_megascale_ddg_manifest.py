#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_ddg_csv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split(",")
            if len(parts) < 4:
                continue
            pos, wt, mut, ddg = parts[0], parts[1], parts[2], parts[3]
            wt_pos = None
            if len(wt) > 1:
                wt_letter = wt[0]
                wt_digits = "".join([c for c in wt[1:] if c.isdigit()])
                if wt_digits:
                    wt_pos = int(wt_digits)
                    wt = wt_letter
            try:
                pos_i = int(pos)
            except ValueError:
                pos_i = None
            if wt_pos is not None:
                pos_i = wt_pos
            if pos_i is None:
                continue
            try:
                ddg_f = float(ddg)
            except ValueError:
                continue
            rows.append((pos_i, wt, mut, ddg_f))
    return rows


def infer_chain_id(protein_id):
    if protein_id.endswith(".pdb") and "_" in protein_id:
        tail = protein_id[:-4].split("_")[-1]
        if len(tail) == 1 and tail.isalnum():
            return tail
    if protein_id.endswith(".pdb") and "_" not in protein_id:
        return "A"
    return "A"


def pdb_residue_numbers(pdb_path, chain_id=None):
    seen = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            chain = line[21].strip()
            if chain_id and chain != chain_id:
                continue
            resseq = line[22:26].strip()
            if not resseq:
                continue
            try:
                seen.add(int(resseq))
            except ValueError:
                continue
    return seen


def main():
    parser = argparse.ArgumentParser(description="Build MegaScale DDG manifests.")
    parser.add_argument("--splits-root", required=True, help="Path to derived ThermoMPNN splits.")
    parser.add_argument("--assets-root", required=True, help="Path to mega_assets directory.")
    parser.add_argument("--outdir", required=True, help="Output manifests directory.")
    args = parser.parse_args()

    splits_root = Path(args.splits_root).resolve()
    assets_root = Path(args.assets_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": splits_root / "mega_train_ids.txt",
        "val": splits_root / "mega_val_ids.txt",
        "test": splits_root / "mega_test_ids.txt",
    }

    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

    pdb_dir = assets_root / "pdbs"
    fasta_dir = assets_root / "fastas"
    ddg_dir = assets_root / "ddg_csvs"
    for d in [pdb_dir, fasta_dir, ddg_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing assets directory: {d}")

    manifest_paths = {
        "train": outdir / "megascale_train.csv",
        "val": outdir / "megascale_val.csv",
        "test": outdir / "megascale_test.csv",
    }

    summary_lines = []
    for split, split_path in split_files.items():
        protein_ids = read_ids(split_path)
        protein_count = len(protein_ids)
        mutation_rows = 0

        out_path = manifest_paths[split]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "protein_id",
                    "split",
                    "pdb_path",
                    "fasta_path",
                    "ddg_csv_path",
                    "mutation",
                    "wt_aa",
                    "mut_aa",
                    "mut_pos",
                    "ddg",
                    "chain_id",
                    "source_dataset",
                ],
            )
            writer.writeheader()

            for protein_id in protein_ids:
                pdb_path = (pdb_dir / protein_id).resolve()
                fasta_candidates = [
                    (fasta_dir / f"{protein_id[:-4]}.fasta").resolve(),
                    (fasta_dir / f"{protein_id[:-4]}_A.fasta").resolve(),
                ]
                ddg_candidates = [
                    (ddg_dir / f"{protein_id[:-4]}.csv").resolve(),
                    (ddg_dir / f"{protein_id[:-4]}_A.csv").resolve(),
                ]

                fasta_path = next((p for p in fasta_candidates if p.exists()), None)
                ddg_path = next((p for p in ddg_candidates if p.exists()), None)

                if not pdb_path.exists():
                    raise FileNotFoundError(f"Missing PDB: {pdb_path}")
                if fasta_path is None or not fasta_path.exists():
                    raise FileNotFoundError(f"Missing FASTA for {protein_id}")
                if ddg_path is None or not ddg_path.exists():
                    raise FileNotFoundError(f"Missing ddG CSV for {protein_id}")

                chain_id = infer_chain_id(protein_id)
                ddg_rows = parse_ddg_csv(ddg_path)
                max_pos = None
                with open(fasta_path, "r") as fasta_f:
                    seq_lines = [line.strip() for line in fasta_f if line.strip() and not line.startswith(">")]
                if seq_lines:
                    max_pos = len("".join(seq_lines))
                pdb_resnums = pdb_residue_numbers(pdb_path, chain_id if chain_id else None)
                if pdb_resnums:
                    max_resnum = max(pdb_resnums)
                    max_pos = min(max_pos, max_resnum) if max_pos is not None else max_resnum
                for pos, wt, mut, ddg in ddg_rows:
                    if max_pos is not None and pos > max_pos:
                        continue
                    if pdb_resnums and pos not in pdb_resnums:
                        continue
                    mutation_rows += 1
                    writer.writerow({
                        "protein_id": protein_id,
                        "split": split,
                        "pdb_path": str(pdb_path),
                        "fasta_path": str(fasta_path.resolve()),
                        "ddg_csv_path": str(ddg_path.resolve()),
                        "mutation": f"{wt}{pos}{mut}",
                        "wt_aa": wt,
                        "mut_aa": mut,
                        "mut_pos": pos,
                        "ddg": ddg,
                        "chain_id": chain_id if chain_id else "",
                        "source_dataset": "megascale",
                    })

        summary_lines.append(f"{split}_proteins: {protein_count}")
        summary_lines.append(f"{split}_mutations: {mutation_rows}")

    summary_path = outdir / "megascale_manifest_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
