#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. PREDICT FLEXIBILITY OF A GIVEN PDB FILE
###############################
# Predict flexibility for a single PDB and write the profile to a TXT file
PDB_FOLDER="${REPO_DIR}/test_data/inference_examples/from_pdb_folder"
mkdir -p "$(dirname "${TXT_OUT}")"
backflip-predict "${PDB_FOLDER}/1ubq.pdb" --tag backflip-1.0 --output "${PDB_FOLDER}/inference_results/1ubq_global_rmsf.txt"
# or write the predicted flexibility as b factors in a new PDB file
backflip-predict "${PDB_FOLDER}/1ubq.pdb" --tag backflip-1.0 --output "${PDB_FOLDER}/inference_results/1ubq_global_rmsf.pdb"

# 2. ANNOTATE A DATASET OF PDB FILES
# Annotate a folder of PDBs with predicted flexibility (written to B factor columns of pdb files)
backflip-annotate "${PDB_FOLDER}" --tag backflip-1.0 --device cuda --cuda-memory-gb 8 --output-folder "${PDB_FOLDER}/inference_results/annotated_pdbs"
