#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. PREDICT FLEXIBILITY OF A GIVEN PDB FILE
###############################
PDB_FOLDER="${REPO_DIR}/test_data/inference_examples/from_pdb_folder"

# Predict flexibility for a single PDB and write the isotropic RMSF profile to a TXT file
backflip-predict "${PDB_FOLDER}/1ubq.pdb" --tag backflip-2.1 --output "${PDB_FOLDER}/inference_results/1ubq_global_rmsf.txt"
# or write it as B-factors into a new CIF file (requires --rmsf-as-bfactor). This internally
# writes to an 'npz/' and 'cifs/' subfolder next to --output (see below), then moves the CIF
# out of 'cifs/' to the exact --output path; the 'npz/' subfolder with the raw predictions is
# left in place next to it.
backflip-predict "${PDB_FOLDER}/1ubq.pdb" --tag backflip-2.1 --output "${PDB_FOLDER}/inference_results/1ubq_global_rmsf.cif" --rmsf-as-bfactor

# 2. ANNOTATE A DATASET OF PDB FILES
###############################
# Annotate a folder of PDBs with predicted flexibility. This always writes a npz/ subfolder
# (raw per_res_covariance/pairwise_couplings/pairwise_DCCM predictions), and additionally a
# cifs/ subfolder (isotropic RMSF written into the B-factor column) since --rmsf-as-bfactor
# is passed below.
backflip-annotate "${PDB_FOLDER}" --tag backflip-2.1 --device cuda --cuda-memory-gb 8 --rmsf-as-bfactor --output-folder "${PDB_FOLDER}/inference_results/annotated"
