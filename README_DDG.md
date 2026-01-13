# DeltaG (DDG) Pipeline for MegaScale

Goal: ddG prediction on the MegaScale dataset using BackFlip's IPA trunk with official ThermoMPNN splits.

## What was added
- DDG dataset + datamodule that read manifest CSVs and reuse BackFlip featurization.
- Mutation-aware DDG head on top of the IPA trunk.
- No-IPA baseline (MLP over pooled node features).
- Scripts to derive ThermoMPNN splits, build MegaScale manifests, and summarize results.
- Slurm scripts for smoke runs and full IPA/No-IPA runs.

## Repro steps (use environment variables, no hardcoded paths)
From the BackFlip repo root:

```bash
# Paths (adjust to your local data locations)
export REPO_ROOT=$(pwd)
export DATA_ROOT=/path/to/deltaG
export SPLITS_DIR=$DATA_ROOT/external/thermompnn_splits
export ASSETS_DIR=$SPLITS_DIR/raw/mega_assets

# 1) Derive ThermoMPNN splits (if you downloaded dataset_splits)
python scripts/derive_thermompnn_splits.py \
  --pkl $SPLITS_DIR/raw/dataset_splits/mega_splits.pkl \
  --outdir $SPLITS_DIR/derived \
  --prefix mega

# 2) Build MegaScale manifests
python scripts/build_megascale_ddg_manifest.py \
  --train-ids $SPLITS_DIR/derived/mega_train_ids.txt \
  --val-ids $SPLITS_DIR/derived/mega_val_ids.txt \
  --test-ids $SPLITS_DIR/derived/mega_test_ids.txt \
  --assets-root $ASSETS_DIR \
  --outdir $DATA_ROOT/manifests

# 3) Smoke run (small real-data check)
python experiments/train_ddg.py \
  data.dataset.train_manifest=$DATA_ROOT/manifests/megascale_train.csv \
  data.dataset.val_manifest=$DATA_ROOT/manifests/megascale_val.csv \
  data.dataset.test_manifest=$DATA_ROOT/manifests/megascale_test.csv \
  data.dataset.train_max_rows=2000 \
  data.dataset.val_max_rows=200 \
  data.dataset.test_max_rows=200 \
  data.loader.batch_size=4 \
  experiment.run_name=smoke_real \
  experiment.trainer.max_epochs=3 \
  experiment.use_wandb=false

# 4) Full IPA run
python experiments/train_ddg.py \
  data.dataset.train_manifest=$DATA_ROOT/manifests/megascale_train.csv \
  data.dataset.val_manifest=$DATA_ROOT/manifests/megascale_val.csv \
  data.dataset.test_manifest=$DATA_ROOT/manifests/megascale_test.csv \
  experiment.run_name=ipa_full \
  experiment.seed=0 \
  experiment.trainer.max_epochs=30 \
  experiment.early_stopping.patience=5 \
  experiment.use_wandb=false

# 5) Full No-IPA baseline
python experiments/train_ddg.py \
  data.dataset.train_manifest=$DATA_ROOT/manifests/megascale_train.csv \
  data.dataset.val_manifest=$DATA_ROOT/manifests/megascale_val.csv \
  data.dataset.test_manifest=$DATA_ROOT/manifests/megascale_test.csv \
  model.ddg.model_type=noipa \
  experiment.run_name=noipa_full \
  experiment.seed=0 \
  experiment.trainer.max_epochs=30 \
  experiment.early_stopping.patience=5 \
  experiment.use_wandb=false

# 6) Summarize results
python scripts/summarize_ddg_results.py \
  --ipa-run $REPO_ROOT/runs/ipa_full \
  --noipa-run $REPO_ROOT/runs/noipa_full \
  --out $REPO_ROOT/tables/ipa_vs_noipa_megascale.tsv
```

## Output artifacts
- `runs/<run_name>/metrics.csv`
- `runs/<run_name>/test_metrics.json`
- `runs/<run_name>/test_preds.csv`

## Sample manifest
`manifests/sample_megascale_manifest.csv` is a schema example only and uses fake paths.
