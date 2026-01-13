# DDG Pipeline Changelog

## Added files
- backflip/backflip/models/ddg_model.py: IPA-based DDG predictor and No-IPA baseline.
- backflip/backflip/models/ddg_module.py: Lightning module for training/eval and artifact logging.
- backflip/backflip/models/ddg_metrics.py: RMSE/Pearson/Spearman metrics.
- backflip/configs/train_ddg.yaml: Hydra entry config for DDG training.
- backflip/configs/model/ddg.yaml: DDG model config.
- backflip/configs/data/ddg.yaml: DDG data config.
- backflip/scripts/train_ddg.slurm: Generic DDG training Slurm script.
- backflip/scripts/train_megascale_smoke.slurm: Small real-data smoke run.
- backflip/scripts/train_megascale_smoke_big.slurm: Larger smoke run.
- backflip/scripts/train_megascale_ipa.slurm: Full IPA training run.
- backflip/scripts/train_megascale_noipa.slurm: Full No-IPA baseline run.
- backflip/scripts/derive_thermompnn_splits.py: Parse ThermoMPNN split pkls to readable splits.
- backflip/scripts/build_megascale_ddg_manifest.py: Build MegaScale DDG manifests from assets.
- backflip/scripts/summarize_ddg_results.py: Summarize IPA vs No-IPA results table.
- backflip/manifests/sample_megascale_manifest.csv: Schema-only example manifest.
- README_DDG.md: Overview and reproducible commands.
- CHANGELOG_DDG.md: Change log for this DDG work.

## Modified files
- backflip/backflip/data/ddg_dataloader.py: Manifest-driven DDG dataset, collate, PDB support.
- backflip/experiments/train_ddg.py: Training entry point and evaluation outputs.
- backflip/configs/experiment/ddg.yaml: Trainer defaults and early stopping.
- backflip/.gitignore: Ignore data and run artifacts, allow sample manifest.
