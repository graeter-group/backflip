#!/bin/bash
#SBATCH --job-name=megascale_ipa
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH --time=12:00:00
#SBATCH --output=/proj/berzelius-2021-29/users/x_sasag/projects/deltaG/backflip_trainlogs/output_%j.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_sasag/projects/deltaG/backflip_trainlogs/error_%j.err

module purge
module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate backflip

cd /proj/berzelius-2021-29/users/x_sasag/projects/deltaG/backflip
export PYTHONPATH=$(pwd)

python experiments/train_ddg.py \
  data.dataset.train_manifest=/proj/berzelius-2021-29/users/x_sasag/projects/deltaG/manifests/megascale_train.csv \
  data.dataset.val_manifest=/proj/berzelius-2021-29/users/x_sasag/projects/deltaG/manifests/megascale_val.csv \
  data.dataset.test_manifest=/proj/berzelius-2021-29/users/x_sasag/projects/deltaG/manifests/megascale_test.csv \
  experiment.run_name=ipa_full \
  experiment.seed=0 \
  experiment.trainer.max_epochs=30 \
  experiment.early_stopping.patience=5 \
  experiment.use_wandb=false
