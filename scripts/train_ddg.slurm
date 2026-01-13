#!/bin/bash
#SBATCH --job-name=backflip_ddg_train
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

# Update these paths and optional overrides as needed.
python experiments/train_ddg.py \
  data.dataset.manifest_path=dev_ddg/ddg_manifest.csv \
  experiment.dev_run=true \
  data.dataset.dev_run=true
