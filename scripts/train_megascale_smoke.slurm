#!/bin/bash
#SBATCH --job-name=megascale_smoke
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH --time=02:00:00
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
  data.dataset.train_max_rows=2000 \
  data.dataset.val_max_rows=200 \
  data.dataset.test_max_rows=200 \
  data.loader.batch_size=4 \
  experiment.run_name=smoke_real \
  experiment.trainer.max_epochs=3 \
  experiment.use_wandb=false
