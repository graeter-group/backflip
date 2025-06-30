import os
import GPUtil
import torch
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.trainer import Trainer

from gafl.experiment_utils import get_pylogger

from backflip.models.flexibility_module import FlexibilityModule
from backflip.data.pdb_dataloader import PdbDataset


torch.set_float32_matmul_precision('high')
log = get_pylogger(__name__)

class InferFlexibility:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)

        self._cfg = cfg
        self._data_cfg = cfg.data
        print(cfg.data.dataset)
        self._infer_cfg = cfg.inference
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._flexibility_module = FlexibilityModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
        )
        self._flexibility_module.eval()
        self._flexibility_module._output_dir = self._output_dir

    def model_infer(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        metrics_train_dataset = PdbDataset(
                dataset_cfg=self._data_cfg.dataset, is_training=True
            )
        dataloader = torch.utils.data.DataLoader(
            metrics_train_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.test(self._flexibility_module, dataloaders=dataloader)
        data = self._flexibility_module.predicted_values
        data_safe_path = Path(self._cfg.safe_path)
        data_safe_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(data_safe_path, **data)
        log.info(f'Saved predicted values to {data_safe_path}')

@hydra.main(version_base=None, config_path="../confs", config_name="inference_flex")
def run(cfg: DictConfig) -> None:
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = InferFlexibility(cfg)
    sampler.model_infer()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()