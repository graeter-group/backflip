import os
import GPUtil
import torch
import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
import logging
from typing import Union

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from gafl.experiment_utils import get_pylogger, flatten_dict

from backflip.models.flexibility_module import FlexibilityModule
from backflip.data.pdb_dataloader import PdbDataModule



class Experiment:
    def __init__(self, *, cfg: DictConfig, log=get_pylogger(__name__)):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment

        self.log = log
        self.trainer = None

        OmegaConf.set_struct(self._exp_cfg, True)

        self.create_data_module()
        self.create_module()

    def create_module(self):
        self._model: LightningModule = FlexibilityModule(self._cfg)

    def create_data_module(self):
        self._datamodule: LightningDataModule = PdbDataModule(self._data_cfg)
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            self.log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            if self._exp_cfg.use_wandb:
                logger = WandbLogger(
                    **self._exp_cfg.wandb,
                )
            else:
                logger = None
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            self.log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(flatten_dict(cfg_dict))
            if self._exp_cfg.use_wandb and isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._exp_cfg.num_devices]
        self.log.info(f"Using devices: {devices}")

        self.trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            replace_sampler_ddp=False, # in later versions of pytorch lightning, this is called use_distributed_sampler
            enable_progress_bar=self._exp_cfg.use_tqdm,
            enable_model_summary=True,
            devices=devices,
        )

        self.trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

        # load best checkpoint
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        best_ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')

        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found at {best_ckpt_path}.")

        self.log.info(f"Evaluating checkpoint with best val loss at \n{Path(best_ckpt_path).absolute()}")

        self.test(ckpt_path=best_ckpt_path, evaluate_on_train=True)


    def test(self, ckpt_path: Union[Path, str], inference_dir: Union[Path, str]=None, evaluate_on_train: bool=True):
        """
        Evaluates the model on the test set and saves the predictions and metrics.

        Args:
            inference_dir (Union[Path, str]): Directory to save the test data and metrics.
            evaluate_on_train (bool, optional): If True, the model is also evaluated on the training and validation sets. Defaults to True.
        """

        self.log.info(f"Evaluating checkpoint at \n{Path(ckpt_path).absolute()}")

        if inference_dir is None:
            inference_dir = Path(ckpt_path).parent

        self._model = FlexibilityModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg
        )

        if self.trainer is None:
            devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._exp_cfg.num_devices]
            self.log.info(f"Using devices: {devices}")

            self.trainer = Trainer(
                **self._exp_cfg.trainer,
                logger=False,
                replace_sampler_ddp=False, # in later versions of pytorch lightning, this is called use_distributed_sampler
                enable_progress_bar=self._exp_cfg.use_tqdm,
                enable_model_summary=False,
                devices=devices,
            )

        # run test
        # disable logging:
        logging.getLogger().setLevel(logging.WARNING)
        self.trainer.test(self._model, datamodule=self._datamodule)
        logging.getLogger().setLevel(logging.INFO)


        # save test data that is now written in the pl module class
        pdbnames = self._model.test_targets['pdb_name']
        # dict of lists of len num_pdbs:
        targets = {metric: list_of_data for metric,list_of_data in self._model.test_targets.items() if metric != 'pdb_name'}
        predictions = self._model.test_predictions
        
        for i, pdbname in enumerate(pdbnames):
            target = {metric: targets[metric][i] for metric in targets}
            prediction = {metric: predictions[metric][i] for metric in predictions}
            # transform to np:
            target = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k,v in target.items()}
            prediction = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k,v in prediction.items()}
            # save to npz:
            Path(f'{inference_dir}/test_data/{pdbname}').mkdir(parents=True, exist_ok=True)
            np.savez(f'{inference_dir}/test_data/{pdbname}/targets.npz', **target)
            np.savez(f'{inference_dir}/test_data/{pdbname}/predictions.npz', **prediction)

        metrics_str_log = {}

        # save the metrics:
        metrics = self._model.test_metrics
        # transform to pd string representation
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'{inference_dir}/test_metrics.csv')
        metrics_str = metrics_df.to_string()
        metrics_str_log['test'] = metrics_str
        with open(f'{inference_dir}/test_metrics.txt', 'w') as f:
            f.write(metrics_str)

        if evaluate_on_train:

            self.log.info(f"Test performance:\n{metrics_str}")
            self.log.info(f"Now evaluating on train and val sets.")

            # disable logging:
            logging.getLogger().setLevel(logging.WARNING)
            train_dl = self._datamodule.train_dataloader()
            val_dl = self._datamodule.val_dataloader()
            logging.getLogger().setLevel(logging.INFO)

            for dataloader, name in [(train_dl, 'train'), (val_dl, 'val')]:
                self._model.test_metrics = None # reset metrics

                # disable logging:
                logging.getLogger().setLevel(logging.WARNING)
                self.trainer.test(self._model, dataloaders=dataloader)
                logging.getLogger().setLevel(logging.INFO)

                # print and store metrics:
                metrics = self._model.test_metrics
                # transform to pd string representation
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(f'{inference_dir}/{name}_metrics.csv')
                metrics_str = metrics_df.to_string()
                metrics_str_log[name] = metrics_str
                with open(f'{inference_dir}/{name}_metrics.txt', 'w') as f:
                    f.write(metrics_str)

            for name in ['train', 'val']:
                metrics_str = metrics_str_log[name]
                self.log.info(f"\n{name} performance: \n {metrics_str_log[name]}")
        marker = "="*70

        self.log.info(f"\n{marker}\nTest performance: \n {metrics_str_log['test']}\n{marker}\n")
        self.log.info(f"Test data saved to {inference_dir}/test_data")