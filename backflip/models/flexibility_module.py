# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import shutil
import logging
from pytorch_lightning import LightningModule
import copy
from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import json
# import plt only for saving to files for speedup:
import matplotlib
import matplotlib.pyplot as plt

from backflip.data.pdb_dataloader import PdbDataset
from backflip.models.flexibility_model import FlexibilityModel
from backflip.data.profile_metrics import get_metrics
from backflip.models.pearson_loss import PearsonCorrelationLoss


class FlexibilityModule(LightningModule):

	def __init__(self, cfg):
		super().__init__()
		self._print_logger = logging.getLogger(__name__)
		self._exp_cfg = cfg.experiment
		self._model_cfg = cfg.model
		self._data_cfg = cfg.data

		self.create_model()
		self.mse_loss = torch.nn.MSELoss(reduction='none')
		self.pearson_loss = PearsonCorrelationLoss(dim=-2) # dim=-2 is the residue dimension in our convention of target shape (B, N, C)
		self.huber_loss = torch.nn.SmoothL1Loss(reduction='none', beta=self._exp_cfg.huber_beta)

		self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
		os.makedirs(self._sample_write_dir, exist_ok=True)

		self.train_predictions = defaultdict(list)
		self.train_targets = defaultdict(list)

		self.valid_predictions = defaultdict(list)
		self.valid_targets = defaultdict(list)

		self.test_predictions = defaultdict(list)
		self.test_targets = defaultdict(list)

		self.save_hyperparameters()

		self.val_epoch = 0

		self.valid_plot_pdbnames = None
		self.train_plot_pdbnames = None


	def create_model(self):
		self.model = FlexibilityModel(self._model_cfg)

		
	def on_train_start(self):
		self._epoch_start_time = time.time()


	def loss_fn(self, batch: Any, model_output: Any):
		''''
		Returns a dictionary of loss tensors of shape (num_batch)
		'''

		def profile_loss(profile_type, losses, total_loss):
			'''
			Local helper function to calculate loss for a given scalar profile type.
			'''
			if not profile_type in model_output.keys():
				raise ValueError(f"Expected {profile_type} in model_output, got {[k for k in model_output.keys()]}")
			if not profile_type in batch.keys():
				raise ValueError(f"Expected {profile_type} in batch, got {[k for k in batch.keys()]}")
			pred_profile = model_output[profile_type]
			target = batch[profile_type]
			for loss_fn, weight in self._exp_cfg.loss_weights.items():
				weight_ = self._exp_cfg.prediction_weights[profile_type] * weight
				# determine the term:
				if loss_fn == 'mse' and weight_ > 0:
					this_term = self.mse_loss(pred_profile, target).mean(dim=-1) * weight_
				elif loss_fn == 'pearson' and weight_ > 0:
					this_term = self.pearson_loss(pred_profile, target) * weight_
				elif loss_fn == 'huber' and weight_ > 0:
					this_term = self.huber_loss(pred_profile, target).mean(dim=-1) * weight_
				else:
					if weight_ == 0:
						continue
					else:
						raise ValueError(f"Unknown loss function: {loss_fn}")

				# add the term to total loss and track it:
				losses[profile_type+'_'+loss_fn] = this_term
				total_loss = total_loss + this_term
			return losses, total_loss


		losses = {}
		total_loss = 0

		for profile_type in self._exp_cfg.prediction_weights.keys():
			losses, total_loss = profile_loss(profile_type, losses, total_loss)

		losses["loss"] = total_loss
		return losses


	def model_step(self, batch: Any):
		model_output = self.model(batch)

		losses = self.loss_fn(batch, model_output)
		return losses, model_output


	def training_step(self, batch: Any, stage: int):
		step_start_time = time.time()
		
		batch_losses, model_output = self.model_step(batch)

		num_batch = batch['res_mask'].shape[0]

		total_losses = {k: v.mean() for k,v in batch_losses.items()}

		# average over batch:
		train_loss = total_losses["loss"]

		for k,v in total_losses.items():
			self._log_scalar(f"train/{k}", v, prog_bar=False, on_step=True, batch_size=num_batch)

		with torch.no_grad():
			for profile_type in self._model_cfg.flexibility.outputs:
				if profile_type == 'None':
					continue
				self.train_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
				self.train_predictions[profile_type].extend(model_output[profile_type][...,0].detach().cpu().numpy())
			self.train_targets['pdb_name'].extend([batch['pdb_name'][i] if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])
		
		# Training throughput
		self._log_scalar("train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
		self._log_scalar("train/batch_size", float(num_batch), prog_bar=False)
		step_time = time.time() - step_start_time
		self._log_scalar("train/examples_per_second", num_batch / step_time)

		return train_loss
	

	def on_train_epoch_end(self):
		epoch_time = (time.time() - self._epoch_start_time) / 60.0
		self.log(
			'train/epoch_time_minutes',
			epoch_time,
			on_step=False,
			on_epoch=True,
			prog_bar=False
		)
		self._epoch_start_time = time.time()
		

		# Calculate dict of metric dicts	
		train_epoch_metrics = get_metrics(self.train_predictions, self.train_targets, self._model_cfg.flexibility.outputs)

		# Log metrics
		for target_type, metrics in train_epoch_metrics.items():
			for metric_name, metric_val in metrics.items():
				self._log_scalar(
					f'train-{target_type}/{metric_name}',
					metric_val,
					on_step=False,
					on_epoch=True,
					prog_bar=False,
				)

		# plot examples
		if hasattr(self._exp_cfg, 'plot_val_every_n') and self._exp_cfg.plot_val_every_n is not None and self.val_epoch % self._exp_cfg.plot_val_every_n == 0:
			self.plot_examples(mode='train')

		self.train_predictions.clear()
		self.train_targets.clear()


	def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):

		num_batch, num_res = batch['trans_1'].shape[:2]

		batch_losses, model_output = self.model_step(batch)

		# average over batch:
		total_losses = {k: v.mean() for k,v in batch_losses.items()}

		for k,v in total_losses.items():
			self._log_scalar(f"valid/{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=num_batch)

		with torch.no_grad():
			for profile_type in self._model_cfg.flexibility.outputs:
				if profile_type == 'None':
					continue
				self.valid_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
				self.valid_predictions[profile_type].extend(model_output[profile_type][...,0].detach().cpu().numpy())
			self.valid_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])
	
		
	def on_validation_epoch_end(self):
		# Calculate dict of metric dicts	
		valid_epoch_metrics = get_metrics(self.valid_predictions, self.valid_targets, self._model_cfg.flexibility.outputs)

		# Log metrics
		for target_type, metrics in valid_epoch_metrics.items():
			for metric_name, metric_val in metrics.items():
				self._log_scalar(
					f'valid-{target_type}/{metric_name}',
					metric_val,
					on_step=False,
					on_epoch=True,
					prog_bar=False,
				)

		# plot examples
		if hasattr(self._exp_cfg, 'plot_val_every_n') and self._exp_cfg.plot_val_every_n is not None and self.val_epoch % self._exp_cfg.plot_val_every_n == 0:
			self.plot_examples(mode='valid')

		self.valid_predictions.clear()
		self.valid_targets.clear()

		self.val_epoch += 1


	def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):

		num_batch, num_res = batch['trans_1'].shape[:2]

		batch_losses, model_output = self.model_step(batch)

		with torch.no_grad():
			for profile_type in self._model_cfg.flexibility.outputs:
				if profile_type == 'None':
					continue
				self.test_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
				self.test_predictions[profile_type].extend(model_output[profile_type][...,0].detach().cpu().numpy())
			self.test_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])

	def on_test_epoch_start(self):
		self.test_predictions.clear()
		self.test_targets.clear()

	def on_test_epoch_end(self):
		"""
		At the end of the test epoch, predictions, targets and metrics are stored in the class under attributes test_predictions, test_targets and test_epoch_metrics.
		"""
		# Calculate dict of metric dicts	
		self.test_metrics = get_metrics(self.test_predictions, self.test_targets, self._model_cfg.flexibility.outputs)


	def plot_profile(self, target, prediction, pdb_name, profile_type_name='Flexibility'):
		matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-interactive backend suited for script environments
		assert isinstance(target, np.ndarray) and isinstance(prediction, np.ndarray), f"target and prediction must be numpy arrays, got {type(target)} and {type(prediction)}"
		assert target.shape == prediction.shape, f"target and prediction must have the same shape, got {target.shape} and {prediction.shape}"
		assert len(target.shape) == 1, f"target and prediction must be 1D arrays, got {target.shape} and {prediction.shape}"

		# push a plot of predicted and target local flexibilities
		num_res = len(target)
		plotpath = os.path.join(
			self._sample_write_dir,
			f'flex_{pdb_name}.png'
		)

		rmse = np.sqrt(np.mean((target - prediction)**2))

		fig, ax = plt.subplots()
		ax.plot(np.arange(num_res), target, label='Target')
		ax.plot(np.arange(num_res), prediction, label='Prediction')
		ax.set_xlabel('Residue Index')
		ax.set_ylabel(profile_type_name)
		ax.set_title(f'{pdb_name} - RMSE: {rmse:.2f}')
		fig.legend()
		fig.savefig(plotpath)
		plt.close(fig)

		run_name = Path(self._sample_write_dir).name + '/' + Path(self._sample_write_dir).parent.name + '/' + Path(self._sample_write_dir).parent.parent.name

		if isinstance(self.logger, WandbLogger):
			out = [run_name + " : " + pdb_name, wandb.Image(plotpath)]
			return out


	def _log_scalar(
			self,
			key,
			value,
			on_step=True,
			on_epoch=False,
			prog_bar=True,
			batch_size=None,
			sync_dist=False,
			rank_zero_only=True
		):
		if sync_dist and rank_zero_only:
			raise ValueError('Unable to sync dist when rank_zero_only=True')
		self.log(
			key,
			value,
			on_step=on_step,
			on_epoch=on_epoch,
			prog_bar=prog_bar,
			batch_size=batch_size,
			sync_dist=sync_dist,
			rank_zero_only=rank_zero_only
		)


	def configure_optimizers(self):
		return torch.optim.AdamW(
			params=self.model.parameters(),
			**self._exp_cfg.optimizer
		)


	def plot_examples(self, mode:str='valid'):
		examples = []

		assert mode in ['valid', 'train'], f"Mode must be 'valid' or 'train', got {mode}"

		predictions = self.valid_predictions if mode == 'valid' else self.train_predictions
		targets = self.valid_targets if mode == 'valid' else self.train_targets

		# determine sample idxs for plotted proteins that remain constant for each epoch:
		pdbnames = [str(pdb_name) for pdb_name in targets['pdb_name']]
		if mode == 'valid':
			if self.valid_plot_pdbnames is None:
				self.valid_plot_pdbnames = pdbnames if len(pdbnames) < 10 else [pdbnames[i] for i in torch.randperm(len(pdbnames))[:10]]
			sample_idxs = [pdbnames.index(pdb_name) for pdb_name in self.valid_plot_pdbnames]
		else:
			if self.train_plot_pdbnames is None:
				self.train_plot_pdbnames = pdbnames if len(pdbnames) < 10 else [pdbnames[i] for i in torch.randperm(len(pdbnames))[:10]]
			sample_idxs = [pdbnames.index(pdb_name) for pdb_name in self.train_plot_pdbnames]

		PROFILE_TYPE_NAME = {"sequence_local_rmsf": "Seq.-local RMSF", "global_rmsf": "RMSF", "smoothed_sequence_local_rmsf": "Smoothed Seq.-local RMSF", "local_flex": "Old Seq.-local RMSF"}

		for profile_type in predictions.keys():
			if profile_type not in targets.keys():
				continue

			profile_type_name = PROFILE_TYPE_NAME[profile_type] if profile_type in PROFILE_TYPE_NAME else profile_type

			for i in sample_idxs:
				pdb_name = targets['pdb_name'][i]
				target = targets[profile_type][i]
				prediction = predictions[profile_type][i]
				examples.append([self.current_epoch] + self.plot_profile(target, prediction, pdb_name, profile_type_name))


			if isinstance(self.logger, WandbLogger):
				self.logger.log_table(
				key=f'{mode}-examples/{profile_type}',
				columns=["Epoch", "Protein", profile_type_name],
				data=examples)