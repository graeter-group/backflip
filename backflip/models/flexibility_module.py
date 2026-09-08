# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import torch
import time
import os
import wandb
import numpy as np
import logging
from pytorch_lightning import LightningModule
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
# import plt only for saving to files for speedup:
import matplotlib
import matplotlib.pyplot as plt

from backflip.models.flexibility_model import FlexibilityModelIPA
from backflip.data.profile_metrics import get_metrics
from backflip.utils import eigen_decomposition, log_C
from backflip.data.flexibility_utils import batched_rmsf_from_covar

def log_frobenius_loss_per_residue(
    C_pred: torch.Tensor,
    C_true: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    C_pred, C_true: (B, N, 3, 3) SPD covariances per residue.

    Returns:
        scalar loss if reduction in {"mean","sum"},
        else (B, N) loss per residue.
    """
    B, N, D, D2 = C_pred.shape
    assert D == D2 == 3

    C_pred_flat = C_pred.reshape(B * N, D, D)
    C_true_flat = C_true.reshape(B * N, D, D)

    evals_pred, evecs_pred = eigen_decomposition(C_pred_flat, eps=eps)
    evals_true, evecs_true = eigen_decomposition(C_true_flat, eps=eps)

    logC_pred = log_C(evals_pred, evecs_pred)
    logC_true = log_C(evals_true, evecs_true)

    diff = logC_pred - logC_true
    loss_per = torch.sum(diff * diff, dim=(-2, -1))  # (B*N,)

    if reduction == "mean":
        return loss_per.mean()
    if reduction == "sum":
        return loss_per.sum()
    return loss_per.view(B, N)

def correlation_matrix_distance_loss(
    C_pred: torch.Tensor,
    C_true: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Adapted from doi: 10.1109/VETECS.2005.1543265
    Input shape (..., N, N)
    CMD: d_corr(R1, R2) = 1 - tr(R1 R2) / (||R1||_F ||R2||_F)
    """
    # tr(R1 R2) = sum_ij R1_ij * R2_ji
    tr_C1C2 = torch.einsum("...ij,...ji->...", C_true, C_pred)

    n1 = torch.sqrt((C_true * C_true).sum(dim=(-2, -1)))
    n2 = torch.sqrt((C_pred * C_pred).sum(dim=(-2, -1)))

    denom = (n1 * n2).clamp_min(eps)
    dist = 1.0 - (tr_C1C2 / denom)

    if reduction == "mean":
        return dist.mean()
    if reduction == "sum":
        return dist.sum()
    if reduction == "none":
        return dist
    raise ValueError(f"Unknown reduction: {reduction}")

def pearson_loss_dccm(pred_dccm: torch.Tensor, target_dccm: torch.Tensor, eps: float = 1e-8):
    """
    pred_dccm, target_dccm: (B, N, N)
    returns: scalar loss = 1 - mean_pearson
    """
    B, N, _ = pred_dccm.shape
    device = pred_dccm.device

    iu = torch.triu_indices(N, N, offset=1, device=device)
    x = pred_dccm[:, iu[0], iu[1]]    # (B, K)
    y = target_dccm[:, iu[0], iu[1]]  # (B, K)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    num = (x * y).sum(dim=1)  # (B,)
    den = torch.sqrt((x * x).sum(dim=1) * (y * y).sum(dim=1) + eps)  # (B,)
    corr = num / den  # (B,)

    return 1.0 - corr.mean()

def mse_loss_dccm(pred_dccm: torch.Tensor, target_dccm: torch.Tensor):
    """
    pred_dccm, target_dccm: (B, N, N)
    returns: scalar MSE over off-diagonal entries only
    """
    # TODO: i can only applu loss to the lower or upper triangle in principle 
    B, N, _ = pred_dccm.shape
    device = pred_dccm.device
    offdiag = ~torch.eye(N, device=device, dtype=torch.bool)
    pred_dccm = pred_dccm[:, offdiag]
    target_dccm = target_dccm[:, offdiag]
    diff = pred_dccm - target_dccm
    # MSE over off-diagonal entries only
    this_term = (diff ** 2).mean()
    return this_term

class FlexibilityModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data

        self.create_model()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='none', beta=self._exp_cfg.huber_beta)
        self.mse_loss_dccm = mse_loss_dccm

        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        
        # Frobenius norm loss on N,3,3:
        self.frobenius_loss_per_res = log_frobenius_loss_per_residue
        self.reduction_frobenius = self._exp_cfg.get("reduction_frobenius", "mean")
        # check if aux loss on rmsf helps
        self.mse_rmsf = self._exp_cfg.get("mse_rmsf", False)
        self.mse_rmsf_weight = self._exp_cfg.get("mse_rmsf_weight", 0.0)
        
        # Correlation matrix distance loss on N,N:
        self.cmd_loss = correlation_matrix_distance_loss
        self.mae_cmd = self._exp_cfg.get("mae_cmd", False)
        self.mae_cmd_weight = self._exp_cfg.get("mae_cmd_weight", 0.0)
        self.huber_cmd = self._exp_cfg.get("huber_cmd", False)
        self.huber_cmd_weight = self._exp_cfg.get("huber_cmd_weight", 0.0)

        self.pearson_dccm = self._exp_cfg.get("pearson_dccm", False)
        self.pearson_dccm_weight = self._exp_cfg.get("pearson_dccm_weight", 0.0)
        self.pearson_loss_dccm = pearson_loss_dccm
        
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
        if self._model_cfg.model_type == 'ipa':
            self.model = FlexibilityModelIPA(self._model_cfg)
        else:
            raise ValueError(f"Unknown model type: {self._model_cfg.model_type}")
    
    def on_train_start(self):
        self._epoch_start_time = time.time()

    def loss_fn(self, batch: Any, model_output: Any):
        ''''
        Handles the loss calculation for the flexibility model. Handles node and edge outputs separately.
        Returns a dictionary of loss tensors of shape (num_batch)
        '''
        losses = {"node": {}, "edge": {}}

        def node_loss(node_outs, losses, total_loss):
            '''
            Local helper function to calculate loss for a given scalar profile type.
            '''
            for out in node_outs:
                # B, N, 1
                profile_type = out['name']
                pred_profile = model_output["node"][profile_type]
                target = batch[profile_type]
                
                for loss_fn, weight in self._exp_cfg.node_loss_weights.items():
                    weight_ = self._exp_cfg.node_prediction_weights[profile_type] * weight
                    # determine the term:
                    if loss_fn == 'mse' and weight_ > 0:
                        this_term = self.mse_loss(pred_profile, target).mean(dim=-1) * weight_
                    elif loss_fn == 'huber' and weight_ > 0:
                        # B, N, 1
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

        node_losses = {'None': torch.tensor(0.0)}
        node_total_loss = torch.tensor(0.0)
        # node_losses, node_total_loss = node_loss(self.model.node_outs, node_losses, node_total_loss)
        losses["node"].update(node_losses)
        losses["node"]["node_total_loss"] = node_total_loss

        def edge_loss(edge_outs, losses, total_loss):
            '''
            for edge loss we apply mse loss.
            '''
            for out in edge_outs:
                profile_type = out['name']
                pred_profile = model_output["edge"][profile_type]
                target_full = batch[profile_type]
                assert target_full.shape == pred_profile.shape

                # TODO: logics of the losses has to be reworked, now ugly with all things i tested
                for loss_fn, weight in self._exp_cfg.edge_loss_weights.items():
                    weight_ = self._exp_cfg.edge_prediction_weights[profile_type] * weight
                    if weight_ <= 0:
                        continue
                    this_term = None
                    
                    if loss_fn == "log_frobenius":
                        if profile_type == "per_res_covariance":
                            this_term = self.frobenius_loss_per_res(pred_profile, target_full, reduction=self.reduction_frobenius) * weight_
                            if self.mse_rmsf:
                                # B, N
                                rmsf_pred = batched_rmsf_from_covar(pred_profile)
                                rmsf_gt = batched_rmsf_from_covar(target_full)
                                mse_rmsf_term = torch.mean((rmsf_pred - rmsf_gt) ** 2) * self.mse_rmsf_weight
                                losses[f"{profile_type}_rmsf_mse"] = mse_rmsf_term
                                this_term = this_term + mse_rmsf_term.mean()

                    elif loss_fn == 'corr_matrix_dist':
                        if profile_type == 'pairwise_couplings':
                            # fix the variable in the reduction
                            # this one is scale invariant. will emphasize the global geometry of the covariance, but not the absolute scale - thus MAE. Works better than log frobenius.
                            this_term = self.cmd_loss(pred_profile, target_full, reduction=self.reduction_frobenius) * weight_
                            if self.mae_cmd:
                                # mean over batch
                                mae_term = torch.mean(torch.abs(pred_profile - target_full)) * self.mae_cmd_weight
                                losses[f"{profile_type}_cmd_mae"] = mae_term
                                this_term = this_term + mae_term.mean()
                            if self.huber_cmd:
                                huber_term = torch.mean(self.huber_loss(pred_profile, target_full)) * self.huber_cmd_weight
                                losses[f"{profile_type}_cmd_huber"] = huber_term
                                this_term = this_term + huber_term.mean()
					
                    elif loss_fn == "mse":
                        if profile_type == "pairwise_DCCM":
                            this_term = self.mse_loss_dccm(pred_profile, target_full).mean() * weight_
                            if self.pearson_dccm:
                                pearson_term = self.pearson_loss_dccm(pred_profile, target_full) * self.pearson_dccm_weight
                                losses[f"{profile_type}_pearson"] = pearson_term
                                this_term = this_term + pearson_term
                        else:
                            continue
                    else:
                        raise ValueError(f"Unknown loss function: {loss_fn}")

                    if this_term is None:
                        continue
                    
                    losses[f"{profile_type}_{loss_fn}"] = this_term
                    total_loss = total_loss + this_term
                    
            return losses, total_loss
        
        edge_losses = {}
        edge_total_loss = torch.tensor(0.0)
        edge_losses, edge_total_loss = edge_loss(self.model.edge_outs, edge_losses, edge_total_loss)
        losses["edge"].update(edge_losses)
        losses["edge"]["edge_total_loss"] = edge_total_loss
        return losses

    def model_step(self, batch: Any):
        model_output = self.model(batch)
        losses = self.loss_fn(batch, model_output)
        return losses, model_output

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        batch_losses, model_output = self.model_step(batch)
        # print(batch.keys())
        num_batch = batch['res_mask'].shape[0]
        total_losses = {}
        total_losses['node_loss'] = {}
        total_losses['edge_loss'] = {}

        for k,v in batch_losses['node'].items():
            total_losses['node_loss'][k] = v.mean()

        for k,v in batch_losses['edge'].items():
            total_losses['edge_loss'][k] = v.mean()
        
        # average over batch:
        train_loss = total_losses["node_loss"]["node_total_loss"] + total_losses["edge_loss"]["edge_total_loss"]

        for k,v in total_losses['node_loss'].items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, on_step=True, batch_size=num_batch)
        for k,v in total_losses['edge_loss'].items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, on_step=True, batch_size=num_batch)

        # TODO: finish here
        with torch.no_grad():
            # node outputs
            for profile_type in self._model_cfg.flexibility.node_outputs:
                profile_type = profile_type['name']
                if profile_type == 'None':
                    continue
                self.train_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
                self.train_predictions[profile_type].extend(model_output['node'][profile_type][...,0].detach().cpu().numpy())

            # edge outputs
            for profile_type in self._model_cfg.flexibility.edge_outputs:
                profile_type = profile_type['name']
                if profile_type == 'None':
                    continue
                # NOTE: might cause issues
                self.train_targets[profile_type].extend(batch[profile_type].detach().cpu().numpy())
                self.train_predictions[profile_type].extend(model_output['edge'][profile_type].detach().cpu().numpy())

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
        train_epoch_metrics = get_metrics(self.train_predictions, self.train_targets, self._model_cfg.flexibility)

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

        # # plot examples
        # if hasattr(self._exp_cfg, 'plot_val_every_n') and self._exp_cfg.plot_val_every_n is not None and self.val_epoch % self._exp_cfg.plot_val_every_n == 0:
        #     self.plot_examples(mode='train')

        self.train_predictions.clear()
        self.train_targets.clear()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):

        num_batch, num_res = batch['trans_1'].shape[:2]
        batch_losses, model_output = self.model_step(batch)

        total_losses = {}
        total_losses['node_loss'] = {}
        total_losses['edge_loss'] = {}
        
        for k,v in batch_losses['node'].items():
            total_losses['node_loss'][k] = v.mean()
        for k,v in batch_losses['edge'].items():
            total_losses['edge_loss'][k] = v.mean()
            
        for k,v in total_losses['node_loss'].items():
            self._log_scalar(f"valid/{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=num_batch)
        for k,v in total_losses['edge_loss'].items():
            self._log_scalar(f"valid/{k}", v, prog_bar=False, on_step=False, on_epoch=True, batch_size=num_batch)

        self._log_scalar("valid/loss", total_losses['node_loss']['node_total_loss'] + total_losses['edge_loss']['edge_total_loss'], prog_bar=False, on_step=False, on_epoch=True, batch_size=num_batch)

        with torch.no_grad():
            # node outputs
            for out in self._model_cfg.flexibility.node_outputs:
                profile_type = out['name']
                if profile_type == 'None':
                    continue
                self.valid_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
                self.valid_predictions[profile_type].extend(model_output['node'][profile_type][...,0].detach().cpu().numpy())
            self.valid_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])

            # edge outputs
            for out in self._model_cfg.flexibility.edge_outputs:
                profile_type = out['name']
                if profile_type == 'None':
                    continue
                # NOTE: might cause issues
                self.valid_targets[profile_type].extend(batch[profile_type].detach().cpu().numpy())
                self.valid_predictions[profile_type].extend(model_output['edge'][profile_type].detach().cpu().numpy())
            self.valid_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])
        
    def on_validation_epoch_end(self):
        # Calculate dict of metric dicts	
        valid_epoch_metrics = get_metrics(self.valid_predictions, self.valid_targets, self._model_cfg.flexibility)

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
        # if hasattr(self._exp_cfg, 'plot_val_every_n') and self._exp_cfg.plot_val_every_n is not None and self.val_epoch % self._exp_cfg.plot_val_every_n == 0:
        #     self.plot_examples(mode='valid')

        self.valid_predictions.clear()
        self.valid_targets.clear()
        self.val_epoch += 1

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):

        num_batch, num_res = batch['trans_1'].shape[:2]
        batch_losses, model_output = self.model_step(batch)

        with torch.no_grad():
            
            for out in self._model_cfg.flexibility.node_outputs:
                profile_type = out['name']
                if profile_type == 'None':
                    continue
                self.test_targets[profile_type].extend(batch[profile_type][...,0].detach().cpu().numpy())
                self.test_predictions[profile_type].extend(model_output['node'][profile_type][...,0].detach().cpu().numpy())
            self.test_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])
            
            for out in self._model_cfg.flexibility.edge_outputs:
                profile_type = out['name']
                if profile_type == 'None':
                    continue
                # NOTE: not sure if that's correct - need to debug; in the old branch it was 1:1 as in nodes
                self.test_targets[profile_type].extend(batch[profile_type].detach().cpu().numpy())
                self.test_predictions[profile_type].extend(model_output['edge'][profile_type].detach().cpu().numpy())
            self.test_targets['pdb_name'].extend([str(batch['pdb_name'][i]) if 'pdb_name' in batch else 'Unknown' for i in range(num_batch)])
    
    def on_test_epoch_start(self):
        self.test_predictions.clear()
        self.test_targets.clear()

    def on_test_epoch_end(self):
        """
        At the end of the test epoch, predictions, targets and metrics are stored in the class under attributes test_predictions, test_targets and test_epoch_metrics.
        """
        # Calculate dict of metric dicts
        self.test_metrics = get_metrics(self.test_predictions, self.test_targets, self._model_cfg.flexibility)

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