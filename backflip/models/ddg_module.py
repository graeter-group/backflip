# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule

from backflip.models.ddg_metrics import rmse, pearson_corr, spearman_corr
from backflip.models.ddg_model import DDGPredictor, NoIPAMLP


class DDGModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data

        self._run_dir = Path(self._exp_cfg.run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_csv = self._run_dir / "metrics.csv"
        self._test_metrics_json = self._run_dir / "test_metrics.json"
        self._test_preds_csv = self._run_dir / "test_preds.csv"

        self._ddg_model = self._build_model()
        self._loss_fn = self._build_loss()

        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.test_meta = []
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

        self.save_hyperparameters()

    def _build_model(self):
        ddg_conf = getattr(self._model_cfg, 'ddg', None)
        model_type = 'ipa'
        if ddg_conf is not None and hasattr(ddg_conf, 'model_type'):
            model_type = ddg_conf.model_type
        if model_type == 'noipa':
            return NoIPAMLP(self._model_cfg)
        return DDGPredictor(self._model_cfg)

    def _build_loss(self):
        loss_type = getattr(self._exp_cfg, 'ddg_loss', 'mse')
        if loss_type == 'huber':
            beta = getattr(self._exp_cfg, 'huber_beta', 1.0)
            return torch.nn.SmoothL1Loss(beta=beta)
        return torch.nn.MSELoss()

    def configure_optimizers(self):
        lr = self._exp_cfg.optimizer.lr
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, batch):
        return self._ddg_model(batch)

    def _log_epoch_metrics(self, split, preds, targets, loss_val=None):
        metrics = {
            "rmse": rmse(preds, targets),
            "pearson": pearson_corr(preds, targets),
            "spearman": spearman_corr(preds, targets),
        }
        if loss_val is not None:
            metrics["loss"] = float(loss_val)

        for name, val in metrics.items():
            self.log(f"{split}/{name}", val, on_epoch=True, prog_bar=False, logger=True)

        row = {"epoch": int(self.current_epoch), "split": split, **metrics}
        self._append_metrics_csv(row)
        return metrics

    def _append_metrics_csv(self, row):
        write_header = not self._metrics_csv.exists()
        df = pd.DataFrame([row])
        df.to_csv(self._metrics_csv, mode="a", header=write_header, index=False)

    def _accumulate(self, pred, target, split, meta=None):
        pred = pred.detach().cpu().numpy().astype(np.float32)
        target = target.detach().cpu().numpy().astype(np.float32)
        if split == "train":
            self.train_preds.extend(pred.tolist())
            self.train_targets.extend(target.tolist())
        elif split == "val":
            self.val_preds.extend(pred.tolist())
            self.val_targets.extend(target.tolist())
        elif split == "test":
            self.test_preds.extend(pred.tolist())
            self.test_targets.extend(target.tolist())
            if meta is not None:
                self.test_meta.extend(meta)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch["ddg"].float()
        loss = self._loss_fn(pred, target)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self._train_loss_sum += float(loss.detach().cpu())
        self._train_loss_count += 1
        self._accumulate(pred, target, split="train")
        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return
        loss_val = None
        if self._train_loss_count > 0:
            loss_val = self._train_loss_sum / self._train_loss_count
        self._log_epoch_metrics("train", self.train_preds, self.train_targets, loss_val=loss_val)
        self.train_preds.clear()
        self.train_targets.clear()
        self._train_loss_sum = 0.0
        self._train_loss_count = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch)
        target = batch["ddg"].float()
        loss = self._loss_fn(pred, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._val_loss_sum += float(loss.detach().cpu())
        self._val_loss_count += 1
        self._accumulate(pred, target, split="val")
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        loss_val = None
        if self._val_loss_count > 0:
            loss_val = self._val_loss_sum / self._val_loss_count
        self._log_epoch_metrics("val", self.val_preds, self.val_targets, loss_val=loss_val)
        self.val_preds.clear()
        self.val_targets.clear()
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch)
        target = batch["ddg"].float()
        self._accumulate(pred, target, split="test", meta=batch.get("meta"))
        return pred

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        metrics = self._log_epoch_metrics("test", self.test_preds, self.test_targets)
        with open(self._test_metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)

        rows = []
        for meta, pred, target in zip(self.test_meta, self.test_preds, self.test_targets):
            row = {
                "protein_id": meta.get("protein_id"),
                "mutation": meta.get("mutation"),
                "ddg": float(target),
                "pred": float(pred),
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(self._test_preds_csv, index=False)

        self.test_preds.clear()
        self.test_targets.clear()
        self.test_meta.clear()
