import os
from pathlib import Path

import hydra
import torch
from typing import Any
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from hydra.utils import get_original_cwd

from gafl.experiment_utils import get_pylogger, flatten_dict

from backflip.data.ddg_dataloader import DDGDataModule
from backflip.models.ddg_module import DDGModule

log = get_pylogger(__name__)
torch.set_float32_matmul_precision("high")


def _prepare_run_dirs(cfg):
    base = Path(get_original_cwd())
    run_dir = base / cfg.experiment.run_dir
    ckpt_dir = base / cfg.experiment.checkpointer.dirpath
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.experiment.run_dir = str(run_dir)
    cfg.experiment.checkpointer.dirpath = str(ckpt_dir)
    if "default_root_dir" in cfg.experiment.trainer:
        cfg.experiment.trainer.default_root_dir = str(base / cfg.experiment.trainer.default_root_dir)
    return run_dir


@hydra.main(version_base=None, config_path="../configs", config_name="train_ddg")
def main(cfg: DictConfig):
    if getattr(cfg.experiment, "dev_run", False):
        OmegaConf.set_struct(cfg.experiment.trainer, False)
        cfg.experiment.use_wandb = False
        cfg.experiment.trainer.max_epochs = 1
        cfg.experiment.trainer.limit_train_batches = 1
        cfg.experiment.trainer.limit_val_batches = 1
        cfg.experiment.trainer.limit_test_batches = 1
        cfg.data.loader.num_workers = 0
        OmegaConf.set_struct(cfg.experiment.trainer, True)
    seed_everything(cfg.experiment.seed, workers=True)

    run_dir = _prepare_run_dirs(cfg)

    logger = None
    if cfg.experiment.use_wandb:
        logger = WandbLogger(**cfg.experiment.wandb)

    callbacks = [ModelCheckpoint(**cfg.experiment.checkpointer)]
    if hasattr(cfg.experiment, "early_stopping") and cfg.experiment.early_stopping is not None:
        callbacks.append(EarlyStopping(**cfg.experiment.early_stopping))

    accelerator = cfg.experiment.trainer.get("accelerator", "gpu")
    if accelerator == "cpu":
        devices = 1
    else:
        try:
            import GPUtil  # noqa: WPS433
            devices = GPUtil.getAvailable(order="memory", limit=8)[: cfg.experiment.num_devices]
        except Exception:
            devices = cfg.experiment.num_devices
    log.info(f"Using devices: {devices}")

    datamodule = DDGDataModule(cfg.data)
    model = DDGModule(cfg)

    trainer = Trainer(
        **cfg.experiment.trainer,
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        enable_progress_bar=cfg.experiment.use_tqdm,
        enable_model_summary=True,
        devices=devices,
    )

    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f.name)
    if logger is not None and hasattr(logger, "experiment"):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = dict(flatten_dict(cfg_dict))
        logger.experiment.config.update(flat_cfg)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.experiment.warm_start)

    best_path = callbacks[0].best_model_path
    if best_path is None or best_path == "":
        best_path = os.path.join(cfg.experiment.checkpointer.dirpath, "best.ckpt")
    log.info(f"Evaluating best checkpoint at {best_path}")
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
