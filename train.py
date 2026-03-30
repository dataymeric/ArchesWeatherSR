"""Main script to run pipeline for training or inference (specify `mode` arg as "train" or "test").

Arguments are configured with Hydra, which reads the `configs/` folder to compose the config.
You can change arguments either by modifying the config files or through command-line.

Example:
    python -m geoarches.main_hydra \
    module=archesweather \\ # Uses configs/module/archesweather.yaml
    dataloader=era5 \\ # Uses configs/dataloader/era5.yaml
    ++name=default_run \\ # Name of run, used for logging and saving checkpoints
"""

import os
import signal
import warnings
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.data import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader

from archesweathersr.utils.logging_utils import setup_logger

console_logger = setup_logger("geoarches.main_hydra", level="INFO")


def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: collate_fn([d[k] for d in batch]) for k in elem}
    if isinstance(elem, TensorDict):
        fields = {k: collate_fn([td.get(k) for td in batch]) for k in elem.keys()}
        new_bs = torch.Size([len(batch), *elem.batch_size])
        return TensorDict(
            fields, batch_size=new_bs, device=elem.device, non_blocking=True
        )
    return default_collate(batch)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:  # noqa E722
        pass

    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings(
        action="ignore", message="No device id is provided", category=UserWarning
    )
    console_logger.info(f"Working directory: {os.getcwd()}")

    main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    console_logger.info(f"Main node status: {main_node}")

    exp_logger = None
    ckpt_path = None

    # Check if experiment already exists and try to resume
    console_logger.info("Checking if experiment exists...")
    ckpt_dir = Path(cfg.exp_dir).joinpath("checkpoints")
    if ckpt_dir.exists():
        console_logger.info("Experiment already exists: trying to resume it.")
        exp_cfg = OmegaConf.load(Path(cfg.exp_dir) / "config.yaml")
        if cfg.resume or cfg.mode == "test":
            cfg.module = exp_cfg.module
            cfg.dataloader = exp_cfg.dataloader
            cfg.cluster = exp_cfg.cluster
            console_logger.info(f"Hydra config:\n{cfg}")
            try:
                cli_overrides = HydraConfig.get().overrides.task
                console_logger.info("Got CLI arguments from direct launch.")
            except:  # noqa E722
                cli_overrides = getattr(cfg, "cli_overrides", [])
            cli_overrides = [
                x.removeprefix("++") for x in cli_overrides if x.startswith("+")
            ]
            OmegaConf.set_struct(cfg, False)
            cfg.merge_with_dotlist(cli_overrides)
            console_logger.info(f"Updated config:\n{cfg}")
        else:
            OmegaConf.resolve(cfg)
            if cfg.module != exp_cfg.module:
                console_logger.error("Module config mismatch. Exiting...")
                console_logger.error(f"Old config:\n{exp_cfg.module}")
                console_logger.error(f"New config:\n{cfg.module}")
                return

            if cfg.dataloader != exp_cfg.dataloader:
                console_logger.error("Dataloader config mismatch. Exiting...")
                console_logger.error(f"Old config:\n{exp_cfg.dataloader}")
                console_logger.error(f"New config:\n{cfg.dataloader}")
                return

        console_logger.info("Trying to find checkpoints...")
        ckpts = sorted(ckpt_dir.iterdir(), key=os.path.getmtime)
        if ckpts:
            console_logger.info(f"Found checkpoints: {ckpts}")
            if hasattr(cfg, "ckpt_filename_match"):
                ckpts = [x for x in ckpts if str(cfg.ckpt_filename_match) in x.name]
            ckpt_path = ckpts[-1]
            console_logger.info(f"Using checkpoint: {ckpt_path}")

    if cfg.log:
        console_logger.info("Setting up WandB logger...")
        console_logger.info(f"WandB mode: {cfg.cluster.wandb_mode}")
        console_logger.info(
            f"WandB service status: {os.environ.get('WANDB_DISABLE_SERVICE', 'variable unset')}"
        )
        run_id = cfg.name
        exp_logger = loggers.WandbLogger(
            **(
                dict(entity=cfg.entity) if hasattr(cfg, "entity") and cfg.entity else {}
            ),
            project=cfg.project,
            name=cfg.name,
            id=run_id,
            save_dir="wandblogs",
            offline=(cfg.cluster.wandb_mode != "online"),
            resume="allow",
        )

    if cfg.log and main_node and not Path(cfg.exp_dir).joinpath("checkpoints").exists():
        console_logger.info("Registering experiment on main node.")
        hparams = OmegaConf.to_container(cfg, resolve=True)
        exp_logger.log_hyperparams(hparams)
        Path(cfg.exp_dir).mkdir(exist_ok=True, parents=True)
        with open(Path(cfg.exp_dir) / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.mode == "train":
        console_logger.info("Setting up validation dataloader...")
        val_args = getattr(cfg.dataloader, "validation_args", {})
        val_set = instantiate(cfg.dataloader.dataset, **val_args)
        val_loader = StatefulDataLoader(
            val_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            snapshot_every_n_steps=cfg.save_step_frequency,
        )

        console_logger.info("Setting up training dataloader...")
        train_set = instantiate(cfg.dataloader.dataset)
        train_loader = StatefulDataLoader(
            train_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            snapshot_every_n_steps=cfg.save_step_frequency,
        )

    elif cfg.mode == "test":
        console_logger.info("Setting up test dataloader...")
        test_args = getattr(cfg.dataloader, "test_args", {})
        test_set = instantiate(cfg.dataloader.dataset, **test_args)
        test_loader = StatefulDataLoader(
            test_set,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,  # avoid correlated batches
            collate_fn=collate_fn,
            persistent_workers=True,
            snapshot_every_n_steps=cfg.save_step_frequency,
        )

    OmegaConf.resolve(cfg.module)
    pl_module = instantiate(cfg.module.module, cfg.module)

    if hasattr(cfg, "load_ckpt"):
        load_ckpt_dir = Path(cfg.load_ckpt).joinpath("checkpoints")
        load_ckpt_path = sorted(load_ckpt_dir.iterdir(), key=os.path.getmtime)[-1]
        pl_module.load_state_dict(
            torch.load(load_ckpt_path, map_location="cpu")["state_dict"]
        )

    checkpointer = ModelCheckpoint(
        every_n_train_steps=cfg.save_step_frequency,
        save_last=True,
        save_top_k=20,
        monitor="step",
        mode="max",
        dirpath=f"{cfg.exp_dir}/checkpoints/",
        filename="checkpoint_{step}",
    )

    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        strategy="ddp" if torch.cuda.is_available() else "auto",
        num_nodes=getattr(cfg.cluster, "num_nodes", 1),
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        gradient_clip_val=1,
        max_steps=cfg.max_steps,
        enable_checkpointing=True,
        callbacks=[TQDMProgressBar(), checkpointer],
        logger=exp_logger,
        plugins=[SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)],
        limit_train_batches=getattr(cfg, "limit_train_batches", 1.0),
        limit_val_batches=getattr(cfg, "limit_val_batches", 1.0),
        limit_test_batches=getattr(cfg, "limit_test_batches", cfg.limit_val_batches),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=1,
    )

    if cfg.mode == "train":
        console_logger.info("Starting training...")
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)
    elif cfg.mode == "test":
        trainer.test(pl_module, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
