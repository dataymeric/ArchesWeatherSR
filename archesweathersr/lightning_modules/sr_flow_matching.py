import importlib.resources
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import diffusers
import geoarches.stats as geoarches_stats
import pandas as pd
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from geoarches.dataloaders import era5
from geoarches.lightning_modules import BaseLightningModule
from hydra.utils import instantiate
from rich.pretty import pretty_repr
from tensordict.tensordict import TensorDict
from tqdm import tqdm

import archesweathersr.stats as archesweathersr_stats
from archesweathersr.backbones.dit import TimestepEmbedder
from archesweathersr.utils.tensordict_utils import (
    tensordict_apply,
    tensordict_cat,
    tensordict_interp,
)

geoarches_stats_path = importlib.resources.files(geoarches_stats)
archesweathersr_stats_path = importlib.resources.files(archesweathersr_stats)


class DownscalingDiffusionModule(BaseLightningModule):
    """Flow matching module for statistical downscaling of weather forecasts."""

    def __init__(
        self,
        cfg,
        name="superres",
        cond_dim=32,
        num_train_timesteps=1000,
        scheduler="euler",
        prediction_type="sample",
        conditional="",
        state_normalization="residual",
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        sd3_timestep_sampling=True,
        interp_args={"mode": "bicubic", "align_corners": False},
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)
        self.embedder = instantiate(cfg.embedder)

        # cond_dim should be given as arg to the backbone
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)

        if scheduler in ["flow", "euler"]:
            self.console_logger.info("Using Euler solver")
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

        self.inference_scheduler = deepcopy(self.noise_scheduler)

        deg = 0.25  # degree step for area weights, hardcoded to 0.25 deg
        area_weights = torch.arange(-90, 90 + 1e-6, deg).mul(torch.pi / 180).cos()
        area_weights = (area_weights / area_weights.mean())[:, None]  # lat coeffs

        self.val_metrics = nn.ModuleList(
            [
                instantiate(metric, **cfg.val.metrics_kwargs)
                for metric in cfg.val.metrics
            ]
        )
        self.test_metrics = nn.ModuleDict(
            {
                metric_name: instantiate(metric, **cfg.inference.metrics_kwargs)
                for metric_name, metric in cfg.inference.metrics.items()
            }
        )

        pressure_levels = torch.tensor(era5.pressure_levels).float()
        vertical_coeffs = (pressure_levels / pressure_levels.mean()).reshape(-1, 1, 1)

        total_coeff = 6 + 4  # same ponderation for each variable
        surface_coeffs = 4 * torch.tensor(1).reshape(-1, 1, 1, 1)
        level_coeffs = 6 * torch.tensor(1).reshape(-1, 1, 1, 1)

        self.register_buffer(
            "loss_surface_coeffs", area_weights * surface_coeffs / total_coeff
        )
        self.register_buffer(
            "loss_level_coeffs",
            area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )

        if state_normalization == "residual":
            era5_stats = torch.load(
                str(
                    archesweathersr_stats_path / "era5_1440x721_norm_stats_1979-2018.pt"
                ),
                weights_only=True,
            )
            self.era5_mean = TensorDict(
                level=era5_stats["level"]["mean"], surface=era5_stats["surface"]["mean"]
            )
            self.era5_std = TensorDict(
                level=era5_stats["level"]["std"], surface=era5_stats["surface"]["std"]
            )
            scaler_file = (
                str(archesweathersr_stats_path / "sr_residual_norm_bicubic.pt")
                if interp_args["mode"] == "bicubic"
                else str(archesweathersr_stats_path / "sr_residual_norm_linear.pt")
            )
            scaler = TensorDict(
                **torch.load(
                    str(archesweathersr_stats_path / scaler_file), weights_only=False
                )
            )
            self.state_scaler = (
                scaler / self.era5_std
            )  # inverse: we divide by state_scaler

        self.conditional_keys = self.conditional.split("+") if self.conditional else []

    def forward(self, batch, noisy_state, timesteps, is_sampling=False):
        device = batch["state"]["lowres"].device

        up_lowres_state = tensordict_interp(
            batch["state"]["lowres"],
            target=batch["state"]["highres"],
            **self.interp_args,
        )

        x_noisy = noisy_state
        if "teacher" in self.conditional_keys:
            # teacher forcing: in training use previous highres, autoregressive at inference
            x_noisy = tensordict_cat([batch["prev_state"]["highres"], x_noisy], dim=1)

        times = pd.to_datetime(batch["timestamp"].cpu().numpy() * 10**9).tz_localize(
            None
        )
        month = torch.tensor(times.month).to(device)
        hour = torch.tensor(times.hour).to(device)

        cond_emb = (
            self.month_embedder(month)
            + self.hour_embedder(hour)
            + self.timestep_embedder(timesteps)
        )

        x = self.embedder.encode(up_lowres_state, x_noisy)
        x = self.backbone(x, cond_emb)
        out = self.embedder.decode(x)

        if is_sampling and self.prediction_type == "sample":
            sigmas = timesteps / self.noise_scheduler.config.num_train_timesteps
            sigmas = sigmas[:, None, None, None, None]
            out = (noisy_state - out).apply(lambda x: x / sigmas)

        return out

    def training_step(self, batch, batch_nb):
        device, bs = batch["state"]["lowres"].device, batch["state"]["lowres"].shape[0]

        if self.sd3_timestep_sampling:
            u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        else:
            indices = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bs,)
            ).long()

        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = self.noise_scheduler.timesteps[indices].to(device)

        sigmas = self.noise_scheduler.sigmas.to(
            device=device, dtype=batch["state"]["lowres"].dtype
        )
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()[:, None, None, None, None]

        noise = torch.randn_like(batch["state"]["highres"])

        up_lowres_state = tensordict_interp(
            batch["state"]["lowres"],
            target=batch["state"]["highres"],
            **self.interp_args,
        )
        residual = batch["state"]["highres"] - up_lowres_state  # r_t = y_t - x̂_t

        if self.state_normalization:
            residual = tensordict_apply(
                torch.div, residual, self.state_scaler.to(self.device)
            )

        noisy_state = noise.apply(lambda x: x * sigma) + residual.apply(
            lambda x: x * (1.0 - sigma)
        )
        target_state = (
            residual if self.prediction_type == "sample" else noise - residual
        )

        pred = self.forward(batch, noisy_state, timesteps)
        loss = self.loss(pred, target_state, timesteps)

        self.mylog(loss=loss)

        return loss

    def loss(self, pred, gt, timesteps=None, **kwargs):
        loss_coeffs = TensorDict(
            surface=self.loss_surface_coeffs, level=self.loss_level_coeffs
        )
        if self.prediction_type == "sample":
            sigmas = timesteps / self.noise_scheduler.config.num_train_timesteps
            snr_weights = (1 - sigmas) / sigmas
            snr_weights = snr_weights.to(self.device)[:, None, None, None, None]
            loss_coeffs = loss_coeffs.apply(lambda x: x * snr_weights)

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)
        return sum(weighted_error.mean().values())

    @torch.no_grad()
    def sample(
        self,
        batch,
        seed=None,
        num_steps=None,
        disable_tqdm=False,
        scale_input_noise=None,
        **kwargs,
    ):
        """Sample a downscaled state. kwargs are forwarded to the scheduler step."""
        scheduler = self.inference_scheduler
        num_steps = num_steps or self.cfg.inference.num_steps
        scheduler.set_timesteps(num_steps)

        scheduler_kwargs = dict(s_churn=self.cfg.inference.s_churn)
        scheduler_kwargs.update(kwargs)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        up_lowres_state = tensordict_interp(
            batch["state"]["lowres"],
            target=batch["state"]["highres"],
            **self.interp_args,
        )

        xt = up_lowres_state.apply(
            lambda x: torch.empty_like(x).normal_(generator=generator)
        )

        scale_input_noise = scale_input_noise or getattr(
            self.cfg.inference, "scale_input_noise", None
        )
        if scale_input_noise is not None:
            xt = xt * scale_input_noise

        for t in tqdm(scheduler.timesteps, disable=disable_tqdm):
            pred = self.forward(
                batch, xt, timesteps=torch.tensor([t]).to(self.device), is_sampling=True
            )

            # Preserve step_index to work around a quirk in the scheduler
            step_index = getattr(scheduler, "_step_index", None)

            def scheduler_step(*args, _si=step_index, **kwargs):
                out = scheduler.step(*args, **kwargs)
                if hasattr(scheduler, "_step_index"):
                    scheduler._step_index = _si
                return out.prev_sample

            xt = tensordict_apply(scheduler_step, pred, t, xt, **scheduler_kwargs)

            if step_index is not None:
                scheduler._step_index = step_index + 1

        out = xt.detach()

        if self.state_normalization:
            out = tensordict_apply(torch.mul, out, self.state_scaler.to(self.device))

        return up_lowres_state + out

    def validation_step(self, batch, batch_nb):
        val_num_members = self.cfg.val.num_members

        samples = [
            self.sample(batch, seed=j + batch_nb * 10**6, disable_tqdm=True).unsqueeze(
                1
            )
            for j in tqdm(range(val_num_members))
        ]

        denormalize = self.trainer.val_dataloaders.dataset.denormalize
        targets = denormalize(batch["state"]["highres"].unsqueeze(1))
        preds = [denormalize(s) for s in samples]

        for metric in self.val_metrics:
            metric.update(targets, preds)

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            scores = metric.compute()
            self.log_dict(scores, sync_dist=True)
            self.console_logger.info(pretty_repr(scores))
            metric.reset()

    def on_test_epoch_start(self):
        dataset = self.trainer.test_dataloaders.dataset

        suffix = getattr(self.cfg.inference, "test_filename_suffix", "")
        now = datetime.today().strftime("%m%d%H%M")
        self.test_filename = (
            f"{dataset.domain}-{now}"
            f"-num_steps={self.cfg.inference.num_steps}"
            f"-members={self.cfg.inference.num_members}"
            f"-{suffix}.pt"
        )
        Path("evalstore").joinpath(self.name).mkdir(exist_ok=True, parents=True)

    def test_step(self, batch, batch_nb):
        dataset = self.trainer.test_dataloaders.dataset
        num_members = self.cfg.inference.num_members

        samples = [
            self.sample(batch, seed=j + batch_nb * 10**6, disable_tqdm=True).unsqueeze(
                1
            )
            for j in range(num_members)
        ]

        for metric in self.test_metrics.values():
            metric.update(
                dataset.denormalize(batch["state"]["highres"].unsqueeze(1)),
                [dataset.denormalize(s) for s in samples],
            )

    def on_test_epoch_end(self):
        all_metrics = {}
        for metric in self.test_metrics.values():
            scores = metric.compute()
            self.log_dict(scores, sync_dist=True)
            all_metrics.update(scores)
            metric.reset()
        all_metrics["hparams"] = dict(self.cfg.inference)

        torch.save(all_metrics, Path("evalstore") / self.name / self.test_filename)

    def configure_optimizers(self):
        self.console_logger.info("Configuring optimizers...")
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        sched = {
            "scheduler": sched,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [opt], [sched]
