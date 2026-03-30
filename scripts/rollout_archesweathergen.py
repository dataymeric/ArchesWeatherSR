"""
Run ArchesWeatherGen inference batched by 6 samples per task.

Each task:
  - loads model + dataset
  - processes 6 consecutive dataset indices as one batch
  - runs 10-member rollouts
  - converts each rollout to xarray and writes one NetCDF

Usage (example):
  python scripts/rollout_archesweathergen.py --task-id 0
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.base_module import load_module


# ---------------- helpers ----------------
def collate_six(dataset, start_idx, end_idx):
    """Stack 6 consecutive samples into one batch."""
    items = [dataset[i] for i in range(start_idx, end_idx)]
    keys = items[0].keys()
    batch = {}
    for k in keys:
        vals = [it[k] for it in items]
        batch[k] = torch.stack(vals, dim=0)
    return batch


# ---------------- main ----------------
def main(task_id: int = 0):
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # --- load ---
    module, _cfg = load_module("archesweathergen")
    module = module.to(device).eval()

    dataset = Era5Forecast(
        path="era5/240x121/weatherbench2/4xyearly/",
        domain="test_z0012",
        lead_time_hours=24,
        load_prev=True,
        norm_scheme="pangu",
        multistep=10,
    )

    # --- task slicing ---
    per_task = 6
    n = len(dataset)
    start_idx = min(task_id * per_task, n)
    end_idx = min(start_idx + per_task, n)

    if start_idx >= n:
        print(f"Task {task_id}: nothing to do.")
        return

    print(f"Task {task_id}: running indices [{start_idx}..{end_idx - 1}]")

    # --- build one batch of 6 ---
    batch = collate_six(dataset, start_idx, end_idx)
    batch = {k: v.to(device) for k, v in batch.items()}

    # --- rollout ensemble ---
    members = 10
    iterations = 10
    scale_input_noise = 1.05

    samples = [
        module.sample_rollout(
            batch,
            batch_nb=start_idx,  # not important
            member=j,
            iterations=iterations,
            disable_tqdm=True,
            scale_input_noise=scale_input_noise,
        )
        for j in tqdm(range(members), desc=f"Task {task_id} members")
    ]

    # --- convert + save ---
    xr_dataset_list = [
        dataset.convert_trajectory_to_xarray(
            sample,
            timestamp=batch["timestamp"],
            denormalize=True,
        )
        for sample in samples
    ]

    xr_dataset = xr.concat(
        xr_dataset_list,
        pd.Index(list(range(members)), name="number"),
    )

    output_dir = Path("evalstore/archesweathergen_rollouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"archesweathergen_rollouts_task{task_id:04d}_idx{start_idx}-{end_idx}.nc"
    )

    xr_dataset.to_netcdf(output_file)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    main(task_id=args.task_id)
