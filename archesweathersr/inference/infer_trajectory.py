"""
AWG+SR trajectory rollout.

Processes exactly ONE TASK (time slice) and generates multiple ensemble members.
Each member:
 - runs AWG rollout for n_steps
 - applies SR to each step
 - regrids back to lowres
 - feeds regridded output back to AWG

Usage (example):
  python infer_trajectory.py --task-id 0
"""

import argparse
import glob
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import xarray_regrid  # noqa: F401
from tensordict import TensorDict
from tqdm import tqdm
from xarray_regrid import Grid

sys.path.append(str(Path(__file__).parent.parent))

from geoarches.dataloaders.era5 import Era5Forecast, level_variables, surface_variables
from geoarches.lightning_modules.base_module import load_module

# ---------------- config ----------------
SCRATCH_DIR = os.getenv("SCRATCH")
DEFAULT_INPUTS_GLOB = f"{SCRATCH_DIR}/pretrained/evalstore/archesweathergen_rollouts/archesweathergen_rollouts_task*.nc"
DEFAULT_OUTPUT_DIR = f"{SCRATCH_DIR}/pretrained/evalstore/awg_sr_bicubic_trajectories"
DEFAULT_ERA5_PATH = f"{SCRATCH_DIR}/era5/240x121/weatherbench2/4xyearly/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_ENSEMBLE_MEMBERS = 10  # Number of ensemble members to generate
NUM_ROLLOUT_STEPS = 10  # Number of AWG+SR rollout steps
LEAD_TIME_HOURS = 24  # Hours per step
DEFAULT_SR_MODULE = "downscaling-era5-2"


# ----------------------------------------


def prepare_sr_inputs_for_model(batch_tdict: TensorDict, timestamps_sec: torch.Tensor):
    """
    Prepare SR model inputs from TensorDict and timestamp.
    Common pattern: pass 'surface', 'level', and 'timestamp' (seconds).
    """
    input = {
        "timestamp": timestamps_sec.to(DEVICE),
        "state": {
            "lowres": TensorDict(
                {
                    "surface": batch_tdict["surface"],
                    "level": batch_tdict["level"],
                }
            ).to(DEVICE),
            "highres": TensorDict(
                {
                    "surface": torch.randn(1, 4, 1, 721, 1440),
                    "level": torch.randn(1, 6, 13, 721, 1440),
                }
            ).to(DEVICE),
        },
    }
    return input


def make_target_regrid_dataset(res_lat=1.5, res_lon=1.5):
    """Create target grid for conservative regridding."""
    grid = Grid(
        north=90,
        east=360 - res_lon,
        south=-90,
        west=0,
        resolution_lat=res_lat,
        resolution_lon=res_lon,
    )
    return grid.create_regridding_dataset()


@torch.no_grad()
def rollout_awg_sr_regrid(
    dataset,  # Era5Forecast dataset
    task_idx,  # Task index to process
    n_steps=10,  # number of steps to generate (t1..t_n_steps)
    lead_time_hours=24,  # hours per step
    device="cuda",
    # two "era5_like" objects with different stats:
    era5_awg_like=None,
    era5_sr_like=None,
    # models / helpers
    awg=None,
    sr_model=None,
    prepare_sr_inputs_for_model=None,
    # regrid
    regrid_target_dataset=None,
    regrid_vars=None,
    skipna=False,
    latitude_coord="latitude",
    # seeds
    member=0,
    batch_nb=0,
    seed_base_awg=123_000,
    seed_base_sr=None,
    # awg kwargs
    awg_kwargs=None,
):
    """
    Pipeline:
      phys(t) -> AWG_norm(t) -> AWG predicts AWG_norm(t+1)
      -> phys(t+1) -> SR_norm(t+1) -> SR predicts SR_norm_hi(t+1)
      -> phys_hi(t+1) -> regrid -> phys_low(t+1)
      -> AWG_norm_low(t+1) fed as state for next step

    Stores (lists):
      lowres_awg_norm_xr : lowres, AWG-normalized, for debugging
      lowres_phys_xr     : lowres, physical
      sr_high_phys_xr    : highres, physical
      sr_regrid_phys_xr  : lowres, physical (after regrid)
    """
    if regrid_target_dataset is None:
        regrid_target_dataset = make_target_regrid_dataset(1.5, 1.5)

    if regrid_vars is None:
        regrid_vars = level_variables + surface_variables

    if seed_base_sr is None:
        seed_base_sr = task_idx * 1000

    out = {
        "lowres_awg_norm_xr": [],
        "lowres_phys_xr": [],
        "sr_high_phys_xr": [],
        "sr_regrid_phys_xr": [],
    }

    # time delta
    dt_sec = int(lead_time_hours * 3600)

    # AWG loop batch at t0
    loop_batch_awg = {
        k: v[None].to(device) for k, v in dataset[task_idx].items()
    }  # t-1, t

    # ---- rollout steps
    for step in tqdm(
        range(n_steps), desc=f"Member {member}: AWG → SR → regrid → AWG", disable=False
    ):
        # 1) AWG predicts next state in AWG-normalized space
        seed_awg = int(seed_base_awg + member * 1000 + batch_nb * 10**6 + step)
        td_next_awg = awg.sample(
            loop_batch_awg, seed=seed_awg, disable_tqdm=True, scale_input_noise=1.05
        )  # t+1

        if not isinstance(td_next_awg, TensorDict):
            td_next_awg = TensorDict(td_next_awg, batch_size=[1])
        td_next_awg = td_next_awg.to(device)

        # advance time to t_{step+1}
        vt_sec = loop_batch_awg["timestamp"] + dt_sec

        out["lowres_awg_norm_xr"].append(
            era5_awg_like.convert_to_xarray(td_next_awg, vt_sec)
        )

        # 2) AWG denorm -> physical lowres (t_{step+1})
        td_next_phys = era5_awg_like.denormalize(td_next_awg)
        xr_next_phys = era5_awg_like.convert_to_xarray(td_next_phys, vt_sec)
        out["lowres_phys_xr"].append(xr_next_phys)

        # 3) physical lowres -> SR-normalized lowres, then SR sample
        td_next_sr = era5_sr_like.convert_to_tensordict(xr_next_phys.isel(time=0))
        td_next_sr = era5_sr_like.normalize(td_next_sr)
        td_next_sr = TensorDict(
            {k: v.unsqueeze(0).to(device) for k, v in td_next_sr.items()},
            batch_size=[1],
        )

        sr_inputs = prepare_sr_inputs_for_model(td_next_sr, vt_sec)

        seed_sr = int(seed_base_sr + member * 1000 + batch_nb * 10**6 + step * 10)
        sr_out_srnorm = sr_model.sample(
            sr_inputs, seed=seed_sr, disable_tqdm=True
        )  # t+1, SR-normalized HIGHRES

        # 4) SR denorm -> physical highres
        sr_out_phys = era5_sr_like.denormalize(sr_out_srnorm)
        xr_sr_phys = era5_sr_like.convert_to_xarray(sr_out_phys, vt_sec)
        out["sr_high_phys_xr"].append(xr_sr_phys)

        # 5) regrid physical highres -> physical lowres
        xr_rg_phys = xr.Dataset()
        for var in regrid_vars:
            xr_rg_phys[var] = xr_sr_phys[var].regrid.conservative(
                regrid_target_dataset,
                skipna=skipna,
                latitude_coord=latitude_coord,
            )
        out["sr_regrid_phys_xr"].append(xr_rg_phys)  # t+1, SR regrid to 1.5°

        # 6) physical lowres (regridded) -> AWG-normalized lowres for next AWG step
        td_rg_awg = era5_awg_like.convert_to_tensordict(xr_rg_phys.isel(time=0))
        td_rg_awg = era5_awg_like.normalize(td_rg_awg)
        td_rg_awg = TensorDict(
            {k: v.unsqueeze(0).to(device) for k, v in td_rg_awg.items()}, batch_size=[1]
        )

        loop_batch_awg = {
            "timestamp": vt_sec,
            "lead_time_hours": loop_batch_awg["lead_time_hours"],
            "state": td_rg_awg.to(device),
            "prev_state": loop_batch_awg["state"],
        }

        # Free GPU memory
        del (
            td_next_awg,
            td_next_phys,
            td_next_sr,
            sr_inputs,
            sr_out_srnorm,
            sr_out_phys,
            td_rg_awg,
        )
        torch.cuda.empty_cache()

    return out


def save_member_trajectories(
    out: dict,
    temp_dir: Path,
    task_idx: int,
    member_idx: int,
):
    """
    Save trajectories for a single member to temporary files.
    Returns paths to saved files.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save lowres trajectory
    lowres_traj = xr.concat(out["lowres_phys_xr"], dim="time")
    lowres_traj = lowres_traj.sel(level=[300, 500, 700, 850])
    paths["lowres"] = temp_dir / f"task{task_idx:03d}_member{member_idx:02d}_lowres.nc"
    lowres_traj.to_netcdf(paths["lowres"])
    lowres_traj.close()
    del lowres_traj

    # Save highres trajectory
    sr_traj_hi = xr.concat(out["sr_high_phys_xr"], dim="time")
    sr_traj_hi = sr_traj_hi.sel(level=[300, 500, 700, 850])
    paths["highres"] = (
        temp_dir / f"task{task_idx:03d}_member{member_idx:02d}_highres.nc"
    )
    sr_traj_hi.to_netcdf(paths["highres"])
    sr_traj_hi.close()
    del sr_traj_hi

    # Save regridded trajectory
    sr_traj_rg = xr.concat(out["sr_regrid_phys_xr"], dim="time")
    sr_traj_rg = sr_traj_rg.sel(level=[300, 500, 700, 850])
    paths["regrid"] = temp_dir / f"task{task_idx:03d}_member{member_idx:02d}_regrid.nc"
    sr_traj_rg.to_netcdf(paths["regrid"])
    sr_traj_rg.close()
    del sr_traj_rg

    return paths


def combine_member_trajectories(
    temp_dir: Path,
    out_dir: Path,
    task_idx: int,
    num_members: int,
):
    """
    Load and combine all member trajectories into final datasets.
    Cleans up temporary files after saving.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for traj_type in ["lowres", "highres", "regrid"]:
        member_trajs = []

        for member_idx in range(num_members):
            temp_file = (
                temp_dir / f"task{task_idx:03d}_member{member_idx:02d}_{traj_type}.nc"
            )
            if temp_file.exists():
                ds = xr.open_dataset(temp_file)
                member_trajs.append(ds)

        if member_trajs:
            # Concatenate along ensemble dimension
            combined = xr.concat(
                member_trajs,
                pd.Index(range(len(member_trajs)), name="number"),
            )

            init_time = combined.time.values[0] - np.timedelta64(
                24 * 3600 * 10**9, "ns"
            )

            # time to -> prediction_timedelta
            combined = combined.rename({"time": "prediction_timedelta"})
            combined = combined.assign_coords(
                prediction_timedelta=[
                    timedelta(days=i) for i in range(1, len(member_trajs) + 1)
                ]
            )

            # add time as init_time
            combined = combined.expand_dims(time=[init_time])

            # reorder dims
            combined = combined.transpose("number", "prediction_timedelta", "time", ...)

            # Save final output
            out_file = out_dir / f"task{task_idx:03d}_{traj_type}_traj.nc"
            combined.to_netcdf(out_file)
            print(f"Saved {out_file}")

            # Close datasets
            combined.close()
            for ds in member_trajs:
                ds.close()

    # Clean up temp files
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")


@torch.no_grad()
def main(
    task_id: int = 0,
    inputs_glob: str = DEFAULT_INPUTS_GLOB,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    era5_path: str = DEFAULT_ERA5_PATH,
    num_ensemble_members: int = NUM_ENSEMBLE_MEMBERS,
    num_rollout_steps: int = NUM_ROLLOUT_STEPS,
    lead_time_hours: int = LEAD_TIME_HOURS,
):
    """Main function to run AWG+SR trajectory rollout for a single task."""

    files = sorted(glob.glob(inputs_glob))
    if not files:
        raise FileNotFoundError(f"No files match: {inputs_glob}")

    # Build timestamp index: (file_id, time_id, time_value)
    print("Indexing files...")
    timestamps = []
    for fid, f_path in tqdm(enumerate(files), desc="Indexing files"):
        with xr.open_dataset(f_path) as ds:
            file_stamps = [(fid, i, t) for (i, t) in enumerate(ds.time.to_numpy())]
            timestamps.extend(file_stamps)

    if task_id >= len(timestamps):
        print(f"Task {task_id}: nothing to do (>= {len(timestamps)} time slices)")
        return

    file_id, time_id, time_val = timestamps[task_id]
    in_file = Path(files[file_id])
    in_stem = in_file.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Task {task_id}] Processing file: {in_file}, time index: {time_id} ({time_val})"
    )
    print(
        f"Generating {num_ensemble_members} ensemble members with {num_rollout_steps} steps each"
    )

    # Load ERA5 datasets for normalization
    print("Loading ERA5 datasets...")
    era5_sr_like = Era5Forecast(
        path=str(in_file), domain="all", norm_scheme="era5", load_prev=False
    )
    era5_awg_like = Era5Forecast(
        path=era5_path,
        domain="test_z0012",
        lead_time_hours=lead_time_hours,
        load_prev=True,
        norm_scheme="pangu",
        multistep=1,
    )

    # Load models
    print("Loading models...")
    torch.set_grad_enabled(False)

    sr_model = load_module(
        DEFAULT_SR_MODULE, ckpt_fname="last.ckpt", return_config=False
    )
    sr_model = sr_model.to(DEVICE)
    sr_model.eval()

    awg = load_module("archesweathergen", return_config=False)
    awg = awg.to(DEVICE)
    awg.eval()

    # Create temp directory for intermediate files
    temp_dir = out_dir / f"temp_task{task_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process each ensemble member
    print(f"Starting rollout for task {task_id}...")
    for member_idx in range(num_ensemble_members):
        print(f"\n{'=' * 60}")
        print(f"Processing ensemble member {member_idx + 1}/{num_ensemble_members}")
        print(f"{'=' * 60}")

        out = rollout_awg_sr_regrid(
            dataset=era5_awg_like,
            task_idx=task_id,
            lead_time_hours=lead_time_hours,
            n_steps=num_rollout_steps,
            device=DEVICE,
            era5_sr_like=era5_sr_like,  # Using same normalization for both
            era5_awg_like=era5_awg_like,
            awg=awg,
            sr_model=sr_model,
            prepare_sr_inputs_for_model=prepare_sr_inputs_for_model,
            member=member_idx,
            batch_nb=0,
            seed_base_awg=task_id * 1000,
            seed_base_sr=task_id * 1000,
        )

        # Save this member's trajectories immediately
        save_member_trajectories(
            out=out,
            temp_dir=temp_dir,
            task_idx=task_id,
            member_idx=member_idx,
        )

        # Free memory
        del out
        torch.cuda.empty_cache()

        print(f"Member {member_idx} completed and saved")

    # Combine all members into final output
    print("\nCombining all ensemble members...")
    combine_member_trajectories(
        temp_dir=temp_dir,
        out_dir=out_dir,
        task_idx=task_id,
        num_members=num_ensemble_members,
    )

    print(f"\n[Task {task_id}] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--inputs-glob", type=str, default=DEFAULT_INPUTS_GLOB)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--era5-path", type=str, default=DEFAULT_ERA5_PATH)
    parser.add_argument(
        "--num-ensemble-members", type=int, default=NUM_ENSEMBLE_MEMBERS
    )
    parser.add_argument("--num-rollout-steps", type=int, default=NUM_ROLLOUT_STEPS)
    parser.add_argument("--lead-time-hours", type=int, default=LEAD_TIME_HOURS)
    args = parser.parse_args()
    main(
        task_id=args.task_id,
        inputs_glob=args.inputs_glob,
        output_dir=args.output_dir,
        era5_path=args.era5_path,
        num_ensemble_members=args.num_ensemble_members,
        num_rollout_steps=args.num_rollout_steps,
        lead_time_hours=args.lead_time_hours,
    )
