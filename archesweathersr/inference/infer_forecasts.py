"""
Super-resolve ArchesWeather rollout files.

Processes exactly ONE TIME SLICE from one NetCDF file,
iterating its (number, prediction_timedelta) samples. For each sample:
 - compute valid_time = time + prediction_timedelta
 - convert to TensorDict via Era5Forecast.convert_to_tensordict()
 - pass timestamp=valid_time (seconds) to SR model
 - save SR results to a NetCDF named after the input file + time index

Usage (example):
  python infer_forecasts.py --task-id 0
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tensordict import TensorDict
from tqdm import tqdm

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.base_module import load_module

# ---------------- config ----------------
DEFAULT_INPUTS_GLOB = "evalstore/archesweathergen/*.nc"
DEFAULT_OUTPUT_DIR = "evalstore/archesweathergen_sr"
DEFAULT_SR_MODULE = "sr_flow_matching"  # Adjust to your actual SR module name
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_SR_SAMPLES = 1  # Number of SR samples to generate per input sample
# ----------------------------------------


def prepare_sr_inputs_for_model(batch_tdict: TensorDict, timestamps_sec: torch.Tensor):
    """
    TODO: adapt to your SR model's expected input structure.
    Common pattern in your stack is to pass 'surface', 'level', and 'timestamp' (seconds).
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


def iter_samples_from_time_slice(ds_time: xr.Dataset):
    """
    Yield (xr_slice, valid_time_npdt64, init_time) for each (number, prediction_timedelta).
    ds_time is already a single-time slice Dataset.
    xr_slice is a single-member/lead slice Dataset (ready for convert_to_tensordict).
    """
    init_time = ds_time["time"].values  # np.datetime64[ns]
    numbers = ds_time["number"].values
    leads = ds_time["prediction_timedelta"].values  # np.timedelta64[ns]

    for ni in range(numbers.shape[0]):
        for li in range(leads.shape[0]):
            lead_td = leads[li]
            valid_time = init_time + lead_td
            state = ds_time.isel(number=ni, prediction_timedelta=li)
            yield state, valid_time, init_time


def save_sr_batch_leadtimes(
    sr_tdict_list: list,
    era5_like: Era5Forecast,
    init_time_sec: int,
    temp_dir: Path,
    member_idx: int,
    sr_sample_idx: int,
    batch_idx: int,
):
    """
    Save a batch of lead times for a specific (member, SR sample) pair.
    Memory-efficient: processes batches of lead times at a time.
    """
    # Stack into single TensorDict with shape [B, T, ...]
    batch_trajectory = TensorDict(
        {
            k: torch.stack(
                [td[k] for td in sr_tdict_list], dim=1
            )  # B, T, var, pl, lat, lon
            for k in sr_tdict_list[0].keys()
        },
        batch_size=[1, len(sr_tdict_list)],
    )

    # Convert to xarray
    timestamp_tensor = torch.tensor([init_time_sec], dtype=torch.int64)
    xr_batch = era5_like.convert_trajectory_to_xarray(
        batch_trajectory,
        timestamp=timestamp_tensor,
        denormalize=True,
        levels=[300, 500, 700, 850],
    )

    # Save to temporary file for this batch
    temp_file = (
        temp_dir
        / f"member{member_idx:02d}_sr{sr_sample_idx:02d}_batch{batch_idx:03d}.nc"
    )
    xr_batch.to_netcdf(temp_file)
    xr_batch.close()
    return temp_file


def combine_batches_into_trajectory(
    temp_dir: Path,
    member_idx: int,
    sr_sample_idx: int,
    num_batches: int,
):
    """
    Combine batch files into a single trajectory file.
    Returns the path to the combined trajectory file.
    """
    batch_files = []
    for bi in range(num_batches):
        temp_file = (
            temp_dir / f"member{member_idx:02d}_sr{sr_sample_idx:02d}_batch{bi:03d}.nc"
        )
        batch_files.append(temp_file)

    # If only one batch, just rename it
    if num_batches == 1:
        traj_file = temp_dir / f"member{member_idx:02d}_sr{sr_sample_idx:02d}.nc"
        batch_files[0].rename(traj_file)
        return traj_file

    # Load and concatenate along prediction_timedelta dimension
    datasets = [xr.open_dataset(f) for f in batch_files]
    combined = xr.concat(datasets, dim="prediction_timedelta")

    # Close individual datasets
    for ds in datasets:
        ds.close()

    # Save combined trajectory
    traj_file = temp_dir / f"member{member_idx:02d}_sr{sr_sample_idx:02d}.nc"
    combined.to_netcdf(traj_file)
    combined.close()

    # Clean up batch files
    for f in batch_files:
        if f.exists():
            f.unlink()

    return traj_file


def combine_saved_trajectories(
    temp_dir: Path,
    out_dir: Path,
    in_stem: str,
    time_idx: int,
    num_members: int,
    num_sr_samples: int,
    has_number_dim: bool = True,
):
    """
    Load and combine all saved trajectories into final dataset.
    Cleans up temporary files after saving.
    """
    members_list = []

    for ni in range(num_members):
        sr_samples_list = []
        for si in range(num_sr_samples):
            temp_file = temp_dir / f"member{ni:02d}_sr{si:02d}.nc"
            xr_traj = xr.open_dataset(temp_file)
            sr_samples_list.append(xr_traj)

        # Concatenate all SR samples along new dimension (only if > 1)
        if num_sr_samples == 1:
            xr_member = sr_samples_list[0]
        else:
            xr_member = xr.concat(
                sr_samples_list,
                pd.Index(range(num_sr_samples), name="sr_sample"),
            )
        members_list.append(xr_member)

        # Close datasets to free memory
        for ds in sr_samples_list:
            ds.close()

    # Concat members (only if original data had number dimension)
    if has_number_dim and num_members > 1:
        xr_dataset = xr.concat(
            members_list,
            pd.Index(range(num_members), name="number"),
        )
    elif num_members == 1:
        # For deterministic forecasts, don't add a number dimension
        xr_dataset = members_list[0]
    else:
        # Fallback: if we have multiple members but no number dim, still concat
        xr_dataset = xr.concat(
            members_list,
            pd.Index(range(num_members), name="number"),
        )
    print(xr_dataset)

    # Save final output
    out_file = out_dir / f"{in_stem}_time{time_idx:03d}_sr.nc"
    xr_dataset.to_netcdf(out_file)
    print(f"Saved {out_file}")

    xr_dataset.close()

    # Clean up temp files
    import shutil

    shutil.rmtree(temp_dir)
    print(f"Cleaned up temp directory: {temp_dir}")


@torch.no_grad()
def main(
    task_id: int = 0,
    inputs_glob: str = DEFAULT_INPUTS_GLOB,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_sr_samples: int = NUM_SR_SAMPLES,
):
    files = sorted(glob.glob(inputs_glob))
    if not files:
        raise FileNotFoundError(f"No files match: {inputs_glob}")

    # Build timestamp index: (file_id, time_id, time_value)
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
    print(f"Generating {num_sr_samples} SR samples per input sample")

    # Load models
    era5_like = Era5Forecast(
        path=files[file_id], domain="all", norm_scheme="era5", load_prev=False
    )
    sr_model = load_module(DEFAULT_SR_MODULE, return_config=False)
    sr_model.eval().to(DEVICE)

    # Load and select single time slice
    ds = xr.open_dataset(in_file).isel(time=time_id)
    init_time = ds["time"].values
    init_time_sec = np.int64(np.datetime64(init_time, "s").astype(int))

    # Check if this is an ensemble or deterministic forecast
    has_number_dim = "number" in ds.dims
    if has_number_dim:
        numbers = ds["number"].values
    else:
        numbers = np.array([0])  # Single deterministic forecast

    leads = ds["prediction_timedelta"].values

    # Create temp directory for intermediate files
    temp_dir = out_dir / f"temp_task{task_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process by (member, sr_sample) with batched lead times to balance memory and efficiency
    batch_size = 10
    total_iterations = len(numbers) * num_sr_samples * len(leads)
    with tqdm(total=total_iterations, desc=f"SR {in_stem}_t{time_id}") as pbar:
        for ni in range(len(numbers)):
            for si in range(num_sr_samples):
                # Process lead times in batches
                num_batches = (
                    len(leads) + batch_size - 1
                ) // batch_size  # Ceiling division

                for batch_idx in range(num_batches):
                    # Determine lead time range for this batch
                    start_li = batch_idx * batch_size
                    end_li = min(start_li + batch_size, len(leads))

                    batch_tdicts = []

                    # Process batch of lead times
                    for li in range(start_li, end_li):
                        # Select the data slice, handling both ensemble and deterministic cases
                        if has_number_dim:
                            xr_one = ds.isel(number=ni, prediction_timedelta=li)
                        else:
                            xr_one = ds.isel(prediction_timedelta=li)

                        vtime = init_time + leads[li]

                        # Process single sample
                        td = era5_like.convert_to_tensordict(xr_one)
                        td = era5_like.normalize(td)
                        td = {k: v.unsqueeze(0).to(DEVICE) for k, v in td.items()}

                        vt_sec = torch.tensor(
                            [np.int64(np.datetime64(vtime, "s").astype(int))],
                            dtype=torch.int64,
                        )

                        inputs = prepare_sr_inputs_for_model(
                            TensorDict(td, batch_size=[1]), vt_sec
                        )

                        # Generate SR sample for this lead time
                        seed = int(task_id * 1000 + ni * 100 + li * 10 + si)
                        sr_out = sr_model.sample(inputs, seed=seed)

                        # Store in batch (CPU memory)
                        batch_tdicts.append(sr_out.detach().cpu())

                        # Free GPU memory immediately
                        del sr_out, inputs, td
                        torch.cuda.empty_cache()

                        pbar.update(1)

                    # Save this batch immediately to disk
                    save_sr_batch_leadtimes(
                        batch_tdicts,
                        era5_like,
                        init_time_sec,
                        temp_dir,
                        ni,
                        si,
                        batch_idx,
                    )

                    # Free CPU memory after saving batch
                    del batch_tdicts
                    torch.cuda.empty_cache()

                # After all batches processed, combine them into a trajectory
                combine_batches_into_trajectory(temp_dir, ni, si, num_batches)

    # Combine all saved trajectories into final output
    combine_saved_trajectories(
        temp_dir,
        out_dir,
        in_stem,
        time_id,
        len(numbers),
        num_sr_samples,
        has_number_dim,
    )

    ds.close()
    print(f"[Task {task_id}] Done: {in_file}, time {time_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--inputs-glob", type=str, default=DEFAULT_INPUTS_GLOB)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-sr-samples", type=int, default=NUM_SR_SAMPLES)
    args = parser.parse_args()
    main(
        task_id=args.task_id,
        inputs_glob=args.inputs_glob,
        output_dir=args.output_dir,
        num_sr_samples=args.num_sr_samples,
    )
