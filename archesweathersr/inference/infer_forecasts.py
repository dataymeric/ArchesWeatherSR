"""Super-resolve ArchesWeather forecasts files.

Each script invocation handles exactly one task, identified by ``--task-id``.
A task corresponds to a single initialization time (time slice) taken from the
pool of NetCDF files that match ``--inputs-glob``.  For that time slice every
(number, prediction_timedelta) combination is processed:

1. The valid time is computed as ``init_time + prediction_timedelta``.
2. The low-resolution forecast state is converted to a normalised
   ``TensorDict`` via ``Era5Forecast.convert_to_tensordict``.
3. The SR model is called with ``timestamp=valid_time`` (Unix seconds).
4. Super-resolved outputs are written to a NetCDF file named after the
   input file stem and time index.

Lead times are processed in batches to bound peak GPU/CPU memory usage.
Intermediate per-member, per-SR-sample, per-batch files are stored in a
temporary directory and cleaned up once the final combined file is written.

Example:
    Run super-resolution for task 0 with default paths::

        python infer_forecasts.py --task-id 0

    Override input glob and output directory::

        python infer_forecasts.py \\
            --task-id 5 \\
            --inputs-glob "data/forecasts/*.nc" \\
            --output-dir "data/sr_forecasts"
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
    """Build the input dictionary expected by the SR model.

    Packages a normalized low-resolution state together with a randomly
    initialized high-resolution noise field and a timestamp into the nested
    dictionary format consumed by the SR flow-matching model.

    Args:
        batch_tdict: Normalised low-resolution state with keys ``"surface"``
            (shape ``[B, surface_vars, 1, lat, lon]``) and ``"level"``
            (shape ``[B, level_vars, pressure_levels, lat, lon]``).
        timestamps_sec: Unix timestamps in seconds for each batch element,
            shape ``[B]``.

    Returns:
        A nested dictionary with the structure::

            {
                "timestamp": Tensor,          # shape [B], on DEVICE
                "state": {
                    "lowres": TensorDict,     # surface + level, on DEVICE
                    "highres": TensorDict,    # random noise, on DEVICE
                },
            }
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
    """Iterate over all (ensemble member, lead time) combinations in a time slice.

    Args:
        ds_time: A single-time-slice ``xr.Dataset`` with dimensions
            ``number`` and ``prediction_timedelta``.

    Yields:
        A tuple ``(xr_slice, valid_time, init_time)`` where:

        - ``xr_slice`` – single-member/lead ``xr.Dataset`` ready for
          ``Era5Forecast.convert_to_tensordict``.
        - ``valid_time`` – ``np.datetime64[ns]`` equal to
          ``init_time + prediction_timedelta``.
        - ``init_time`` – ``np.datetime64[ns]`` initialisation time of the
          forecast.
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
    """Denormalize and write a batch of SR lead-time outputs to a NetCDF file.

    Stacks a list of per-lead-time ``TensorDict`` objects into a single
    trajectory, converts it to ``xr.Dataset`` via
    ``Era5Forecast.convert_trajectory_to_xarray``, and saves the result to a
    temporary file named by ``(member_idx, sr_sample_idx, batch_idx)``.

    Args:
        sr_tdict_list: List of SR output ``TensorDict`` objects, one per lead
            time, each with shape ``[1, vars, ...]`` on CPU.
        era5_like: ``Era5Forecast`` instance used for denormalization and
            xarray conversion.
        init_time_sec: Forecast initialization time as a Unix timestamp
            (seconds).
        temp_dir: Directory in which the temporary batch file is written.
        member_idx: Zero-based ensemble member index; used in the filename.
        sr_sample_idx: Zero-based SR sample index; used in the filename.
        batch_idx: Zero-based batch index within this member/SR-sample pair;
            used in the filename.

    Returns:
        Path to the written temporary NetCDF file.
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
    """Concatenate batch NetCDF files into a single trajectory file.

    Reads all batch files belonging to a ``(member_idx, sr_sample_idx)`` pair,
    concatenates them along the ``prediction_timedelta`` dimension, and saves
    the result.  Batch files are deleted after concatenation.  If only one
    batch exists the file is simply renamed without loading it into memory.

    Args:
        temp_dir: Directory containing the batch files.
        member_idx: Zero-based ensemble member index.
        sr_sample_idx: Zero-based SR sample index.
        num_batches: Total number of batch files to combine.

    Returns:
        Path to the combined trajectory NetCDF file.
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
    """Assemble per-member trajectory files into the final output NetCDF.

    Reads all per-member, per-SR-sample trajectory files from ``temp_dir``,
    concatenates SR samples along a new ``sr_sample`` dimension (when
    ``num_sr_samples > 1``), concatenates ensemble members along the
    ``number`` dimension (when the input data has that dimension and
    ``num_members > 1``), and writes the combined dataset to ``out_dir``.
    The entire temporary directory is removed on completion.

    Args:
        temp_dir: Directory containing the per-member trajectory files written
            by :func:`combine_batches_into_trajectory`.
        out_dir: Destination directory for the final NetCDF file.
        in_stem: Stem of the input NetCDF filename, used to construct the
            output filename.
        time_idx: Time-slice index within the input file, used in the output
            filename.
        num_members: Number of ensemble members (or 1 for deterministic).
        num_sr_samples: Number of SR samples generated per member.
        has_number_dim: Whether the original input had a ``number`` dimension.
            Defaults to ``True``.
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
    """Entry point for SR forecast inference.

    Indexes all NetCDF files matching ``inputs_glob``, maps ``task_id`` to a
    specific ``(file, time_slice)`` pair, and runs super-resolution over every
    ``(ensemble_member, lead_time)`` combination in that time slice.
    Lead times are processed in batches of 10 to keep peak memory bounded.
    Results are written to ``{output_dir}/{in_stem}_time{time_idx:03d}_sr.nc``.

    Decorated with ``@torch.no_grad()`` — no gradients are computed.

    Args:
        task_id: Zero-based index into the flattened list of
            ``(file, time_slice)`` pairs across all input files.  Used to
            distribute work across parallel jobs.  Defaults to ``0``.
        inputs_glob: Glob pattern for input NetCDF forecast files.
            Defaults to ``DEFAULT_INPUTS_GLOB``.
        output_dir: Directory where SR output files are written.
            Defaults to ``DEFAULT_OUTPUT_DIR``.
        num_sr_samples: Number of independent SR samples to generate per
            ``(member, lead_time)`` pair.  Defaults to ``NUM_SR_SAMPLES``.

    Raises:
        FileNotFoundError: If no files match ``inputs_glob``.
    """
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
