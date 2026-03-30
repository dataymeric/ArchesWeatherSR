import argparse
import os
from pathlib import Path

import hdf5plugin
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from archesweathersr.utils.logging_utils import setup_logger

console_logger = setup_logger("INFO")


GRID_LOOKUP = {
    "1.5": [
        "240x121",
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    ],
    "0.25": [
        "1440x721",
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    ],
}


def get_grid_from_res(res_str):
    if res_str not in GRID_LOOKUP:
        raise ValueError(
            f"Unsupported resolution '{res_str}', choose from: {list(GRID_LOOKUP.keys())}"
        )
    return GRID_LOOKUP[res_str]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--years",
    nargs="+",  # Accepts 1 or more arguments as a list.
    type=int,
    default=list(range(1979, 2023)),
    help="Year(s) to download. By default downloads all 1979-2022.",
)
parser.add_argument("--res", default="1.5", type=str, help="resolution to download")
parser.add_argument("--folder", default=None, help="where to store outputs")
parser.add_argument(
    "--compress",
    default=False,
    action="store_true",
    help="whether to compress data",
)
args = parser.parse_args()

grid = get_grid_from_res(args.res)

base_path = "gs://weatherbench2/datasets"
obs_path = f"{base_path}/era5/{grid[1]}"

if args.folder is None:
    args.folder = Path(__file__).parent.joinpath(f"data/era5_{grid}/")
Path(args.folder).mkdir(parents=True, exist_ok=True)


if args.res == "1.5":
    MIN_FILE_SIZE_BYTES = 3_475_000_000  # inferior limit
elif args.res == "0.25":
    MIN_FILE_SIZE_BYTES = 347_500_000_000
else:
    MIN_FILE_SIZE_BYTES = 1_000_000  # a very low fallback

keep_list = [
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

if args.compress:
    blosc_lz4 = hdf5plugin.Blosc2(cname="lz4", clevel=1)

with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
    refresh_per_second=1,
) as progress:
    overall_task = progress.add_task(
        f"[cyan]Downloading ERA5 @ {args.res}°...", total=len(args.years) * 4
    )
    obs_xarr = xr.open_zarr(
        obs_path, storage_options={"session_kwargs": {"trust_env": True}}
    )
    for year in args.years:
        for hour in (0, 6, 12, 18):
            fname = Path(args.folder) / f"era5_{grid[0]}_{year}_{hour:02d}h.h5"
            if Path(fname).exists():
                if os.stat(fname).st_size < MIN_FILE_SIZE_BYTES:
                    console_logger.warning(
                        f"{fname} exists but appears incomplete, redownloading."
                    )
                    os.remove(fname)  # file is corrupted
                else:
                    console_logger.info(
                        f"{fname} already exists and appears complete, skipping."
                    )
                    progress.update(overall_task, advance=1)
                    continue
            console_logger.info(f"Downloading year {year} hour {hour}...")
            ds = obs_xarr.sel(time=obs_xarr.time.dt.year.isin([year]))
            ds = ds.sel(time=ds.time.dt.hour.isin([hour]))
            ds = ds[[v for v in ds.data_vars if v in keep_list]]
            console_logger.info(ds)

            encoding_spec = None
            if args.compress:
                encoding_spec = {}
                for var in ds.data_vars:
                    if len(ds[var].dims) == 4:
                        encoding_spec[var] = {
                            **blosc_lz4,
                            "chunksizes": (1, 1, 721, 1440),
                            "dtype": np.float32,
                        }
                    elif len(ds[var].dims) == 3:
                        encoding_spec[var] = {
                            **blosc_lz4,
                            "chunksizes": (1, 721, 1440),
                            "dtype": np.float32,
                        }

            with ProgressBar():
                ds.to_netcdf(fname, engine="h5netcdf", encoding=encoding_spec)
            console_logger.info(f"{fname} downloaded.")

            progress.update(overall_task, advance=1)
