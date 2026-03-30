from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import xarray as xr
from tensordict import TensorDict
from tqdm import tqdm

from archesweathersr.utils.logging_utils import setup_logger


class XarrayDataset(torch.utils.data.Dataset):
    """
    dataset to read a list of xarray files and iterate through it by timestamp.
    constraint: it should be indexed by at least one dimension named "time".

    Child classes that inherit this class, should implement convert_to_tensordict()
    which converts an xarray dataset into a tensordict (to feed into the model).
    """

    def __init__(
        self,
        path: str,
        variables: Dict[str, List[str]],
        dimension_indexers: Optional[Dict[str, list]] = None,
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = False,
        limit_examples: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Args:
            path: Single filepath or directory holding xarray files.
            variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
                e.g. {surface: [data_var1, datavar2, ...], level: [...]}
                Used in convert_to_tensordict() to read data arrays in the xarray dataset and convert to tensordict.
            dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel(dimension_indexers).
            filename_filter: To filter files within `path` based on filename.
            return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
            warning_on_nan: Whether to log warning if nan data found.
            limit_examples: Return set number of examples in dataset
        """
        self.console_logger = setup_logger(__name__, "DEBUG" if debug else "INFO")
        self.variables = variables
        self.dimension_indexers = dimension_indexers
        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        self.filename_filter = filename_filter
        # Workaround to avoid calling ds.sel() after ds.transponse() to avoid OOM.
        self.already_ran_index_selection = False

        self._discover_files(path)
        self._set_xarray_engine()
        self._build_index(limit_examples)

        self.cached_dataset = None
        self.cached_fileid = None

    def _discover_files(self, path: str):
        """Identifies and lists files in the given path."""
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.endswith(".zarr"):
            self.console_logger.info(f"Zarr store detected: {path}.")
            self.files = [path]
        elif p.is_file():
            self.console_logger.info(f"Single file detected: {path}.")
            self.files = [path]
        else:
            self.console_logger.info(
                f"Directory detected. Searching for files in: {path}"
            )
            files = list(p.glob("*"))
            if not files:
                raise FileNotFoundError(f"No files found in directory: {path}")

            self.files = sorted(
                [str(f) for f in files if self.filename_filter(f.name)],
                key=lambda f: f.replace("6h", "06h").replace("0h", "00h"),
            )
            if not self.files:
                raise FileNotFoundError(
                    "No files remain after applying 'filename_filter'."
                )

        # Store files as bytes for better memory handling
        self.files = np.array(self.files, dtype=np.bytes_)

    def _set_xarray_engine(self):
        """Determines the appropriate xarray engine for opening files."""
        engine_mapping = {
            ".nc": "netcdf4",
            ".nc4": "netcdf4",
            ".h5": "h5netcdf",
            ".hdf5": "h5netcdf",
            ".grib": "cfgrib",
            ".zarr": "zarr",
        }

        file_extension = Path(self.files[0].decode("UTF-8")).suffix
        engine = engine_mapping.get(file_extension)
        if engine is None:
            self.console_logger.warning(
                f"Unrecognized file extension: '{file_extension}'xarray will attempt to use a default engine."
            )
        self.xr_options = {"engine": engine, "cache": True, "decode_timedelta": True}

    def _build_index(self, limit_examples: Optional[int] = None):
        """Builds an index mapping for timestamps."""
        timestamps = []
        for fid, f_path_bytes in tqdm(enumerate(self.files), desc="Indexing files"):
            f_path = f_path_bytes.decode("UTF-8")
            with xr.open_dataset(f_path, **self.xr_options) as ds:
                file_stamps = [(fid, i, t) for (i, t) in enumerate(ds.time.to_numpy())]
                timestamps.extend(file_stamps)
            if limit_examples and len(self.timestamps) > limit_examples:
                self.console_logger.info(
                    f"Limiting number of examples loaded to {limit_examples}."
                )
                timestamps = timestamps[:limit_examples]
                break

        # Store timestamps as numpy array for better memory handling
        self.timestamps = np.array(
            sorted(timestamps, key=lambda x: x[-1]), dtype=np.int64
        )

    def set_timestamp_bounds(self, low, high):
        mask = (self.timestamps[:, -1].astype("datetime64[ns]") >= low) & (
            self.timestamps[:, -1].astype("datetime64[ns]") < high
        )
        self.timestamps = self.timestamps[mask]

        self.console_logger.info(
            f"Timestamp bounds applied: {low} to {high}. Dataset size is now {len(self.timestamps)}."
        )

    def __len__(self):
        return len(self.timestamps)

    def convert_to_tensordict(self, xr_dataset):
        """
        Convert xarray dataset to tensordict.

        By default, it uses a mapping key from self.variables,
            e.g. {surface:[data_var1, data_var2, ...], level:[...]}
        """
        # Optionally select dimensions.
        if self.dimension_indexers and not self.already_ran_index_selection:
            xr_dataset = xr_dataset.sel(self.dimension_indexers)
        self.already_ran_index_selection = False  # Reset for next call.

        np_arrays = {
            key: xr_dataset[list(variables)].to_array().to_numpy()
            for key, variables in self.variables.items()
        }
        tdict = TensorDict(
            {
                key: torch.from_numpy(np_array).float()
                for key, np_array in np_arrays.items()
            }
        )
        return tdict

    def __getitem__(self, i, return_timestamp=False):
        file_id, line_id, timestamp = self.timestamps[i]

        if self.cached_fileid != file_id:
            if self.cached_dataset is not None:
                self.cached_dataset.close()
            self.cached_dataset = xr.open_dataset(
                self.files[file_id].decode("UTF-8"), **self.xr_options
            )
            self.cached_fileid = file_id

        assert self.cached_dataset is not None
        obs = self.cached_dataset.isel(time=line_id)
        tdict = self.convert_to_tensordict(obs)

        if self.warning_on_nan:
            if any([x.isnan().any().item() for x in tdict.values()]):
                ts_readable = datetime.fromtimestamp(
                    timestamp // 10**9, tz=timezone.utc
                ).isoformat()
                self.console_logger.warning(
                    f"NaN values detected for {ts_readable} in file #{file_id}, line #{line_id} ({self.files[file_id]})"
                )

        if return_timestamp or self.return_timestamp:
            timestamp = torch.tensor(
                timestamp // 10**9, dtype=torch.int32
            )  # time in seconds
            return tdict, timestamp
        return tdict
