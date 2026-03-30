from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from tqdm import tqdm

from archesweathersr.utils.logging_utils import setup_logger


def get_filter_mask(domain: str, time: pd.DatetimeIndex) -> np.ndarray:
    filters = {
        "all": lambda t: np.ones(len(t), dtype=bool),
        "last_train": lambda t: t.year == 2018,
        "last_train_z0012": lambda t: (t.year == 2018) & (t.hour.isin([0, 12])),
        "train": lambda t: ~t.year.isin([2019, 2020, 2021]),
        "train_z0012": lambda t: ~t.year.isin([2019, 2020, 2021])
        & t.hour.isin([0, 12]),
        "val": lambda t: t.year.isin([2018, 2019, 2020]),
        "val_z0012": lambda t: t.year.isin([2018, 2019, 2020]) & t.hour.isin([0, 12]),
        "test": lambda t: t.year.isin([2019, 2020, 2021]),
        "test_z0012": lambda t: t.year.isin([2019, 2020, 2021]) & t.hour.isin([0, 12]),
        "empty": lambda t: np.zeros(len(t), dtype=bool),
    }

    try:
        return filters[domain](time)
    except KeyError:
        raise ValueError(f"Unknown domain: {domain}")


class HDF5Dataset(torch.utils.data.Dataset):
    """Dataset to read HDF5 files.

    Supports two modes depending on how ``variables`` is passed:

    1. Pre-stacked: variables are already stacked in the file.
       Pass a flat list of HDF5 dataset keys::

           variables=["surface", "level"]

       Each key is loaded as-is (shape ``(time, n_vars, ...)``), and returned
       directly as the corresponding TensorDict entry.

    2. Per-variable: one HDF5 dataset per variable.
       Pass a dict mapping TensorDict keys to lists of variable names::

           variables={"surface": ["2m_temperature", "10m_u"], "level": ["geopotential", "temperature"]}

       Variables in each group are stacked along a new axis 0 → shape ``(n_vars, ...)``,
       matching the netcdf convention.

    The dataset can load a single HDF5 file or a list of files from a directory.
    Each file is expected to contain a ``/time`` dataset of UNIX timestamps (seconds).
    """

    def __init__(
        self,
        path: str,
        variables: Union[List[str], Dict[str, List[str]]],
        warning_on_nan: Optional[bool] = True,
        limit_examples: Optional[int] = None,
        return_timestamp: bool = False,
    ):
        """
        Args:
            path: Single HDF5 file or directory of ``*.h5`` files.
            variables: Either a flat ``list[str]`` (pre-stacked mode) or a
                ``dict[str, list[str]]`` (per-variable mode).
            warning_on_nan: Log a warning when NaN values are found.
            limit_examples: Cap the number of examples loaded.
            return_timestamp: If ``True``, ``__getitem__`` returns ``(tdict, timestamp)``.
        """
        self.console_logger = setup_logger(__name__, "INFO")
        self.path = path
        self.variables = variables
        self.warning_on_nan = warning_on_nan
        self.return_timestamp = return_timestamp

        self._discover_files(path)
        self._build_index(limit_examples)

        mode = "pre-stacked" if isinstance(variables, list) else "per-variable"
        self.console_logger.info(f"HDF5 mode: {mode}.")

        self.cached_file = None
        self.cached_file_idx = None

    def _discover_files(self, path: str):
        """Identifies and lists files in the given path."""
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if p.is_file():
            self.console_logger.info(f"Single file detected: {path}.")
            self.files = [path]
        else:
            self.console_logger.info(
                f"Directory detected. Searching for files in: {path}"
            )
            self.files = sorted(list(p.glob("*.h5")))
            if not self.files:
                raise FileNotFoundError(f"No files found in directory: {path}")

        # Store files as bytes for better memory handling
        self.files = np.array(self.files, dtype=np.bytes_)

    def _decode_time(self, time_ds) -> np.ndarray:
        """Decode an HDF5 time dataset to int64 UNIX seconds since epoch.

        Respects CF-convention ``units`` / ``calendar`` attributes when present
        (e.g. ``"hours since 1900-01-01"``). Falls back to treating raw values
        as UNIX seconds when no ``units`` attribute is found.

        Returns
            np.ndarray of int64 (seconds since 1970-01-01)
        """
        raw = time_ds[:]
        units = time_ds.attrs.get("units", None)
        if units is not None:
            if isinstance(units, bytes):
                units = units.decode()
            calendar = time_ds.attrs.get("calendar", "standard")
            if isinstance(calendar, bytes):
                calendar = calendar.decode()
            import cftime

            dates = cftime.num2date(raw, units=units, calendar=calendar)
            seconds = np.array(
                [
                    int(np.datetime64(d.isoformat(), "s").astype(np.int64))
                    for d in dates
                ],
                dtype=np.int64,
            )
        else:
            # Assume raw values are already UNIX seconds
            seconds = raw.astype(np.int64)
        return seconds

    def _build_index(self, limit_examples: Optional[int] = None):
        """Builds an index mapping for timestamps."""
        timestamps = []

        for file_idx, f_path_bytes in tqdm(
            enumerate(self.files), desc="Indexing files"
        ):
            f_path = f_path_bytes.decode("UTF-8")
            with h5py.File(f_path, mode="r") as f:
                time_s = self._decode_time(f["time"])
                filestamps = [
                    (file_idx, time_idx, time) for time_idx, time in enumerate(time_s)
                ]
                timestamps.extend(filestamps)
            if limit_examples and len(timestamps) > limit_examples:
                self.console_logger.info(
                    f"Limiting number of examples loaded to {limit_examples}."
                )
                timestamps = timestamps[:limit_examples]
                break

        # Store timestamps as int64 numpy array (UNIX seconds since epoch)
        self.timestamps = np.array(
            sorted(timestamps, key=lambda x: x[-1]), dtype=np.int64
        )

    def filter_timestamps(
        self,
        domain: Optional[str] = None,
        timerange: Optional[tuple[np.datetime64, np.datetime64]] = None,
    ):
        """Filters the dataset timestamps based on domain or a time range."""
        self.console_logger.info(
            f"Filtering timestamps from {len(self.timestamps)} examples."
        )

        if domain is not None:
            mask = np.where(
                get_filter_mask(
                    domain, pd.to_datetime(self.timestamps[:, -1], unit="s")
                )
            )[0]
            self.timestamps = self.timestamps[mask]

        if timerange is not None:
            low, high = timerange
            assert isinstance(low, np.datetime64) and isinstance(high, np.datetime64), (
                "'timerange' must be a tuple of two 'np.datetime64' objects."
            )

            mask = (self.timestamps[:, -1].astype("datetime64[s]") >= low) & (
                self.timestamps[:, -1].astype("datetime64[s]") <= high
            )
            self.timestamps = self.timestamps[mask]

        self.console_logger.info(
            f"timestamps filtered to {len(self.timestamps)} examples."
        )

    def postprocess_tdict(self, tdict: TensorDict) -> TensorDict:
        """Hook for subclasses to apply post-processing after loading (e.g. lat flip, lon roll)."""
        return tdict

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, i, return_timestamp=False):  # type: ignore[override]
        file_idx, time_idx, timestamp = self.timestamps[i]

        if self.cached_file_idx != file_idx:
            if self.cached_file is not None:
                self.cached_file.close()
            self.cached_file = h5py.File(
                self.files[file_idx].decode("UTF-8"), mode="r", libver="latest"
            )
            self.cached_file_idx = file_idx

        assert self.cached_file is not None
        f = self.cached_file

        if isinstance(self.variables, list):
            # Pre-stacked: each key is already a stacked dataset, load directly.
            tdict = TensorDict(
                {var: torch.from_numpy(f[var][time_idx]) for var in self.variables}
            )
        else:
            # Per-variable: stack individual variable arrays along a new axis 0.
            tdict = TensorDict(
                {
                    key: torch.from_numpy(
                        np.stack([f[var][time_idx] for var in var_list], axis=0)
                    )
                    for key, var_list in self.variables.items()
                }
            )

        tdict = self.postprocess_tdict(tdict)

        if self.warning_on_nan:
            if any([x.isnan().any().item() for x in tdict.values()]):
                self.console_logger.warning(
                    f"NaN values detected for {np.int64(timestamp).astype('datetime64[s]')} ({self.files[file_idx]})"
                )

        if return_timestamp or self.return_timestamp:
            timestamp = torch.tensor(timestamp, dtype=torch.int32)  # UNIX seconds
            return tdict, timestamp

        return tdict
