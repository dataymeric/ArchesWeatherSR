import gc
import importlib.resources
from typing import Dict, List, Optional, cast

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from .. import stats as geoarches_stats
from .era5 import level_variables, surface_variables
from .hdf5 import HDF5Dataset


class ERA5Dataset(HDF5Dataset):
    """Loads ERA5-like data from a pre-processed HDF5 file.

    Applies standard transformations like coordinate flipping and longitude rolling.
    """

    def __init__(
        self,
        path: str,
        domain: str = "train",
        variables: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            path: Path to a single HDF5 file.
            domain: Specifies the data split (e.g., "train", "val", "test").
            variables: List of variables (HDF5 dataset keys) to load.
        """
        if variables is None:
            variables = dict(surface=surface_variables, level=level_variables)

        super().__init__(
            path,
            variables=variables,
            warning_on_nan=True,
        )

        self.filter_timestamps(domain=domain)

        # Detect lat/lon order from file
        with h5py.File(cast(bytes, self.files[0]).decode("UTF-8"), mode="r") as f:
            if "latitude" in f:
                lats = f["latitude"][:]
                self.flip_lat = bool(
                    lats[0] < lats[-1]
                )  # ascending → need to flip to N→S
            else:
                self.flip_lat = False

            if "longitude" in f:
                lons = f["longitude"][:]
                self.roll_lon = bool(
                    lons[0] >= 0
                )  # 0→360 → roll to center on Europe; -180→180 → skip
            else:
                self.roll_lon = False

    def postprocess_tdict(self, tdict):
        # Per-variable mode: unsqueeze surface to add dummy level dim → (n_vars, 1, lat, lon)
        if not isinstance(self.variables, list) and "surface" in tdict:
            tdict["surface"] = tdict["surface"].unsqueeze(-3)

        # Flip latitudes if ascending (so output is always N→S)
        if self.flip_lat:
            tdict = tdict.apply(lambda x: x.flip(-2))

        # Roll longitude by half to center on Europe, only if lons were 0→360
        if self.roll_lon:
            halflon = list(tdict.values())[0].shape[-1] // 2
            tdict = tdict.apply(lambda x: x.roll(halflon, -1))

        return tdict


class ERA5Forecast(ERA5Dataset):
    """Loads ERA5 data for forecasting from an HDF5 file.

    Handles loading of previous/future states and normalization.
    """

    def __init__(
        self,
        path: str,
        domain: str = "train",
        variables: Optional[Dict[str, List[str]]] = None,
        norm_scheme: str | None = "pangu",
        timedelta_hours: int | None = None,
        lead_time_hours: int = 24,
        multistep: int = 1,
        load_prev: bool = True,
    ):
        super().__init__(path, domain=domain, variables=variables)
        self.domain = domain
        self.norm_scheme = norm_scheme
        self.lead_time_hours = lead_time_hours
        self.multistep = multistep
        self.load_prev = load_prev

        # Set specific time bounds for validation/test sets
        if domain in ("val", "val_z0012", "test", "test_z0012"):
            year = 2019 if domain.startswith("val") else 2020
            start_time = np.datetime64(f"{year}-01-01T00:00:00")
            if self.load_prev:
                start_time -= np.timedelta64(self.lead_time_hours, "h")
            end_time = np.datetime64(f"{year + 1}-01-01T00:00:00")
            end_time += np.timedelta64(self.multistep * self.lead_time_hours, "h")

            self.filter_timestamps(timerange=(start_time, end_time))

        if timedelta_hours is None:
            self.timedelta = 6 if "z0012" not in domain else 12

        # include vertical component by default
        geoarches_stats_path = importlib.resources.files(geoarches_stats)

        # Setup normalization
        if self.norm_scheme == "pangu":
            norm_file_path = geoarches_stats_path / "pangu_norm_stats2_with_w.pt"
            pangu_stats = torch.load(str(norm_file_path), weights_only=True)

            self.data_mean = TensorDict(
                surface=pangu_stats["surface_mean"],
                level=pangu_stats["level_mean"],
            )
            self.data_std = TensorDict(
                surface=pangu_stats["surface_std"],
                level=pangu_stats["level_std"],
            )
        elif self.norm_scheme == "era5":
            norm_file_path = (
                geoarches_stats_path / "era5_1440x721_norm_stats_1979-2018.pt"
            )
            era5_stats = torch.load(str(norm_file_path), weights_only=True)

            self.data_mean = TensorDict(
                dict(
                    surface=era5_stats["surface"]["mean"],
                    level=era5_stats["level"]["mean"],
                )
            )
            self.data_std = TensorDict(
                dict(
                    surface=era5_stats["surface"]["std"],
                    level=era5_stats["level"]["std"],
                ),
            )

    def __len__(self):
        offset = self.multistep + int(self.load_prev)
        return super().__len__() - offset * self.lead_time_hours // self.timedelta

    def __getitem__(self, i: int, normalize: bool = True) -> Dict:
        i += int(self.load_prev) * self.lead_time_hours // self.timedelta
        out = {}

        out["timestamp"] = torch.tensor(self.timestamps[i][-1]).int()  # UNIX seconds
        out["state"] = super().__getitem__(i)
        out["lead_time_hours"] = torch.tensor(
            self.lead_time_hours * self.multistep
        ).int()

        T_step = self.lead_time_hours // self.timedelta
        if self.multistep > 0:
            out["next_state"] = super().__getitem__(i + T_step)
        if self.multistep > 1:
            future_states = [
                super().__getitem__(i + k * T_step)
                for k in range(1, self.multistep + 1)
            ]
            out["future_states"] = torch.stack(future_states, dim=0)
        if self.load_prev:
            out["prev_state"] = super().__getitem__(i - T_step)

        if normalize and self.norm_scheme:
            out = self.normalize(out)

        return out

    def normalize(self, batch):
        if self.norm_scheme is None:
            return batch

        device = list(batch.values())[0].device

        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if "surface" in batch:
            # we can normalize directly
            return (batch - means) / stds
        out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}

        return out

    def denormalize(self, batch):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if "surface" in batch:
            # we can denormalize directly
            return batch * stds + means

        out = {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}
        return out


class ERA5Downscaling(torch.utils.data.Dataset):
    """Dataset for downscaling low-resolution ERA5 data to high-resolution data."""

    def __init__(self, lowres_path, highres_path, **kwargs):
        """
        Args:
            lowres_path: Path to the low-resolution dataset.
            highres_path: Path to the high-resolution dataset.
        """
        super().__init__()
        self.lowres_dataset = ERA5Forecast(path=lowres_path, **kwargs)
        self.highres_dataset = ERA5Forecast(path=highres_path, **kwargs)
        self.console_logger = self.lowres_dataset.console_logger

        assert (
            self.lowres_dataset.lead_time_hours == self.highres_dataset.lead_time_hours
        )
        self.lead_time_hours = self.lowres_dataset.lead_time_hours
        self.timedelta = self.lowres_dataset.timedelta
        self.domain = self.lowres_dataset.domain

        self.console_logger.info("aligning timestamps")
        self._align_timestamps()

    def _align_timestamps(self):
        lowres_times = self.lowres_dataset.timestamps[:, -1]
        highres_times = self.highres_dataset.timestamps[:, -1]

        _, highres_idx, _ = np.intersect1d(
            highres_times, lowres_times, return_indices=True
        )

        self.highres_dataset.timestamps = self.highres_dataset.timestamps[highres_idx]

        # sanity check
        assert np.all(np.isin(self.highres_dataset.timestamps[:, -1], lowres_times)), (
            "Datasets are not aligned!"
        )
        self.console_logger.info("timestamps aligned successfully.")
        self.timestamps = self.highres_dataset.timestamps

    def __len__(self):
        return len(self.lowres_dataset)  # because we cannot access first element

    def __getitem__(self, i, normalize=True):
        lowres_data_dict = self.lowres_dataset.__getitem__(i, normalize=normalize)
        highres_data_dict = self.highres_dataset.__getitem__(i, normalize=normalize)

        assert lowres_data_dict["timestamp"] == highres_data_dict["timestamp"], (
            "Timestamps do not match between low and high resolution datasets.\n"
            f"Low-res timestamp: {lowres_data_dict['timestamp']},\n"
            f"High-res timestamp: {highres_data_dict['timestamp']}"
        )

        combined_data = TensorDict()
        for key in lowres_data_dict.keys():
            if key in ["timestamp", "lead_time_hours"]:
                combined_data[key] = lowres_data_dict[key]
            else:
                combined_data[key] = {
                    "lowres": lowres_data_dict[key],
                    "highres": highres_data_dict[key],
                }

        gc.collect()
        return combined_data

    def denormalize(self, batch):
        out = self.highres_dataset.denormalize(batch)
        return out
