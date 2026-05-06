import os

import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url

from datasets.utils import np_collate
from utils.config.datasets import PDEBenchConfig

SYSTEM_TO_DIR: dict[str, str] = {
    "diff_sorp": "diffusion-sorption/data.h5",
    "burgers": "burgers/data.h5",
}

SYSTEM_TO_LINK: dict[str, str] = {
    "diff_sorp": "https://darus.uni-stuttgart.de/api/access/datafile/133020",
    "burgers": "https://darus.uni-stuttgart.de/api/access/datafile/268190",
}


class PDEBenchDataset(Dataset):
    """
    A wrapper of the torch Dataset for PDEBench

    Parameters
    ----------
    data_file : str
        The path to the HDF5 PDEBench file
    test : bool = False
        Whether this is the test dataset or not
    """

    def __init__(self, data_file: str, test: bool = False) -> None:
        if not os.path.isfile(data_file):
            print(f"File {data_file} was not found")
            quit(1)

        self.data = torch.empty(0)
        self.t = torch.empty(0)
        self.num_grid_points = -1
        self.ds = -float("inf")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.t.clone()


class DiffusionSorptionDataset(PDEBenchDataset):
    def __init__(self, data_file: str, test: bool = False) -> None:
        super().__init__(data_file, test)

        with h5py.File(data_file, "r") as file:
            data = []
            t = None
            ds = None
            num_grid_points = None
            sample = None

            indices = list(file.keys())
            if test:
                indices = indices[int(0.9 * len(indices)) :]
            else:
                indices = indices[: int(0.9 * len(indices))]

            for index in indices:
                entry = file[index]

                assert isinstance(entry, h5py.Group), "Expected entry to be group"
                assert "data" in entry, "Expected entry to contain 'data' object"
                assert "grid" in entry, "Expected entry to contain 'grid' object"

                entry_data = entry["data"]
                entry_grid = entry["grid"]

                assert isinstance(entry_grid, h5py.Group), "Expected grid to be group"
                assert "t" in entry_grid, "Expected grid to contain 't' object"
                assert "x" in entry_grid, "Expected grid to contain 'x' object"

                entry_t = entry_grid["t"]
                entry_x = entry_grid["x"]

                assert isinstance(entry_t, h5py.Dataset), "Expected 't' to be dataset"
                if t is None:
                    t = entry_t[:]
                else:
                    assert (t == entry_t[:]).all(), (
                        f"Expected all indices to have the same time discretisation. Index {index} does not"
                    )

                assert isinstance(entry_data, h5py.Dataset), (
                    "Expected 'data' object to be dataset"
                )
                assert isinstance(entry_x, h5py.Dataset), "Expected 't' to be dataset"

                if sample is None or num_grid_points is None:
                    # TODO: dynamically select stride
                    sample = np.arange(0, entry_data.shape[1], 8)
                    num_grid_points = sample.shape[0]

                sample_x = entry_x[sample]

                if ds is None:
                    ds = sample_x[1] - sample_x[0]

                assert (ds == (sample_x[1:] - sample_x[:-1])).all(), (
                    f"Expected all indices to have the same space discretisation. Index {index} does not. {ds} != {(sample_x[1:] - sample_x[:-1]).mean()}."
                )

                data.append(torch.tensor(entry_data[:, sample]).unsqueeze(0))

        assert t is not None, "No time discretisation found."
        assert ds is not None, "No space discretisation found."
        assert num_grid_points is not None, "Grid size not found."

        self.num_grid_points = num_grid_points
        self.ds = ds
        self.t = torch.tensor(t)
        self.data = torch.vstack(data)

        print(self.data.shape)


class BurgersDataset(PDEBenchDataset):
    def __init__(self, data_file: str, test: bool = False) -> None:
        super().__init__(data_file, test)

        with h5py.File(data_file, "r") as file:
            t_entry = file["t-coordinate"]
            assert isinstance(t_entry, h5py.Dataset), (
                f"Expected 't' to be dataset, received {type(t_entry)}"
            )
            self.t = torch.tensor(t_entry[:])

            x_entry = file["x-coordinate"]
            assert isinstance(x_entry, h5py.Dataset), (
                f"Expected 'x' to be dataset, received {type(x_entry)}"
            )

            # TODO: dynamically select stride
            sample = np.arange(0, x_entry.shape[0], 8)
            self.num_grid_points = sample.shape[0]
            assert self.num_grid_points > 0, (
                f"Number of grid points must be positive, {self.num_grid_points} <= 0"
            )

            sample_x = x_entry[sample]
            self.ds = sample_x[1] - sample_x[0]
            assert self.ds > 0, f"Spatial step must be positive, {self.ds} <= 0"

            data_entry = file["tensor"]
            assert isinstance(data_entry, h5py.Dataset), (
                f"Expected 'tensor' to be dataset, received {type(x_entry)}"
            )

            if test:
                self.data = torch.tensor(
                    data_entry[int(0.9 * data_entry.shape[0]) :, :, sample]
                ).unsqueeze(-1)
            else:
                self.data = torch.tensor(
                    data_entry[: int(0.9 * data_entry.shape[0]), :, sample]
                ).unsqueeze(-1)


SYSTEM_TO_DATASET = {
    "diff_sorp": DiffusionSorptionDataset,
    "burgers": BurgersDataset,
}


def get_pdebench(cfg: PDEBenchConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.system not in SYSTEM_TO_DIR:
        print(f"{cfg.system} is not supported.")
        quit(1)
    filename = os.path.join(cfg.dir, SYSTEM_TO_DIR[cfg.system])

    if not os.path.isfile(filename):
        print("Data file not found. Downloading data ...")
        download_url(
            SYSTEM_TO_LINK[cfg.system],
            os.path.dirname(filename),
            os.path.basename(filename),
        )

    dataset = SYSTEM_TO_DATASET[cfg.system](filename)
    test = SYSTEM_TO_DATASET[cfg.system](filename, test=True)
    train, val = data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, collate_fn=np_collate, shuffle=True
    )
    val_loader = DataLoader(
        val, batch_size=cfg.batch_size, collate_fn=np_collate, shuffle=False
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, collate_fn=np_collate, shuffle=False
    )

    return train_loader, val_loader, test_loader
