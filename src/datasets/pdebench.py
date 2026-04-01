import os

import h5py
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url

from datasets.utils import np_collate
from utils.config.datasets import PDEBenchConfig

SYSTEM_TO_DIR: dict[str, str] = {
    "diff_sorp": "data/pdebench/diffusion-sorption/data.h5"
}

SYSTEM_TO_LINK: dict[str, str] = {
    "diff_sorp": "https://darus.uni-stuttgart.de/api/access/datafile/133020"
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

        with h5py.File(data_file, "r") as file:
            data = []
            t = None

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
                entry_t = entry_grid["t"]
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

                data.append(torch.tensor(entry_data[:]).flatten())

        assert t is not None, "No time discretisation found"
        self.t = torch.tensor(t)
        self.data = torch.vstack(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.t.clone()


def get_pdebench(cfg: PDEBenchConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.system not in SYSTEM_TO_DIR:
        print(f"{cfg.system} is not supported.")
        quit(1)

    if not os.path.isfile(SYSTEM_TO_DIR[cfg.system]):
        print("Data file not found. Downloading data ...")
        download_url(
            SYSTEM_TO_LINK[cfg.system],
            os.path.dirname(SYSTEM_TO_DIR[cfg.system]),
            os.path.basename(SYSTEM_TO_DIR[cfg.system]),
        )

    dataset = PDEBenchDataset(SYSTEM_TO_DIR[cfg.system])
    test = PDEBenchDataset(SYSTEM_TO_DIR[cfg.system])
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
