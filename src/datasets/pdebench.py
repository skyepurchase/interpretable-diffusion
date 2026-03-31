import os

import h5py
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from datasets.utils import np_collate
from utils.config.datasets import PDEBenchConfig

SYSTEM_TO_DIR: dict[str, str] = {
    "diff_sorp": "data/pdebench/1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5"
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

            indices = list(file.keys())
            if test:
                indices = indices[int(0.9 * len(indices)) :]
            else:
                indices = indices[: int(0.9 * len(indices))]

            for index in indices:
                entry = file[index]

                assert isinstance(entry, h5py.Group), "Expected entry to be group"
                assert "data" in entry, "Expected entry to contain 'data' object"

                entry_data = entry["data"]

                assert isinstance(entry_data, h5py.Dataset), (
                    "Expected 'data' object to be dataset"
                )

                data.append(torch.tensor(entry_data[:]).flatten())

        self.data = torch.vstack(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


def get_pdebench(cfg: PDEBenchConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.system not in SYSTEM_TO_DIR:
        print(f"""
1. Clone `pdebench` from https://github.com/pdebench/PDEBench
2. `cd` into the directory and `cd pdebench/data_download`
3. run `uv sync` (you may need to remove the optional dependencies in `pyproject.toml`)
4. run
```
python download_direct.py --root_folder <path-to-this-repo>/data/pdebench --pde_name {cfg.system}
```
        """)

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
