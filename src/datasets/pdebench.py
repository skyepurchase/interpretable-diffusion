import random

import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.config.datasets import PDEBenchConfig

SYSTEM_TO_DIR: dict[str, str] = {
    "diff_sorp": "data/pdebench/1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5"
}


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

    with h5py.File(SYSTEM_TO_DIR[cfg.system], "r") as file:
        index = random.choice(list(file.keys()))

        entry = file[index]

        assert isinstance(entry, h5py.Group), "Expected entry to be group"
        assert "data" in entry, "Expected entry to contain 'data' object"

        data = entry["data"]

        assert isinstance(data, h5py.Dataset), "Expected 'data' object to be dataset"

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.tight_layout()

        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(data[:])

        plt.savefig("test.png", format="png")

    raise NotImplementedError
