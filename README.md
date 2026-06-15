# Interpretable Diffusion

Exploration of using grounded VAEs during the diffusion process to create interpretable/steerable diffusion models.

This project uses `uv` as a dependency manager; if you do not have `uv` already installed follow [the installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## MNIST

Initial exploration was carried out on the MNIST dataset, both a variational autoencoder and denoising diffusion implicit model were created.

### VAE

To train the VAE run the following command
```
uv run src/train.py -c configs/mnist_vae/08JAN26.yaml
```
To then visualise the VAE's reconstruction of unseen digits run
```
uv run src/visualise.py -c configs/mnist_vae/08JAN26.yaml
```

Both scripts take a seed value with `-s`. `visualise.py` has more option which can be found with
```
uv run src/visualise.py --help
```

### DDIM

To train the DDIM run the following command
```
uv run src/train.py -c configs/mnist_ddim/26JAN16.yaml
```
To then visualise generated samples from the trained DDIM run
```
uv run src/visualise.py -c configs/mnist_ddim/26JAN16.yaml
```

Both scripts take a seed value with `-s`. `visualise.py` has more option which can be found with
```
uv run src/visualise.py --help
```

## PDE Bench

[pdebench](https://github.com/pdebench/PDEBench/) dependencies clash with those of this project. The datasets are therefore downloaded in a separate project targeting the data directory in this repository. The following steps where followed

1. Clone `pdebench` https://github.com/pdebench/PDEBench
2. `cd` into the directory and `cd pdebench/data_download`
3. run `uv sync` (you may need to remove the optional dependencies in `pyproject.toml`)
4. run a command from https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download such as
```
python download_direct.py --root_folder <path-to-this-repo>/data/pdebench --pde_name diff_sorp
```
