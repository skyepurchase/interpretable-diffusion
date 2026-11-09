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
The script takes a seed value with `-s`.

### DDIM

To train the DDIM run the following command
```
uv run src/train.py -c configs/mnist_ddim/26JAN16.yaml
```
The script takes a seed value with `-s`.

## PAID

To train PAID and save a checkpoint run the following command
```
uv run src/main.py -c configs/pendulum_paid/mlp.yaml
```
`main.py` saves the trained model under the config's `checkpoint_dir` (a timestamped folder and `latest`); `train.py` does not persist a model.

To then visualise generated samples from the trained model run
```
uv run src/visualise.py -c configs/pendulum_paid/mlp.yaml -o reports/vis
```
Visualisation supports PAID models only. 2D datasets (e.g. `configs/diffusion_paid`) render a grid of PAID output, target, and absolute error with shared colour scales and colorbars; 1D datasets (e.g. `configs/pendulum_paid`) render overlaid output/target lineplots. Both mark the condition/generated boundary with a vline in the key.

Both scripts take a seed value with `-s`. `visualise.py` has more options, notably `--num` for the number of outputs to visualise, which can be found with
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
