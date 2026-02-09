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
