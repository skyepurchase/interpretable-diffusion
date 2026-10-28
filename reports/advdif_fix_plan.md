# Advdif PhysVAE fix plan

Branch: `vibe/reimplementation-errors` (companion to
`reports/advdif_reimplementation_comparison.tex`, which diagnoses issues D1–D8).

Reference: Takeishi & Kalousis, *Physics-Integrated Variational Autoencoders*,
NeurIPS 2021. Advdif hyperparameters from the paper's supplemental (Appendix E.3):
`alpha = 1e-1`, `beta = 1e-2`, `gamma = 1e6`, augmentation samples drawn from
`[0.005, 0.2]`, batch 200, 20000 epochs, best-validation-reconstruction checkpoint.

## Naming conventions (verified against the paper)

The paper's weights are:

- `alpha` — weight of `R_ppc`, the least-action-principle regularizer
  (a.k.a. discrepancy loss over auxiliary-function combinations).
- `beta` — weight of `R_DA,1`, the **unmixing** regularizer
  (encoder's physics-only estimate vs the decoder's physics-only generation).
- `gamma` — weight of `R_DA,2`, the **physics-based data augmentation** regularizer
  (train the physics head on decoder-generated pure-physics data with known
  coefficients).

The reimplementation already uses this mapping
(`train_phys_vae.py: phys_loss = beta * ...`, `data_aug_loss = gamma * ...`,
`disc_loss = alpha * ...`). Caution: the **original repo's** `advdif_train.sh`
names are swapped relative to the paper (`beta=1e6` feeds `--balance-dataug`,
`gamma=1e-2` feeds `--balance-unmix`). When comparing against the original repo,
trust the flag each value is passed to, not the variable name.

Coefficient naming: the diffusion coefficient is `diff_coeff` (paper: `a`) and the
advection coefficient is `adv_coeff` (paper: `b`).

## Phase 0 — Readability (done)

- [x] Rename `a` -> `diff_coeff` and `b` -> `adv_coeff` in the diffusion data
      generator (`src/datasets/pivae.py`) and physics model
      (`src/models/physics_models.py`).
- [x] Rename the generic physics-parameter argument `a_list` -> `phys_params` in
      `physics_models.py` (its old name misleadingly suggested coefficient `a`).
- [x] Fix stale docstrings (the diffusion generator docstring described the
      pendulum dataset).
- [x] Document the `alpha`/`beta`/`gamma` mapping in the model and loss code.
- [x] Erratum in the comparison report: its loss table used the original repo's
      flag naming for `beta`/`gamma`, which is swapped relative to the paper.

## Phase 1 — Correctness of the advdif pipeline

- [ ] **D1: decoder/data time-grid mismatch.** Generator uses
      `t = linspace(0, T=1.0, 50)` so `dt = 1/49 ~ 0.0204`; the paper uses
      `dt = 0.02` with 50 frames (span 0.98). The model config uses
      `timestep: 0.05` (span 2.45) — a 2.45x stretch that biases the estimated
      coefficient. Fix: generate with `t = arange(50) * 0.02`, set
      `timestep: 0.02` (and `integral_level: 2` as in the original), and make
      the decoder integrate to the final **time** (`t_span[-1]`) instead of
      `integral_len` (a step count) — currently each decode also wastes ~20x
      compute integrating to `t = 50`.
- [ ] **D4: augmentation-loss broadcasting bug.** `phys_func(aug_x_mean)` returns
      concatenated `(mu, log_sigma)` of shape `(B, 2)` but is compared to
      `aug_param` of shape `(B, 1)`, regressing `log_sigma` toward the coefficient
      value. Fix: split the output, regress `mu` only; sample augmentation
      parameters from the paper's range `[0.005, 0.2]`.
- [ ] **D5: discrepancy-loss compares wrong tensors.** `full_forward` returns
      means as `(m*B, T, L)` (m-major) but `discrepancy_loss` slices it as
      `(m, B, ...)`, so with `m = 2` it measures the distance between two
      different batch elements instead of two combinations of one element.
      Fix: reshape to `(m, B, T, L)` and take per-sample Hamming-1 pairs.

## Phase 2 — Objective balance

- [ ] **D2: KL balance.** The original weights the KL by the observation variance
      `exp(-13.8) ~ 1e-6` (data noise `sigma = 1e-3`); the reimplementation uses
      `exp(0.01) ~ 1` — a million-fold stronger KL that collapses the physics
      posterior toward the prior mean. Fix: set `x_log_sigma` so
      `exp(x_log_sigma) ~ sigma^2 = 1e-6` (i.e. `x_log_sigma: -13.8`), matching
      the paper's ELBO scale.
- [ ] **D3: augmentation weight.** Config has `gamma: 1`; the paper uses
      `gamma = 1e6`. This term is the only direct supervision of the physics
      coefficient — the paper's ablation shows inference error degrades ~20x
      without it. Fix: `gamma: 1e6` (after D2's rescaling, verify term scales
      against the ELBO as the paper intends).
- [ ] **D8: reconstruction scale.** Recon loss sums over time only and then means
      over batch and space (`train_phys_vae.py:86`), ~12x weaker than the
      original's sum-over-all-dims convention. Fix: sum over time and space,
      mean over batch, and re-check regularizer weights under that scale.

## Phase 3 — Evaluation and training protocol

- [ ] **D6: evaluate the posterior mean** of the physics latent, not a
      reparameterized sample (`PhysVAE.__call__` returns a draw; eval reports its
      MAE). Also report relative error for paper comparability.
- [ ] **Shape/`retain_dim` consistency.** Training requires 3-D `(B, T, L)`
      batches but the configs never set `retain_dim: true`, and the eval path
      `init_y = x[:, :num_grid_points]` assumes 2-D flattened data. Unify both
      paths on one convention and set `retain_dim: true` in the configs.
- [ ] **D7: protocol.** Stop running `train_step` (optimizer update) on the
      validation loader; select and save the best-validation-reconstruction
      checkpoint; raise epochs (paper: 20000; at minimum 5000); add value grad
      clipping 5.0 and weight decay 1e-6 as in the original.

## Phase 4 — Validation

- [ ] Short training run; scatter predicted vs true `diff_coeff`: clustering near
      `0.41 * true` indicates a residual D1 time-scale bias; clustering near the
      prior mean 0.055 indicates residual D2 collapse.
- [ ] Regenerate the dataset after the D1 time-grid fix (delete cached
      `data/pivae/diffusion/*.npy`).
- [ ] Compare coefficient MAE / relative error against the paper's Table 1
      (NN+phys+reg inference error ~0.0095) and record results here.
