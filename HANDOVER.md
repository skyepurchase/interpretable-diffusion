# Handover Dossier: interpretable-diffusion pipeline refactor

For the next agent picking up this session. Read fully before acting.

## Context

- Repo: `interpretable-diffusion` (github.com/skyepurchase/interpretable-diffusion)
- Working branch: `vibe/experiments`, based on `origin/models/paid2` at commit `480172f` ("URGH I just want this to work"). The branch is a fast-forward of models/paid2 — no unique commits yet.
- Project: interpretable/steerable diffusion models. JAX ecosystem (jax, flax, diffrax, diffuse-jax), Python 3.13, `uv` for deps, trackio for experiment tracking, orbax for checkpoints.
- Recent upstream work ("Simplified PAID", "Stop using diffuse") made PAID MLP-based: `src/models/paid.py` no longer imports `diffuse`; it uses `make_mlp` from `src/models/mlp.py` and `compute_betas` from `src/models/ddim.py`. `diffuse-jax` is still a hard dependency because `src/models/ddim.py` imports `CondUNet2D` and `src/models/__init__.py` imports DDIM unconditionally.
- Reference config for the PAID line: `configs/diffusion_paid/mlp.yaml` (has `condition: 10`).

## Goal

Split the current monolithic `src/train.py` (loads config/model/datasets, trains, evaluates, saves) into:

- `src/main.py` — orchestrator: parses config, owns seeds/keys, trackio init/finish, saves model, calls train then eval
- `src/train.py` — training only: single `train()` function, returns the trained model
- `src/eval.py` — evaluation only (already exists, mostly fine as-is)

## Working agreement with the user (IMPORTANT)

- DO NOT make code changes until the user explicitly says go. The user got angry when the agent implemented ahead of authorisation. Present plans, wait for "go".
- One git commit at the end of each step where code changed. Never use `--no-verify` (pre-commit hooks must pass).
- Each step's full checklist must be completed and the test run executed fully with no errors before moving on.
- On failure: attempt fixes, but stop and ask the user after 5 unsuccessful tries.
- The user wants an iterative approach: verify at each stage before proceeding.
- The user wants to be consulted on testing strategy ("we'll work on that together, don't provide suggestions as that will influence my thoughts").

## Plan of attack (agreed, action order)

**Step 0 — Environment cleanup (commit at end)**
- Remove `phys-vae` gitlink: it is a bare submodule entry (160000) with NO `.gitmodules` mapping; git warns about it. `git rm phys-vae` (dir is empty).
- Fix torch to CPU-only: on Linux the default PyPI torch pulls ~5G of NVIDIA CUDA packages. The project only uses torch for CPU DataLoader (`torch.utils.data`). Apply this to `pyproject.toml`:
  - Add `"torch>=2.10.0"` to `[project] dependencies` (torch IS directly imported; this is also REQUIRED because `tool.uv.sources` only applies to direct deps — without it torch still resolves from PyPI)
  - Add:
    ```toml
    [tool.uv.sources]
    torch = { index = "pytorch-cpu" }
    torchvision = { index = "pytorch-cpu" }

    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true
    ```
  - Run `uv lock` (torch becomes `2.14.1+cpu` from download.pytorch.org/whl/cpu; nvidia-*/triton drop out) then `uv sync`
- Verify: `uv run python -c "import torch, jax, flax, diffrax, diffuse, trackio; ..."` plus project imports (`sys.path.insert(0, 'src'); import models, datasets, utils`)
- Commit (pyproject.toml + uv.lock + phys-vae removal)

**Step 1 — Baseline (commit at end)**
- Create `configs/diffusion_paid/pipeline_test.yaml`: copy of `mlp.yaml` with `epochs: 5` (everything else unchanged)
- Run the EXISTING `src/train.py` end-to-end with this config to establish a working baseline before refactoring anything
- Commit the config

**Step 2 — Slim train.py to training only (commit at end)**
- Single public function: `train(config, local=False, *, key) -> TrainResult`
  - Loads datasets via `DATASETS`, builds model via `MODELS`, picks stepper/loss labels, runs the training loop
  - Splits its key internally (data, model init, batches) — reproducible for a given key; main.py does not know the details
  - Returns a dataclass `TrainResult`: model, val_loader, final_loss, best_loss, best_epoch, train_time
- Remove: saving, evaluation call, markdown report, `trackio.init`/`trackio.finish`. KEEP per-epoch `trackio.log` (trackio is a global singleton, no object, so logging from train while main owns init works)
- Remove pretrain control flow entirely (this orphans `new_paid`'s pretrain stepper in `models/__init__.py` — leave the stepper, flag it)
- Keep an adjusted `__main__` runtime block (train-only, no save/eval)
- Update `src/sweep.py` here (breaks otherwise): it must call the new `train(config, ...)` per condition (deepcopy config, set `model_config.condition`), and own its own `trackio.init`/`trackio.finish` per condition
- Test: run pipeline_test.yaml through slimmed train.py; then test sweep with same config changing condition k from 10 to 20
- Commit

**Step 3 — main.py orchestration (commit at end)**
- Args: `-c/--config`, `-s/--seed` (default 0), `--local`
- Load config, compute run name (`model_dataset_system_stamp`, `__`→`_`) and stamp (`%m%d-%H%M`)
- Owns the seed: `key = jax.random.key(seed)` → split into `train_key`/`eval_key`
- `trackio.init(project="Interpretable Diffusion", name=run_name, config={model, dataset, **trainer_config})` ONCE
- `result = train(config, local, key=train_key)`
- `save_model(result.model, config["checkpoint_dir"], stamp)`
- `forecast_mae, coeff_mae = evaluate(result.model, result.val_loader, local, key=eval_key)` (from `src/eval.py`, unchanged)
- Log markdown training report (now including eval metrics + train stats) via `trackio.log`, then `trackio.finish()`; print summary when not `--local`
- Test: full pipeline via main.py with pipeline_test.yaml
- Commit

## Environment findings (previous session)

- `/home/vibeuser/projects` was a 7.6G tmpfs mounted **noexec** — shared libraries inside it cannot be loaded (torch's libtorch_global_deps.so failed with "failed to map segment from shared object"). The user is REBUILDING THE CONTAINER with exec permissions on the tmpfs; this dossier exists because of that. If the rebuilt tmpfs is still noexec, the venv must live outside it (e.g. `uv venv /home/vibeuser/.venvs/interpretable-diffusion && ln -s ... .venv`); root fs had ~9G free.
- `uv sync` originally failed on disk space (torch CUDA ~5G+ in a 7.6G tmpfs). CPU torch is ~200-300M.
- The `edit` tool (exact string replacement) is DENIED by the approval policy in this environment — every call, any file. Use `write_file` (full-file rewrite) instead. `python`/`python3` bare commands are denylisted; use `uv run python ...`.
- uv-managed CPython 3.13 was downloaded to `/home/vibeuser/.local/share/uv/python/`.
- `uv sync` works from the repo root; `uv run` resolves `.venv` in the project dir.

## Current state at handover

- Branch `vibe/experiments` at `480172f`, identical to `origin/models/paid2`.
- Working tree was RESET before this dossier commit: the previously staged `phys-vae` removal and the `pyproject.toml`/`uv.lock` CPU-torch changes were reverted so this dossier is the only change. Step 0 must redo them (instructions above).
- `origin/models/paid2` may have moved on — fetch and check before basing anything new.
- Nothing has been pushed to `origin/vibe/experiments` except this commit.

## Misc notes

- `src/eval.py` has a standalone `__main__` (checkpoint-based evaluation) — keep it; it's a separate use case from the train-then-eval flow.
- `src/sweep.py` currently calls `train()` positionally with the OLD signature (run name, config_string args) — it breaks at step 2 and must be updated there.
- The old train.py started a SEPARATE trackio run for pretrain (`pretrain_` prefix); the agreed new behaviour is a single run per training (pretrain flow is being removed anyway).
- README references `src/train.py`/`src/visualise.py` commands; not yet updated for main.py — flag to user, don't change unprompted.
- Pre-commit config exists (`.pre-commit-config.yaml`); commits must pass hooks (no `--no-verify`).
