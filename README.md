# Process Engineering Optimizer

A universal, config-driven workflow for:

- Designing experiments (DOE plans)
- Analyzing measured process data (screening + effect sizing)
- Training predictive models with uncertainty
- Inverse design: propose tool parameters for a desired target property
- Persisting everything (reports, plots, metrics, docs) for later Retrieval-Augmented Generation (RAG)

This repository does **not** integrate an LLM. It focuses on building a strong offline knowledge base (RAG) and an inverse-design engine you can query via CLI.

## What you get

After one command (`peo run`) the project will:

1. Read your measurement CSV (tool parameters + target properties)
2. Infer parameter bounds and categorical levels (or use your config overrides)
3. Generate DOE plan CSVs (Full factorial, Fractional factorial, Plackett-Burman, Latin hypercube, Central composite, Box-Behnken)
4. Run statistical DOE analysis on your measured data (main effects, interactions, p-values, ANOVA-style summaries)
5. Train multiple models per target (Gaussian Process, Random Forest, MLP ensemble)
   - Optionally train DOE-informed variants that use the DOE p-values to select influential factors and add interaction features (enabled via `doe_to_ml`)
   - Optional hyperparameter autotuning via Optuna (if installed)
   - Overfit guard using train/test R2 gap
   - Uncertainty coverage checks when a model provides uncertainty
6. Save plots (parity, residuals, uncertainty coverage) and JSON reports
7. Build a local vector store (TF-IDF by default) so you can retrieve run knowledge later
8. Support inverse queries like: *target sheet resistance = 12 ohm, propose tool parameters*.

All artifacts are stored under `storage/runs/<run_id>/`.

## Quick start (recommended: conda)

From the repo root:

```bash
conda create -n peo python=3.11 -y
conda activate peo
python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

- Optuna hyperparameter tuning:
  ```bash
  pip install -e ".[opt]"
  ```
- Dense retrieval for RAG:
  ```bash
  pip install -e ".[rag_dense]"
  ```

- Bayesian optimization for inverse design (optional):
  ```bash
  pip install -e ".[bo]"
  ```

- EHVI-style multi-objective Bayesian optimization (MOBO) via Ax/BoTorch (optional, heavier deps):
  ```bash
  pip install -e ".[mobo]"
  ```

## Run the full pipeline

```bash
peo run --config config/config.yaml --data examples/deposition/sample_data.csv --name my_first_run
```

## DOE-informed ML features (DOE -> ML)

PEO can optionally use DOE analysis outputs to improve ML representation:

- **Factor screening**: keep only the most influential factors (by DOE p-value)
- **Interaction features**: add numeric-numeric interaction terms (A*B) for pairs that look significant
- Train extra model variants with a `+doe` suffix (for example `gp+doe`, `random_forest+doe`)

Enable it in your config:

```yaml
doe_to_ml:
  enabled: true
  selection:
    p_value_threshold: 0.15
    top_k: 8
    keep_at_least: 3
    always_keep: []
  interactions:
    enabled: true
    p_value_threshold: 0.15
    top_k: 10
```

The selected factors and interaction features are stored in:

- `storage/runs/<run_id>/manifest.json` under `doe_to_ml`
- the RAG vector store as `type=doe_to_ml` documents
- each `*+doe.joblib` model payload (so inverse design applies the same transform automatically)

For a short run (useful for smoke tests and quick iteration):

```bash
peo run --config config/config.yaml --data examples/deposition/sample_data_small.csv --name quick --fast
```

## Query the trained knowledge base

### One-off query

```bash
peo list-runs
peo query --run-id <RUN_ID> --target sheet_resistance_ohm --value 12 --question "sheet resistance 12 ohm"
```

The output is a ranked list of candidate tool settings. Ranking prefers low error to the requested target and low model uncertainty.

### Interactive mode

```bash
peo chat --run-id <RUN_ID>
```

Then type:

```
sheet_resistance_ohm=12
thickness_nm=380
exit
```

You can also type *general questions* (no `=`). In that case PEO will do retrieval-only
and show the most relevant notes from the run (DOE summaries, training metrics, etc.).

### Trend summaries

For questions like "What is the general trend for sheet resistance?" you can run:

```bash
peo trend --run-id <RUN_ID> --target sheet_resistance_ohm
```

Or in `peo chat`, type something like:

```
trend sheet resistance
```

## Iterative updates (closed-loop learning)

After you run an experiment, you can append new measured rows to a prior run and retrain everything from scratch (DOE analysis, ML models, and RAG index) to improve predictability over iterations.

```bash
peo update --parent-run-id <PARENT_RUN_ID> --new-data ./new_measurements.csv --name iter_1
```

By default, `peo update` reuses the parent run's stored config. If you want to change settings (for example, to enable MOBO), pass an override YAML:

```bash
peo update --parent-run-id <PARENT_RUN_ID> --new-data ./new_measurements.csv --name iter_1 \
  --config-override config/config_override.yaml
```

Each update creates a brand-new run directory (with a new run id) and stores the merged dataset inside that run for reproducibility.

## MOBO experiment suggestions (Ax/BoTorch)

If you enable `mobo.enabled: true` in your config, PEO will automatically generate a batch of candidate recipes using EHVI-style multi-objective Bayesian optimization via Ax/BoTorch.

The suggestions are exported to:

- `storage/runs/<run_id>/reports/suggested_recipes_mobo_<run_id>.csv`

See `examples/mobo/` for two demo scenarios:

- maximize thickness while minimizing sheet resistance
- maximize transmittance while keeping resistance under a threshold

## Examples you can run

Synthetic datasets (generated under `examples/`) include 3 scenarios:

- `examples/deposition/`: thickness and sheet resistance
- `examples/etch/`: etch rate and selectivity
- `examples/cvd/`: thickness and transmittance

Windows PowerShell:

```powershell
.\examples\run_examples.ps1
```

Linux/macOS:

```bash
bash examples/run_examples.sh
```

## Output layout

Each run writes:

- `storage/runs/<run_id>/manifest.json`: run metadata and paths
- `storage/runs/<run_id>/doe/`: DOE plan CSVs
- `storage/runs/<run_id>/reports/`:
  - `training_results.json`
  - per-target predictions NPZ
  - DOE analysis JSON
- `storage/runs/<run_id>/models/`: trained models per target
  - If DOE→ML feature engineering is enabled, additional variants are saved with a `+doe` suffix (e.g., `sheet_resistance_ohm__gp+doe.joblib`)
- `storage/runs/<run_id>/plots/`: parity, residuals, coverage, DOE plots
- `storage/runs/<run_id>/rag/vector_store.joblib`: vector store for retrieval

## Add your own dataset

1. Put your measurement CSV anywhere.
2. Create a config YAML (copy `config/config.yaml` or one of the `examples/*/example_config.yaml`).
3. Set:

- `data.tool_parameters`: columns you control (temperature, pressure, power, flow, gas type, etc)
- `data.target_properties`: measured outputs (thickness, sheet resistance, transmittance, etc)
- `data.categorical_params`: map categorical columns to allowed values (or leave empty to infer)

Then:

```bash
peo run --config path/to/your_config.yaml --data path/to/your_measurements.csv --name my_process
```

## DOE-informed ML features (optional)

If you enable `doe_to_ml`, PEO uses DOE analysis (ANOVA p-values + a light interaction scan) to build a *second* engineered feature space per target.

What you get:

- **Feature selection**: keeps only the most influential factors (and all one-hot columns for selected categorical factors)
- **Interaction terms**: adds products of significant numeric-numeric factor pairs (e.g., `temperature_C*power_W`)
- **Additional model variants**: trained and saved with a `+doe` suffix

Config example:

```yaml
doe_to_ml:
  enabled: true
  selection:
    p_value_threshold: 0.15
    top_k: 8
    keep_at_least: 3
  interactions:
    enabled: true
    p_value_threshold: 0.15
    top_k: 10
```

The training report (`training_results.json`) records the selected base features and interaction features per target, and those summaries are also indexed into the RAG store.

## Dev setup

```bash
pip install -e .
pip install -r requirements-dev.txt
ruff check src tests
pytest -q
```

## GitHub CI/CD

This repo includes GitHub Actions workflows:

- `.github/workflows/ci.yml`: lint + tests (including smoke tests)
- `.github/workflows/release.yml`: build wheel/sdist on tags like `v0.1.0`

## How to push to GitHub from scratch (Windows)

1. Create a new empty repository on GitHub (no README, no license).
2. Open PowerShell in the project folder.
3. Run:

```powershell
git init
git add .
git commit -m "Initial commit"

git branch -M main
# Replace with your repo URL:
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git

git push -u origin main
```

To publish a release build via the included workflow:

```powershell
# Example tag
git tag v0.1.0
git push origin v0.1.0
```

## Notes and current scope

- DOE plans are generated as suggested experiment matrices.
- DOE analysis is performed on measured data (screening, p-values, effect sizing, basic interactions).
- Inverse design supports:
  - `method: random` (sampling-based; configurable budget across bounds/categories)
  - `method: bayesopt` (Bayesian optimization over the surrogate; requires `pip install -e ".[bo]" )
- RAG is currently local retrieval only (TF-IDF by default), storing run notes, analysis summaries, and training summaries.

If you want to plug in an LLM later, you can use the vector store results as context and the inverse design output as grounded candidate answers.

