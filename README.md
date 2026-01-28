# Process Engineering Optimizer

A universal, config-driven workflow for:

- Designing experiments (DOE plans)
- Analyzing measured process data (screening + effect sizing)
- Training predictive models with uncertainty
- Forward prediction: predict properties from tool parameters
- Inverse design: propose tool parameters for desired target property/targets
- Persisting everything (reports, plots, metrics, docs) for later Retrieval-Augmented Generation (RAG)

This repository does **not** integrate an LLM. It focuses on building a strong offline knowledge base (RAG) and an inverse-design engine you can query via CLI.

## What you get

After one command (`peo run`) the project will:

1. Read your measurement CSV (tool parameters + target properties)
2. Infer parameter bounds and categorical levels (or use your config overrides)
3. Generate DOE plan CSVs (Full factorial, Fractional factorial, Plackett-Burman, Latin hypercube, Central composite, Box-Behnken)
4. Run statistical DOE analysis on your measured data (main effects, interactions, p-values, ANOVA-style summaries)
5. Train multiple probabilistic models per target, including:
   - Gaussian Process (GP)
   - Random Forest ensemble
   - MLP ensemble
   - DOE-native Response Surface Model (RSM; Bayesian ridge over quadratic/interaction terms)
   - Hybrid RSM+GP residual (response-surface mean + GP residual)
   - DOE-informed neural hybrid RSM+MLP residual (response-surface trend + neural residual ensemble)
   - Optional DOE-informed variants (`+doe`) that use DOE p-values to select influential factors and add interaction/quadratic features
   - Automatic hyperparameter autotuning:
     - Uses Optuna when installed
     - Otherwise falls back to an internal random-search tuner (no extra deps)
   - CV-first model selection (recommended for DOE-sized datasets)
   - Automatic overfit guard (train/test R2 gap) used both for reporting and model selection penalties
   - Uncertainty calibration (std scaling) and test-time uncertainty coverage checks
6. Save plots (parity, residuals, uncertainty coverage) and JSON reports
7. Build a local vector store (TF-IDF by default) so you can retrieve run knowledge later
8. Support:
   - Inverse design queries (single-target or multi-target)
   - Forward prediction queries (tool parameters → predicted properties)

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
- **Quadratic features**: add curvature terms (A^2) for numeric factors that look significant
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
  quadratic:
    enabled: true
    p_value_threshold: 0.20
    top_k: 6
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

### Inverse design (single target)

```bash
peo list-runs
peo query --run-id <RUN_ID> --target sheet_resistance_ohm --value 12 --question "sheet resistance 12 ohm"
```

The output is a ranked list of candidate tool settings. Ranking prefers low misfit to the requested target, lower model uncertainty, and (optionally) lower OOD distance.

You can also constrain the search with fixed parameters:

```bash
peo query --run-id <RUN_ID> --target thickness_nm --value 350 --fixed "gas=O2"
```

### Inverse design (multiple targets)

```bash
peo query-multi --run-id <RUN_ID> \
  --targets "thickness_nm=350, sheet_resistance_ohm=12" \
  --question "hit thickness and sheet resistance simultaneously"
```

### Forward prediction (process → properties)

```bash
peo predict --run-id <RUN_ID> \
  --params "temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar"
```

Optionally restrict which properties to predict:

```bash
peo predict --run-id <RUN_ID> \
  --params "temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar" \
  --targets "thickness_nm,sheet_resistance_ohm"
```

### Export a CSV template for lab execution (active learning)

Both `query` and `query-multi` support `--export`, which writes a CSV template containing the recommended recipes and blank target columns to be filled with measured outcomes:

```bash
peo query-multi --run-id <RUN_ID> \
  --targets "thickness_nm=350, sheet_resistance_ohm=12" \
  --export ./recommended_recipes.csv
```

After running the experiments, append the measured outcomes via `peo update`.

### Interactive mode

```bash
peo chat --run-id <RUN_ID>
```

Then type:

```
sheet_resistance_ohm=12
thickness_nm=380
thickness_nm=350, sheet_resistance_ohm=12
temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar
thickness_nm=350, gas=O2
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

For an end-to-end demo (export recipes → simulate measurements → update), see `examples/closed_loop/`.

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

Each scenario includes multiple configs to illustrate different modeling and optimization modes:

- Deposition (`examples/deposition/`)
  - `example_config.yaml`: baseline DOE-informed ML
  - `example_config_rsm.yaml`: DOE-native response surface + hybrid RSM+GP residual
  - `example_config_robust_inverse.yaml`: robust inverse design (uncertainty + OOD penalty + diversity)
  - `example_config_multi_target.yaml`: multi-target inverse design with tolerances
- Etch (`examples/etch/`)
  - `example_config.yaml`: baseline
  - `example_config_rsm.yaml`: DOE-native response surface + hybrid RSM+GP residual
  - `example_config_robust_inverse.yaml`: robust multi-target inverse design
- CVD (`examples/cvd/`)
  - `example_config.yaml`: baseline
  - `example_config_rsm.yaml`: DOE-native response surface + hybrid RSM+GP residual
  - `example_config_robust_inverse.yaml`: robust multi-target inverse design
- Closed-loop (active learning) demo: `examples/closed_loop/`

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

## DOE-informed ML features

See the earlier section “DOE-informed ML features (DOE → ML)” for the recommended configuration (including interactions and quadratic terms) and where artifacts are stored.

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
- Forward prediction uses the best-ranked (CV-first) surrogate per target and can report uncertainty when available.
- Inverse design supports:
  - `method: random` (sampling-based; configurable budget across bounds/categories)
  - `method: bayesopt` (Bayesian optimization over the surrogate; requires `pip install -e ".[bo]" )
  - multi-target inverse design (`peo query-multi`) with optional tolerances/weights
  - optional uncertainty and OOD penalties and diversity filtering (to handle non-uniqueness)
- RAG is currently local retrieval only (TF-IDF by default), storing run notes, analysis summaries, and training summaries.

If you want to plug in an LLM later, you can use the vector store results as context and the inverse design output as grounded candidate answers.

