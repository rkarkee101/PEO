# Closed-loop (active learning) example

This example demonstrates a *closed-loop* iteration:

1. Train surrogates on an initial dataset
2. Use inverse design to propose experiments for target properties
3. Run experiments in the lab (simulated here)
4. Append new measurements and retrain (`peo update`)

The goal is to improve both forward (process → properties) and inverse (properties → process) prediction accuracy over iterations.

## 1) Train the initial run

From the repo root:

```bash
peo run --config examples/closed_loop/example_config.yaml \
  --data examples/deposition/sample_data.csv \
  --name closed_loop_iter0
```

Get the run id:

```bash
peo list-runs
```

## 2) Export inverse-design suggestions (recipes)

Multi-target inverse design (you can change the targets):

```bash
peo query-multi --run-id <RUN_ID> \
  --targets "thickness_nm=350, sheet_resistance_ohm=12" \
  --export examples/closed_loop/recommended_recipes.csv
```

The export is a CSV template:

- recommended tool parameters for each suggestion
- blank target columns to be filled with measured outcomes
- convenience prediction columns (`pred_*`, `std_*`) that are ignored by `peo update`

## 3) Simulate lab measurements (replace with your real data)

In the real workflow, you would run the experiments and fill in the measured outcomes.
To keep this repo self-contained, we provide a simulator for the synthetic deposition example.

```bash
python examples/closed_loop/simulate_lab_deposition.py \
  --in examples/closed_loop/recommended_recipes.csv \
  --out examples/closed_loop/new_measurements.csv \
  --seed 123
```

## 4) Update (append new data and retrain)

```bash
peo update --parent-run-id <RUN_ID> \
  --new-data examples/closed_loop/new_measurements.csv \
  --name closed_loop_iter1
```

The new run gets its own run id and is fully reproducible (the merged dataset is saved under the new run).

## 5) Repeat

Repeat steps 2–4 for multiple iterations. Each iteration should improve surrogate quality, especially in the region the optimizer is exploring.
