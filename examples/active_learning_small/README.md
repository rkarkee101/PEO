# Active Learning (Small / Toy) Example

This folder contains a **tiny synthetic dataset** and **fast config** that runs quickly and demonstrates:
- DOE-informed Neural Hybrid surrogate (`rsm_mlp`)
- forward prediction
- inverse design (single and multi-target)
- a simple *active learning* loop via `peo update`

## Files
- `sample_data_tiny.csv` — tiny training dataset (24 rows)
- `example_config_doe_nn_active_learning_small.yaml` — fast config tuned for small data
- `active_learning_update_template.csv` — blank outcomes template for `peo update`

## 1) Train a run
From the project root:

```powershell
peo run --config examples/active_learning_small/example_config_doe_nn_active_learning_small.yaml `
        --data   examples/active_learning_small/sample_data_tiny.csv `
        --name   al_small_v1
```

## 2) Forward prediction
```powershell
peo predict --run-id <RUN_ID> --params "temperature_C=200,pressure_mTorr=10,power_W=150,flow_sccm=40,gas=Ar"
```

## 3) Inverse design (multi-target)
```powershell
peo query-multi --run-id <RUN_ID> --targets "thickness_nm=220,sheet_resistance_ohm=12" --export .\recommended_recipes.csv
```

## 4) Active learning update loop
1. Run the recipes in `recommended_recipes.csv` in the lab.
2. Fill in the measured `thickness_nm` and `sheet_resistance_ohm` columns (keep the process-parameter columns unchanged).
3. Update:

```powershell
peo update --parent-run-id <RUN_ID> --new-data .\recommended_recipes.csv --name al_small_iter1
```

Repeat (iter2, iter3, ...) as you collect more outcomes.

### Quick test without the lab
To test the `update` plumbing without real outcomes, you can copy
`active_learning_update_template.csv` to a new file and fill dummy outcome numbers.
