# MOBO examples (Ax/BoTorch)

These examples show EHVI-style multi-objective Bayesian optimization (MOBO) using **Ax/BoTorch**.

## Install optional deps

```bash
pip install -e ".[mobo]"
```

## Example 1: maximize thickness while minimizing sheet resistance

```bash
peo run --config examples/mobo/example_config_thickness_vs_resistance.yaml \
  --data examples/mobo/sample_data_mobo.csv \
  --name mobo_thickness_vs_resistance
```

The pipeline will automatically export suggested experiments to:

- `storage/runs/<RUN_ID>/reports/suggested_recipes_mobo_<RUN_ID>.csv`

## Example 2: maximize transmittance while keeping resistance under a threshold

```bash
peo run --config examples/mobo/example_config_transmittance_threshold.yaml \
  --data examples/mobo/sample_data_mobo.csv \
  --name mobo_transmittance_threshold
```

This uses an outcome constraint:

- `sheet_resistance_ohm <= 12`

## Iterative update loop

After you execute some suggested experiments in the lab and save the new measurements to a CSV with the same columns, update the models by retraining from scratch on the combined dataset:

```bash
peo update \
  --parent-run-id <RUN_ID> \
  --new-data examples/mobo/new_measurements_iter1.csv \
  --name iter_1
```

You can optionally override config (for example, to enable MOBO or change objectives):

```bash
peo update \
  --parent-run-id <RUN_ID> \
  --new-data examples/mobo/new_measurements_iter1.csv \
  --name iter_1 \
  --config-override config/config_override.yaml
```
