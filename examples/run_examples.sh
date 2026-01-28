#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root.

echo "Running deposition example..."
peo run --config examples/deposition/example_config.yaml --data examples/deposition/sample_data.csv --name deposition
RUN_ID=$(ls -1 storage/runs | sort -r | head -n 1)
echo "Latest run: $RUN_ID"
peo query --run-id "$RUN_ID" --target sheet_resistance_ohm --value 12 --question "sheet resistance 12 ohm"

peo query --run-id "$RUN_ID" --target thickness_nm --value 380 --question "thickness 380 nm"

echo "Multi-target inverse design (deposition)..."
peo query-multi --run-id "$RUN_ID" --targets "thickness_nm=350, sheet_resistance_ohm=12"

echo "Forward prediction (deposition)..."
peo predict --run-id "$RUN_ID" --params "temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar"

echo "Closed-loop demo (export -> simulate -> update)..."
peo query-multi --run-id "$RUN_ID" \
  --targets "thickness_nm=350, sheet_resistance_ohm=12" \
  --export examples/closed_loop/recommended_recipes.csv

python examples/closed_loop/simulate_lab_deposition.py \
  --in examples/closed_loop/recommended_recipes.csv \
  --out examples/closed_loop/new_measurements.csv \
  --seed 123

peo update --parent-run-id "$RUN_ID" \
  --new-data examples/closed_loop/new_measurements.csv \
  --name closed_loop_iter1 \
  --config-override examples/closed_loop/example_config.yaml
RUN_ID_UPD=$(ls -1 storage/runs | sort -r | head -n 1)
echo "Updated run: $RUN_ID_UPD"
peo query-multi --run-id "$RUN_ID_UPD" --targets "thickness_nm=350, sheet_resistance_ohm=12"

echo "Running etch example..."
peo run --config examples/etch/example_config.yaml --data examples/etch/sample_data.csv --name etch
RUN_ID2=$(ls -1 storage/runs | sort -r | head -n 1)
peo query --run-id "$RUN_ID2" --target etch_rate_nm_min --value 120 --question "etch rate 120"

echo "Running CVD example..."
peo run --config examples/cvd/example_config.yaml --data examples/cvd/sample_data.csv --name cvd
RUN_ID3=$(ls -1 storage/runs | sort -r | head -n 1)
peo query --run-id "$RUN_ID3" --target transmittance_pct --value 75 --question "transmittance 75 percent"

# Interactive:
# peo chat --run-id "$RUN_ID3"
