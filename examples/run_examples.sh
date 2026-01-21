#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root.

echo "Running deposition example..."
peo run --config examples/deposition/example_config.yaml --data examples/deposition/sample_data.csv --name deposition
RUN_ID=$(ls -1 storage/runs | sort -r | head -n 1)
echo "Latest run: $RUN_ID"
peo query --run-id "$RUN_ID" --target sheet_resistance_ohm --value 12 --question "sheet resistance 12 ohm"

peo query --run-id "$RUN_ID" --target thickness_nm --value 380 --question "thickness 380 nm"

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
