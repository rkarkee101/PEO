# Example commands for Windows PowerShell
# Run from the repo root.

# (Optional) install editable package
# pip install -e .

Write-Host "Running deposition example..."
peo run --config examples\deposition\example_config.yaml --data examples\deposition\sample_data.csv --name deposition
$runId = (Get-ChildItem storage\runs | Sort-Object Name -Descending | Select-Object -First 1).Name
Write-Host "Latest run: $runId"

Write-Host "Query: target sheet_resistance_ohm=12"
peo query --run-id $runId --target sheet_resistance_ohm --value 12 --question "sheet resistance 12 ohm"

Write-Host "Query: target thickness_nm=380"
peo query --run-id $runId --target thickness_nm --value 380 --question "thickness 380 nm"

Write-Host "Multi-target inverse design (deposition)..."
peo query-multi --run-id $runId --targets "thickness_nm=350, sheet_resistance_ohm=12"

Write-Host "Forward prediction (deposition)..."
peo predict --run-id $runId --params "temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar"

Write-Host "Closed-loop demo (export -> simulate -> update)..."
peo query-multi --run-id $runId --targets "thickness_nm=350, sheet_resistance_ohm=12" --export examples\closed_loop\recommended_recipes.csv
python examples\closed_loop\simulate_lab_deposition.py --in examples\closed_loop\recommended_recipes.csv --out examples\closed_loop\new_measurements.csv --seed 123
peo update --parent-run-id $runId --new-data examples\closed_loop\new_measurements.csv --name closed_loop_iter1 --config-override examples\closed_loop\example_config.yaml
$runIdUpdated = (Get-ChildItem storage\runs | Sort-Object Name -Descending | Select-Object -First 1).Name
Write-Host "Updated run: $runIdUpdated"
peo query-multi --run-id $runIdUpdated --targets "thickness_nm=350, sheet_resistance_ohm=12"

Write-Host "Running etch example..."
peo run --config examples\etch\example_config.yaml --data examples\etch\sample_data.csv --name etch
$runId2 = (Get-ChildItem storage\runs | Sort-Object Name -Descending | Select-Object -First 1).Name
Write-Host "Latest run: $runId2"
peo query --run-id $runId2 --target etch_rate_nm_min --value 120 --question "etch rate 120"

Write-Host "Running CVD example..."
peo run --config examples\cvd\example_config.yaml --data examples\cvd\sample_data.csv --name cvd
$runId3 = (Get-ChildItem storage\runs | Sort-Object Name -Descending | Select-Object -First 1).Name
Write-Host "Latest run: $runId3"
peo query --run-id $runId3 --target transmittance_pct --value 75 --question "transmittance 75 percent"

Write-Host "Interactive mode (Ctrl+C to exit):"
# peo chat --run-id $runId3
