#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
run() { name=$1; shift; env "$@" .venv/bin/python smc3.py 1.0 1.0 $name 2000 10 > results_v2/bl_$name.log 2>&1; }
run BL5_batLit_gpsPh2 PM_PIN='{"bat12":[0.52,0.06],"bat24":[0.30,0.06],"bat36":[0.18,0.05]}'
run BL6_batKugler_gpsPh2 PM_PIN='{"bat12":[0.58,0.06],"bat24":[0.40,0.06],"bat36":[0.27,0.05]}'
