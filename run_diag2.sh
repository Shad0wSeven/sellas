#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
run() { name=$1; shift; env "$@" .venv/bin/python smc3.py 1.0 1.0 $name 2000 10 > results_v2/diag_$name.log 2>&1; }
run X6_pinBAT19_sd07 PM_PIN='{"bat36":[0.19,0.07]}'
run X7_pinBAT19_sd10 PM_PIN='{"bat36":[0.19,0.10]}'
run X8_pinBAT19_sd15 PM_PIN='{"bat36":[0.19,0.15]}'
run X9_pinBAT19_sd02 PM_PIN='{"bat36":[0.19,0.02]}'
