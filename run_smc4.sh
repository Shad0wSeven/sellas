#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
.venv/bin/python smc3.py 1.0 1.0 L0_lean 2000 10 > results_v2/smc4_L0.log 2>&1
PM_OVERRIDE='{"f_dur": [0.5, 0.4]}' .venv/bin/python smc3.py 1.0 1.0 L1_lean_wideFdur 2000 10 > results_v2/smc4_L1.log 2>&1
.venv/bin/python smc3.py 1.0 0.0 L2_lean_noIA 2000 10 > results_v2/smc4_L2.log 2>&1
.venv/bin/python smc3.py 0.5 1.0 L3_lean_tau05 2000 10 > results_v2/smc4_L3.log 2>&1
