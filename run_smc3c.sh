#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
.venv/bin/python smc3.py 1.0 1.0 Z0_lag0 2000 10 > results_v2/smc3_Z0.log 2>&1
PM_OVERRIDE='{"lth_r": [-1.9, 0.6], "p_resp": [0.8, 0.08]}' .venv/bin/python smc3.py 1.0 1.0 Z1_lag0_strongGPS 2000 10 > results_v2/smc3_Z1.log 2>&1
PM_LAG='{"mu_bat": 120, "sd_bat": 60, "mu_gps": 15, "sd_gps": 10}' .venv/bin/python smc3.py 1.0 1.0 Z2_lagBAT120d 2000 10 > results_v2/smc3_Z2.log 2>&1
