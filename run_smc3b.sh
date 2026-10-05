#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
PM_OVERRIDE='{"lth_r": [-1.9, 0.6]}' .venv/bin/python smc3.py 1.0 1.0 E_strongGPS_softIA 2000 10 > results_v2/smc3_E.log 2>&1
PM_OVERRIDE='{"lth_r": [-0.92, 1.2]}' .venv/bin/python smc3.py 1.0 1.0 F_wideGPS_softIA 2000 10 > results_v2/smc3_F.log 2>&1
PM_OVERRIDE='{"lth_r": [-1.9, 0.6], "p_resp": [0.8, 0.08]}' .venv/bin/python smc3.py 1.0 1.0 G_strongGPS_resp80 2000 10 > results_v2/smc3_G.log 2>&1
