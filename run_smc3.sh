#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
.venv/bin/python smc3.py 1.0 1.0 A_tau1_softIA 2000 10 > results_v2/smc3_A.log 2>&1
.venv/bin/python smc3.py 1.0 0.0 B_tau1_noIA 2000 10 > results_v2/smc3_B.log 2>&1
.venv/bin/python smc3.py 2.0 1.0 C_tau2_softIA 2000 10 > results_v2/smc3_C.log 2>&1
.venv/bin/python smc3.py 0.5 1.0 D_tau05_softIA 2000 10 > results_v2/smc3_D.log 2>&1
