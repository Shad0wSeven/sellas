#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
P=.venv/bin/python
$P smc_bio.py F1_main 1.0 1.0 broad 2000 > results_v2/sb_F1.log 2>&1
$P smc_bio.py F2_batLooser 1.0 2.0 broad 2000 > results_v2/sb_F2.log 2>&1
$P smc_bio.py F3_noBATdata 1.0 0 broad 2000 > results_v2/sb_F3.log 2>&1
$P smc_bio.py F4_gpsPh2 1.0 1.0 ph2 2000 > results_v2/sb_F4.log 2>&1
$P smc_bio.py F5_batTighter 1.0 0.6 broad 2000 > results_v2/sb_F5.log 2>&1
