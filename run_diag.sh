#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
run() { name=$1; shift; env "$@" .venv/bin/python smc3.py 1.0 1.0 $name 2000 10 > results_v2/diag_$name.log 2>&1; }
run X0_base
run X1_pinBAT19 PM_PIN='{"bat36":[0.19,0.04]}'
run X2_pinBAT25 PM_PIN='{"bat36":[0.25,0.04]}'
run X3_noDelay PM_OVERRIDE='{"lL":[0.0,0.05]}'
run X4_noDelay_pinBAT19 PM_OVERRIDE='{"lL":[0.0,0.05]}' PM_PIN='{"bat36":[0.19,0.04]}'
run X5_maxGPS_noDelay_pinBAT19 PM_OVERRIDE='{"lL":[0.0,0.05],"p_resp":[0.9,0.03],"f_dur":[0.95,0.04]}' PM_PIN='{"bat36":[0.19,0.04]}'
