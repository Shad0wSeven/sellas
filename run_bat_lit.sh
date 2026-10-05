#!/bin/bash
cd /Users/ayushnayak/Downloads/SLS/regal_model
GPSW='"p_resp":[0.65,0.25,0.2,1.0],"f_dur":[0.5,0.35,0.0,1.0],"lL":[1.1,0.6,0.0,2.5]'
run() { name=$1; shift; env "$@" .venv/bin/python smc3.py 1.0 1.0 $name 2000 10 > results_v2/bl_$name.log 2>&1; }
run BL0_base_for_ref
run BL1_batLit_gpsBroad PM_OVERRIDE="{$GPSW}" PM_PIN='{"bat12":[0.52,0.06],"bat24":[0.30,0.06],"bat36":[0.18,0.05]}'
run BL2_batLit_wider PM_OVERRIDE="{$GPSW}" PM_PIN='{"bat12":[0.52,0.09],"bat24":[0.30,0.09],"bat36":[0.18,0.08]}'
run BL3_batLitKugler PM_OVERRIDE="{$GPSW}" PM_PIN='{"bat12":[0.58,0.06],"bat24":[0.40,0.06],"bat36":[0.27,0.05]}'
run BL4_batLit_gpsBroad_tight PM_OVERRIDE="{$GPSW}" PM_PIN='{"bat12":[0.52,0.04],"bat24":[0.30,0.04],"bat36":[0.18,0.03]}'
