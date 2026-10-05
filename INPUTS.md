# Inputs and parameters of the patient-level Monte Carlo (`patient_model.py`, lean mode = default)

One Monte Carlo draw = (1) draw a parameter vector theta from the priors below, (2) simulate 127 patients under theta, (3) add each arm's Gaussian reporting lag to the biological death date, (4) run a Cox model / log-rank on the dated deaths at the interim (60 events) and at the reported 80th event. SMC re-weights theta so the expected reported counts match 60 / 72 / 78 and the interim outcome (continued) is likely.

## How BAT and GPS are parameterized

- **Everyone** follows the same patient process: relapse-free time from CR2 (declining Weibull hazard, patient multipliers for CR1 duration, cytogenetics, MRD, age, frailty), then post-relapse survival, with background mortality and transplant.
- **BAT** = that process with a relapse-hazard multiplier `th_B` for the ~75% on active therapy. BAT survival is an *output*.
- **GPS** = the same process plus a *durable-responder mixture*: a patient mounts an immune response with probability `p_resp`; a responder is durable with probability `f_dur`; after a patient-specific onset delay (median `exp(lL)` months) a durable responder's relapse hazard drops to 5% of baseline (a cure-like plateau with a small leak). Non-durable responders and non-responders behave like observation. GPS survival (including a tail) is an *output*.

So GPS adds 3 free parameters and BAT adds 1, on top of 6 shared biology parameters and 3 design/interim nuisance parameters: **13 free parameters** in total.

## A. Free parameters (13)

| name | applies to | meaning | prior | allowed range |
|---|---|---|---|---|
| `Mr` | shared | median relapse-free time from CR2, reference (no-maintenance) patient, months | N(3, 0.9) | [1.5, 10] |
| `kr` | shared | Weibull shape of relapse hazard (<1 = declining) | N(0.85, 0.2) | [0.5, 1.4] |
| `c_b` | shared | fraction who never relapse (both arms) | N(0.06, 0.03) | [0, 0.3] |
| `b_long` | shared | log HR long (>=12 mo) vs short CR1 | N(-0.598, 0.25) | [-1.6, 0.2] |
| `sig_f` | shared | frailty SD (log-hazard) | N(0.5, 0.2) | [0.1, 1] |
| `m_p` | shared | median post-relapse survival, months | N(5.5, 1) | [3.5, 8] |
| `th_B` | BAT | BAT: relapse HR active therapy vs observation | N(0.75, 0.15) | [0.4, 1.1] |
| `p_resp` | GPS | GPS: fraction who mount an immune response | N(0.7, 0.1) | [0.35, 0.95] |
| `f_dur` | GPS | GPS: fraction of responders with a durable, cure-like effect | N(0.55, 0.2) | [0.05, 1] |
| `lL` | GPS | GPS: log median onset delay (months) | N(1.1, 0.3) | [0, 2.08] |
| `gamma` | design | enrollment-curve shape | Uniform(1.4, 2.8) | [1.4, 2.8] |
| `z_b` | interim | interim efficacy z bound | N(2.4, 0.15) | [2.2, 3] |
| `h_f` | interim | interim futility HR bound | Uniform(0.74, 0.92) | [0.74, 0.92] |

## B. Fixed at literature values (not sampled)

share long CR1 0.45, poor cytogenetics 0.30, MRD+ 0.40; log HR poor cytogenetics ln 1.6, MRD+ ln 1.4, age per decade ln 1.1, post-relapse age effect ln 1.15; mean CR2-to-randomisation delay 2.5 months (truncated at 6); background mortality multiplier 1.5 (baseline 1.5%/yr at 65, +9%/yr); transplant fraction 0.10 in both arms (post-transplant 3-yr OS 50%, mean time to transplant 8 months); BAT on active therapy 0.75; durable-responder residual relapse hazard 0.05.

## C. Random inputs drawn per patient

CR1 duration, cytogenetics, MRD, age ~ N(66, 9), frailty, time from CR2 to randomisation, relapse time, never-relapse flag, remission-death time, post-relapse survival, transplant flag and time, BAT active flag, GPS responder flag, GPS durable flag, onset delay (log-normal sd 0.4), enrollment jitter, stratified 1:1 arm assignment, and the reporting-lag draw.

## D. Reporting lag (your design)

`reported_date = biological_death_date + max(0, N(mu_arm, sd_arm)) days`, separate for BAT and GPS. `LAG = {mu_bat, sd_bat, mu_gps, sd_gps}` in days, all 0 by default. Set with `PM_LAG='{"mu_bat":120,"sd_bat":60,"mu_gps":15,"sd_gps":10}'`.

## E. Data, outputs, switches

Data: reported counts 60 (10 Dec 2024), 72 (26 Dec 2025), 78 (11 May 2026); 80th not reported by 11 Aug 2026; IDMC continued at the interim. Outputs per trial: counts at the three dates, interim and final Cox HR (standard Cox partial likelihood, validated against `lifelines`), success flag (log-rank z below -2.01), arm survival at 12 / 36 / 60 months and median. `PM_MODE=full` restores the earlier 25-parameter version; `PM_OVERRIDE='{"f_dur":[0.5,0.4]}'` changes a prior.
