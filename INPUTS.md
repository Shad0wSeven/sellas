# Inputs and parameters of the patient-level Monte Carlo (`patient_model.py`)

One Monte Carlo draw = (1) draw a parameter vector theta from the priors below, (2) simulate 127 patients under theta, (3) turn their biological deaths into reported death dates by adding the arm's Gaussian lag, (4) run a Cox model / log-rank on the dated deaths at the interim (60 events) and at the reported 80th event. SMC then re-weights theta so the model's expected reported counts match 60 / 72 / 78 and the interim outcome (continued) is likely.

## A. Parameters sampled per Monte Carlo draw (25)

| name | applies to | meaning | prior | allowed range |
|---|---|---|---|---|
| `Mr` | shared | median relapse-free time from CR2 for the reference (no-maintenance) patient, months | N(3, 0.9) | [1.5, 10] |
| `kr` | shared | Weibull shape of relapse hazard (<1 = declining) | N(0.85, 0.2) | [0.5, 1.4] |
| `c_b` | shared | fraction who never relapse | N(0.08, 0.05) | [0, 0.3] |
| `b_long` | shared | log HR for long (>=12 mo) vs short CR1 | N(-0.598, 0.25) | [-1.6, 0.2] |
| `b_poor` | shared | log HR poor cytogenetics | N(0.47, 0.25) | [-0.2, 1.2] |
| `b_mrd` | shared | log HR MRD+ | N(0.336, 0.25) | [-0.2, 1] |
| `b_age` | shared | log HR per 10 years of age | N(0.0953, 0.1) | [-0.1, 0.5] |
| `sig_f` | shared | frailty SD (log-hazard) | N(0.5, 0.2) | [0.1, 1] |
| `p_long` | shared | share with long CR1 | N(0.45, 0.1) | [0.2, 0.7] |
| `p_poor` | shared | share poor cytogenetics | N(0.3, 0.08) | [0.1, 0.5] |
| `p_mrd` | shared | share MRD+ | N(0.4, 0.1) | [0.15, 0.65] |
| `u_mean` | shared | mean months from CR2 to randomisation (truncated at 6) | N(2.5, 0.8) | [1, 4] |
| `m_p` | shared | median post-relapse survival, months | N(5.5, 1) | [3.5, 8] |
| `b_prage` | shared | log HR post-relapse per 10 years | N(0.14, 0.1) | [-0.1, 0.5] |
| `m_bg` | shared | background-mortality multiplier (age-dependent baseline) | N(1.5, 0.4) | [0.8, 3] |
| `p_sct` | shared | fraction transplanted (both arms) | N(0.1, 0.04) | [0.02, 0.25] |
| `p_act` | BAT | fraction of BAT on active therapy (HMA/ven/LDAC) | N(0.75, 0.1) | [0.4, 0.95] |
| `th_B` | BAT | relapse HR, BAT active vs observation | N(0.75, 0.15) | [0.4, 1.1] |
| `p_resp` | GPS | fraction of GPS patients who are immune responders | N(0.7, 0.1) | [0.35, 0.95] |
| `lth_r` | GPS | log relapse HR for GPS responders after onset | N(-0.916, 0.6) | [-3.51, 0] |
| `th_nr` | GPS | relapse HR for GPS non-responders vs observation | N(1, 0.15) | [0.7, 1.4] |
| `lL` | GPS | log median onset delay of GPS effect (months) | N(1.1, 0.3) | [0, 2.08] |
| `gamma` | design | enrollment-curve shape | Uniform(1.4, 2.8) | [1.4, 2.8] |
| `z_b` | interim | interim efficacy z bound | N(2.4, 0.15) | [2.2, 3] |
| `h_f` | interim | interim futility HR bound | Uniform(0.74, 0.92) | [0.74, 0.92] |

`shared` parameters describe the underlying leukaemia/patient biology and apply to every patient in both arms. `BAT` and `GPS` parameters differ by arm. `design` and `interim` parameters describe enrollment and the unknown interim boundaries.

## B. Random inputs drawn per patient (inside each draw)

CR1 duration (long/short), cytogenetics (poor/other), MRD, age ~ N(66, 9), frailty z, months from CR2 to randomisation, relapse event time, never-relapse flag, remission-death time, post-relapse survival time, transplant flag and time, BAT active-therapy flag, GPS responder flag, GPS onset delay (log-normal, sd 0.4), enrollment jitter, stratified arm assignment (1:1), and the reporting-lag draw.

## C. Fixed inputs

Enrollment anchors (first patient 8 Feb 2021; 105 by 29 Nov 2023; 123 by 26 Mar 2024; 127 by 29 Apr 2024); 64 BAT / 63 GPS; 126-127 patients; background hazard 0.0013 per month at age 65 with +9% per year of age; post-transplant 3-year OS 50%; mean time to transplant 8 months; 80 events at final analysis, interim at 60; one-sided alpha 0.025 with O'Brien-Fleming spending (final HR boundary about 0.638).

## D. Reporting-lag inputs (your design)

`reported_date = biological_death_date + max(0, N(mu_arm, sd_arm)) days`, separate for BAT and GPS.
`LAG = {mu_bat, sd_bat, mu_gps, sd_gps}` in days, **all 0 by default** (so reported = biological). Set with `PM_LAG='{"mu_bat":120,"sd_bat":60,"mu_gps":15,"sd_gps":10}'`.
A demonstration run with BAT lag N(120, 60) days and GPS N(15, 10) days changes the answer little (P(success) 39% vs 39%).

## E. Calibration data and outputs

Data: reported counts 60 (10 Dec 2024), 72 (26 Dec 2025), 78 (11 May 2026); 80th not reported by 11 Aug 2026; IDMC continued at the 60-event interim.
Outputs per simulated trial: reported death counts at the three dates, interim log-rank z and Cox HR, final Cox HR at the reported 80th event, success flag (log-rank z below -2.01), arm-specific survival (12- and 36-month OS, median), relapse and transplant rates.
The Cox model is a standard Cox partial-likelihood fit (Newton iterations, Breslow ties) of GPS vs BAT on the dated deaths; it was validated against `lifelines` (HR 0.557 vs 0.557).
