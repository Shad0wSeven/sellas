# SELLAS REGAL (GPS vs BAT, AML CR2) — modelling code

Patient-level Monte Carlo / SMC models of the REGAL Phase 3 trial, built from public event counts (60 / 72 / 78 deaths), the design paper, and the literature.
Focus: how death-reporting lag affects the BAT arm, and a literature-informed estimate of BAT survival and the final hazard ratio.

| file | what it is |
|---|---|
| `regal_model.py` | v1 one-stage model (Weibull BAT, durable-fraction GPS), ABC calibration, reporting-lag scenarios, O'Brien-Fleming boundaries |
| `regal_model_v2.py` | v2: withdrawal / LTFU, alternative follow-up schedules, two-stage (relapse → post-relapse) structure |
| `patient_model.py` | **v3 patient-level model**: covariates, stratified 1:1 randomisation, CR2→randomisation delay (left truncation), relapse then post-relapse death, transplant leakage, GPS responders with onset delay, visit-based reporting; scientific priors |
| `smc3.py`, `smc3_report.py` | tempered SMC calibration to 60/72/78 with a soft interim-analysis likelihood; HR and success summaries |
| `smc_fit.py`, `smc_report.py` | SMC for the one-stage model |
| `synth_lik.py`, `synth_analyze.py`, `fit_peak*.py`, `refine.py` | synthetic-likelihood and peak-matching experiments |
| `vdm_bridge.py` | converts van der Maas (from-relapse) survival to REGAL's from-randomisation clock |
| `MODEL_EXPLAINED.md` | plain-language explanation of the full model, likelihood and comparison with other models |
| `INPUTS.md` | every Monte Carlo parameter, prior, per-patient input and the Gaussian reporting-lag design |
| `REPORT_v5_why_P_success_is_lower.md`, `REPORT_v6_bat_anchored_gps_flexible.md` | diagnosis of the lower P(success); BAT-anchored / GPS-flexible inference (`PM_PIN`) |
| `REPORT.md`, `REPORT_v2_community_model_choices.md`, `REPORT_v3_patient_model.md` | write-ups (v3 = patient-level model, soft interim likelihood, science-based priors) |
| `results*/` | tables and figures (large pickles are git-ignored; rerun scripts to regenerate) |

Setup: `python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib pandas`.
Example: `.venv/bin/python smc3.py 1.0 1.0 A_tau1_softIA 2000 8` then `.venv/bin/python smc3_report.py`.

Research code, not investment advice. Several inputs come from unverified Reddit posts; see the reports for provenance and caveats.
