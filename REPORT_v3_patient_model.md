# REGAL v3: patient-level model with a soft interim likelihood and science-based priors

Code: `patient_model.py` (patients, priors), `smc3.py` (calibration), `smc3_report.py`, `smc3_figs.py`. Results: `results_v2/smc3_*.csv`, `fig_smc3_patient_model.png`.

## 1. What was wrong with how patients were modelled, and what v3 does

| Before (v1/v2) | v3 |
|---|---|
| Every patient an i.i.d. draw from one Weibull per arm; BAT 3-yr OS and median set directly | Each patient simulated: covariates, relapse process, post-relapse death, transplant, so BAT survival is **derived**, not assumed |
| No patient heterogeneity (no CR1 duration, cytogenetics, MRD, age) | CR1 duration (≥12 mo), poor cytogenetics, MRD+, age (registry: 50 aged 18–64, 66 aged ≥65), plus frailty |
| 1:1 by a random permutation | 1:1 stratified by CR1 duration × cytogenetics × MRD (alternation within strata) |
| Randomization window handled by a prior on BAT median | Simulated: relapse-free time runs from CR2; a patient enters 0–6 months after CR2 **only if still relapse-free and alive**, so left truncation is generated, not assumed |
| Death as one hazard | Death in remission (age-dependent background × multiplier) **or** relapse then post-relapse survival, so GPS's effect on relapse is compressed in OS |
| GPS = cure fraction on OS | GPS patients are immune responders with prob p_resp; responders get a relapse-hazard multiplier after a patient-specific onset delay; non-responders ≈ observation (no maintenance) |
| BAT = one curve | ~75% on active therapy (relapse HR th_B) and ~25% observation |
| No transplant | Transplant in both arms (ITT), post-transplant 3-yr OS 50% |
| Reporting by arm only | GPS patients still in remission are on dosing visits (death known at once); relapsed GPS and all BAT follow the call schedule, with a real-world-discovery probability |

The prior was checked by simulation before fitting (prior-predictive): BAT relapse-free median **5.9 mo from randomization (8.0 from CR2)**, 1-yr RFS 34% (QUAZAR placebo RFS 4.8 mo, 1-yr 27%; CR2 duration 3–14 mo); BAT OS median 14 mo, 3-yr OS 19% (VDM bridge 8–14 mo, 13–22%). The reference relapse time was lowered from 8 to 3 months to get this.

## 2. Interim analysis in the likelihood (no cutoff)

Calibration target (tempered SMC, 2,000 particles, 27 parameters):

log L = −½ Σ ((60,72,78 − μᵢ(θ)) / τ)² + log P(80th not reported by 11 Aug | θ) + log P(IDMC continued | θ)

P(IDMC continued | θ) = P(no efficacy stop AND passed futility), estimated from 120 simulated trials per particle, with the efficacy bound z_b (prior 2.4 ± 0.15; O'Brien-Fleming at 75% information is 2.34) and the futility HR bound h_f (uniform 0.74–0.92, spanning the community's 0.74 [10% conditional power, current trend], 0.83 [design effect] and 0.89 [Hwang-Shih-DeCani]) treated as unknown nuisance parameters. Parameter sets that make a stop likely are down-weighted smoothly.

**Finding:** in the calibrated model the interim risk is mostly *futility* (about 49%), not efficacy (about 5%). Because GPS's effect is delayed, the interim HR is often above the futility bound. The efficacy-stop worry that dominated the earlier discussion matters less here.

## 3. Priors (each within the range supported by sources)

| Parameter | Prior | Source / reasoning |
|---|---|---|
| Relapse hazard shape kr | N(0.85, 0.2) | CR1 relapse rates 40% / 17% / 2% in years 1–3 (declining hazard) |
| Reference relapse-free median Mr | N(3.0, 0.9) mo | Set by prior-predictive match to RFS and OS above |
| Never-relapse fraction | N(0.08, 0.05) | long-term survivors without transplant are rare (Estey; Breems 5-yr OS 6–12%) |
| HR long vs short CR1 | 0.55 | Breems: CR1 >18 mo vs ≤6 mo, 1-yr OS 57% vs 14% (verified); Thalhammer as quoted on Reddit (unverified) |
| HR poor cytogenetics, MRD+ | 1.6, 1.4 | generic AML prognosis; **my assumptions** |
| HR per decade of age | 1.1 | VDM validation cohort (age 69) 4-yr OS 9% vs 16% at age 58 |
| Cohort: age ~ N(66, 9); shares long CR1 45%, poor 30%, MRD+ 40% | | EU register age counts; stratification factors |
| CR2 → randomization delay | mean 2.5 mo, truncated at 6 | protocol: consent ≤ 6 months after CR2 |
| Post-relapse median | N(5.5, 1.0) mo | Bataller 5.3; ven R/R 5.5–6.8; ADMIRAL-type 5–9 |
| Background mortality | 1.5%/yr at 65, +9%/yr, × N(1.5, 0.4) | US life table (approximate, from memory; **unverified**) |
| Transplant fraction (ITT) | N(0.10, 0.04) | QUAZAR placebo 13.7%, oral aza 6.3% (as quoted on Reddit); post-transplant 3-yr OS fixed at 50% |
| BAT on active therapy / HR | 75% / 0.75 | QUAZAR aza vs placebo RFS HR 0.65 (verified); ~25% observation per EU protocol (CW) |
| GPS immune-responder fraction | N(0.70, 0.10) | Ph2 CR1: 64% any response, CD8 86% in HLA-A*02 (9 of 22 patients, 41%); REGAL random sample 80% (company PR, verified) |
| GPS responder relapse HR | lognormal, median 0.4, σ 0.6 | Ph2 CR1 DFS 16.9 mo, 3-yr OS 47%; wide because OCV-501 (WT1 helper peptide, DFS HR 0.93, 5-yr DFS 36% vs 34%) failed |
| GPS non-responder relapse HR | N(1.0, 0.15) | no maintenance ≈ observation |
| Onset delay | lognormal, median 3 mo | CTL peak by week 12 (Hashii 2026); sipuleucel-T delayed separation ~6 mo; 80% of immunotherapy trials delayed, 13/20 by ≥ 3 mo |
| Enrollment shape, pull-lag on the 72 count (Exp, mean 1 mo), real-world discovery 0.5 ± 0.2 | | assumptions |

## 4. Results (protocol call schedule with 50% real-world discovery)

| Run | GPS responder-HR prior | Expected counts (obs 60/72/78) | BAT 3-yr OS | GPS 3-yr OS | HR given counts & interim | P(success) |
|---|---|---|---|---|---|---|
| A (base) | median 0.4 | 58.5 / 73.3 / 78.2 | 32% (28–36) | 44% (40–48) | 0.69 (0.52–0.89) | **33%** |
| B no interim term | median 0.4 | 58.5 / 73.3 / 78.0 | 32% | 44% | 0.68 | 36% |
| C tolerance 2 events | median 0.4 | 57.1 / 73.8 / 78.7 | 32% | 44% | 0.68 | 35% |
| D tolerance 0.5 events | median 0.4 | 59.4 / 72.7 / 77.8 | 33% | 44% | 0.68 | 34% |
| E stronger GPS | median 0.15 | 58.6 / 73.2 / 77.8 | 30% | 46% | 0.64 (0.50–0.83) | 48% |
| F wide GPS | median 0.4, σ 1.2 | 58.7 / 73.3 / 77.8 | 30% | 47% | 0.65 | 47% |
| G strong GPS + 80% responders | median 0.15 | 58.7 / 73.2 / 77.8 | 29% | 47% | 0.63 (0.49–0.82) | **53%** |

BAT median OS in the posterior is about 17–18 months and GPS about 26–31 months. In run A the hazard ratio before conditioning is 0.79 (0.51–1.2) with P(success) 20%.

## 5. Reading the result

- **The three constraints are in tension.** The counts need about 37% of all patients alive at about 41 months of mean follow-up. Literature BAT (3-yr OS ≈ 19%) would then require GPS at ≈ 55–60%, above anything GPS has shown (Phase 2 **CR1** 3-yr OS 47%, and CR2 is harder). The model compromises: BAT is pulled up to about 30%, GPS stays near 45–47%.
- **Hence a weaker answer than v1/v2.** The one-stage models let GPS carry a durable fraction up to 65% directly on OS and returned P(success) 56–63%. In v3 GPS acts only through relapse among responders, after a delay, and post-relapse death compresses the OS gain, so GPS cannot reach 55–60% under Phase 2-consistent priors. P(success) is 33–53% across GPS priors.
- **The interim term barely matters** (33% vs 36% without it), because the posterior already sits where an efficacy stop is unlikely.
- **Parameter shifts are moderate but not trivial** (run A, shift in prior SDs): relapse shape kr −1.6, post-relapse median −1.3 (5.5 → 4.4 mo), enrollment shape −1.2, pull-lag +1.35 (1.4 mo), GPS responder HR −1.2, frailty SD +0.9. All remain inside the prior ranges (`results_v2/smc3_params.csv`).

## 6. Caveats

BAT 3-yr OS of 30% in the posterior is above the VDM-bridge range (13–22%); the model reaches it only because the counts demand survivors, so it is a signal that either BAT is stronger than the literature or GPS is stronger than Phase 2. Several covariate HRs, the life-table hazard and the transplant fraction are my assumptions or unverified. The model has no explicit HLA split, one onset-delay distribution, and the SAP is assumed unstratified log-rank.

## 7. Lean version (default): GPS carries a tail, 13 free parameters

GPS is now a durable-responder mixture (immune response `p_resp`, durable fraction `f_dur`, onset delay), and nuisance parameters with solid literature values are fixed (see `INPUTS.md`). The prior-predictive BAT arm matches the literature: median 13.4 mo, 1-yr OS 54%, 3-yr OS 16%.

| Run | Expected counts | BAT 3-yr OS (median) | GPS 3-yr OS (median) | HR given counts & interim | P(success) |
|---|---|---|---|---|---|
| L0 base (f_dur ~ N(0.55, 0.2)) | 58.8 / 74.0 / 77.5 | 33% (19.8 mo) | 48% (31.5 mo) | 0.65 (0.48–0.84) | 46% |
| L1 wide f_dur prior | 58.8 / 73.9 / 77.4 | 32% (19.1) | 49% (33.8) | 0.62 (0.47–0.82) | 54% |
| L2 no interim term | 58.8 / 74.0 / 77.5 | 34% (20.3) | 47% (30.2) | 0.66 | 42% |
| L3 tolerance 0.5 events | 59.4 / 73.6 / 77.0 | 33% (19.5) | 49% (33.1) | 0.64 | 48% |

The posterior moves GPS toward a large durable component: responders 77%, durable among responders 82% (about 63% of GPS patients), onset median about 2.4 months. Both arms keep tails (BAT 3→5-yr OS 34→23%, GPS 47→39%). BAT 3-yr OS stays near 33%, above the literature range, because the counts still demand about 38% alive.
