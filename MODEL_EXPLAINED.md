# The REGAL model explained (patient-level, likelihood-calibrated)

## 1. The question
REGAL randomised 127 AML patients in second remission (CR2) to GPS (WT1 vaccine) or best available therapy (BAT). Only pooled public facts exist: reported deaths 60 (10 Dec 2024), 72 (26 Dec 2025), 78 (11 May 2026); the 80th not reported by 11 Aug 2026; the IDMC let the trial continue at the 60-event interim. Success = the final stratified-style Cox/log-rank test at the 80th reported event crosses an O'Brien-Fleming boundary (HR about 0.64). The model asks: **which versions of reality (BAT survival, GPS effect) could have produced those facts, and how often would they win?**

## 2. A generative model of the trial (what is simulated)

For one parameter set theta, the model simulates a whole trial, patient by patient:

1. **Enrollment.** 127 patients, dates following the disclosed anchors (105 by Nov 2023, 123 by Mar 2024, 127 by Apr 2024) with a back-loading shape parameter.
2. **Who the patients are.** CR1 duration (long/short), cytogenetics, MRD, age (about N(66, 9)), plus a frailty term. Randomisation is 1:1 stratified on these.
3. **A leukaemia clock that starts at CR2.** Relapse time follows a declining Weibull hazard (most relapses early, a long tail), multiplied by patient factors and the arm effect. A patient enters the trial 0-6 months after CR2 **only if still relapse-free and alive**, so the entry-window selection effect is simulated rather than assumed.
4. **Relapse, then death.** A patient dies in remission (age-dependent background mortality) or relapses and then dies after a post-relapse survival time (median about 5 months). Some patients (about 10%, both arms) are transplanted and then survive with 3-year OS 50%.
5. **BAT arm.** About 75% are on active therapy (azacitidine/venetoclax/low-dose cytarabine) with a relapse-hazard multiplier (about 0.75); the rest are observed. BAT survival is an output.
6. **GPS arm.** A patient mounts an immune response with probability `p_resp`; a responder is durable with probability `f_dur`; after an onset delay (median about 3 months) a durable responder's relapse hazard drops to 5% of baseline (a cure-like tail with a small leak). Everyone else behaves like observation. GPS survival is an output. Because the benefit acts on relapse and post-relapse death is unchanged, OS gain is compressed relative to RFS gain.
7. **Observation layer.** A biological death becomes a reported death after a lag: `lag = max(0, N(mu, sd))` days, separate for BAT and GPS (all 0 for now).
8. **Trial analysis.** At 60 reported events (interim) and at the reported 80th event, run a Cox model of GPS vs BAT on the dated deaths (validated against `lifelines`), plus the log-rank test used for the boundary.

## 3. Parameters (13 free) and where the priors come from
Shared biology: median relapse-free time, relapse-hazard shape, never-relapse fraction, long-CR1 hazard ratio, frailty, post-relapse median. BAT: active-therapy hazard ratio. GPS: `p_resp`, `f_dur`, onset delay. Design and interim: enrollment shape, efficacy z-bound, futility HR-bound. Nuisance quantities with solid literature values are fixed (cohort mix, covariate hazard ratios, transplant fraction, background mortality). Priors are centred on the sources in `REPORT_v3_patient_model.md` and **checked by simulation before fitting**: the prior-predictive BAT arm has relapse-free median 5.9 months from randomisation, OS median 13 months, 3-year OS 16% (van der Maas bridge 13-22%). Full list: `INPUTS.md`.

## 4. The likelihood and how calibration works
Each parameter set gets a log-likelihood from the data:

- **Counts.** `-0.5 * sum(((60,72,78) - expected reported counts)/tau)^2`, with tolerance tau (1 event). The expected counts come from 120 simulated trials per parameter set, so the model's expected path must sit near the observed one.
- **The stall.** `log P(80th not reported by 11 Aug)`.
- **The interim, softly.** `log P(IDMC continued | theta)` = probability of no efficacy stop and passing futility, with the efficacy bound and futility HR bound as uncertain nuisance parameters. Parameter sets that make a stop likely are down-weighted, not cut.
- **Optional BAT literature data.** A soft likelihood on population BAT 12/24/36-month OS (van der Maas bridge: 0.52 / 0.30 / 0.18) with an adjustable width.

A tempered **sequential Monte Carlo** moves 2,000 particles from the prior to the posterior through a ladder of increasing likelihood weight, with resampling and Metropolis moves, so you can watch parameters shift. Then each posterior particle simulates 100 realised trials; trials are weighted to match the observed counts (kernel, sigma 2 events) and to have continued at the interim. HR and P(success) come from those realised trials.

## 5. What it says
- Counts and the interim move P(success) from 9% (prior) to 46% (calibrated, no BAT outcome anchor): odds multiplied by about 9. Adding the BAT literature as data multiplies the odds by about 36-67 (77-87%).
- Without an outcome anchor the data drag BAT 3-year OS to about 33%, above the literature, because the counts need about 38% of patients alive.
- Anchoring BAT to the literature forces GPS to near-cure (about 70-90% durable), well above Phase 2. So P(success) is mainly a function of the BAT anchor: 59% (BAT 27%) to 94% (BAT 18% ± 3).
- Reporting lag and the interim term change little.

## 6. How it compares with the community
| | P(success) | underlying HR |
|---|---|---|
| Ten community models (self-reported, median) | 84% (58-94%) | 0.46 |
| Ours, no BAT anchor | 46% (-38 pts) | 0.71 |
| Ours, BAT 27% ± 5 / Phase 2 GPS | 59% (-26) | 0.64 |
| Ours, BAT 18% ± 8 | 77% (-8) | 0.54 |
| Ours, BAT 18% ± 5, Phase 2 GPS | 78% (-7) | 0.54 |
| Ours, BAT 18% ± 5, broad GPS | 87% (+2) | 0.48 |
| Ours, BAT 18% ± 3, broad GPS | 94% (+10) | 0.42 |

We only match or exceed the community when BAT is pinned at the literature with ± 5 points or tighter and GPS is allowed to be near-cure. The community numbers are self-reported, many LLM-assisted, and some are noise-free "underlying HR" probabilities rather than realised-trial probabilities.

## 7. Assumptions and limits
Several inputs are unverified (follow-up schedule quotes, covariate HRs, life-table hazard, transplant fraction, the van der Maas subset). One onset-delay distribution, no HLA split, no explicit transplant timing beyond a simple rate, unstratified log-rank, boundaries assumed. Reported counts are treated as exact. See the reports for provenance.
