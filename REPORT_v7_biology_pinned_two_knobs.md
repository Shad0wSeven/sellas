# REGAL v7: protocol-faithful randomisation and analysis, biology pinned from papers, two knobs

Code: `regal_bio.py` (model), `bio_scan.py`, `bio_final.py`, `bio_plot.py`, `bio_lib.py`, `bio_scenarios*.py`, `rand_check.py`. Knobs, fixed biology and sources: `BIO_MODEL.md`. Figure: `results_v2/fig_bio_two_knobs.png`.

## 1. Are we doing the right randomisation? Not exactly. Now fixed.

Verified from the design paper: 1:1 IWRS randomisation **stratified on four factors** (CR1 duration, cytogenetics, **CR2 vs CRp2**, MRD) and a primary **Cox model stratified by the same four factors**. We had randomised on three factors (no CRp2), by alternation after sorting, independent of enrollment order, and analysed with an unstratified test. v7 uses 16 strata and sequential permuted blocks in enrollment order, and the stratified Cox/log-rank test.

Checks (1,500 simulated trials): GPS 63.4 patients per trial (sd 1.7); within-stratum imbalance 0.7 patients; GPS share 0.50 in both the first and second half of enrollment; covariates balanced across arms.

Effect on the answer (fixed knobs): block size (2, 4, 6), simple randomisation or the earlier approximation change P(success) by at most 4 points. **The analysis does matter**: the protocol's stratified test is about 7-8 points less likely to succeed than an unstratified test (26.5% vs 34.1%), because 16 strata on 127 patients leave small strata. The SAP may collapse small strata (not stated), so both are reported. Earlier models used the unstratified test and so slightly overstated P(success).

## 2. Biology of steps 4, 5, 6 and what was read

**4. Relapse, then death.** Relapse hazard declines steeply: 40% / 17% / 2% of patients relapse in years 1-3 in CR1 (cumulative-hazard shape about 0.3-0.45); CR2 durations are 3-14 months; QUAZAR placebo RFS is 45% at 6 and 27% at 12 months (median 4.8). Deaths in remission are rare; post-relapse survival is 5.3-6.8 months (Bataller; venetoclax re-treatment). So the model has a declining Weibull relapse clock from CR2 with patient factors, then exponential post-relapse survival, plus background mortality and transplant leakage. The relapse scale and shape are calibrated to the observation arm's RFS (reference values: median 0.58 and shape 0.39 on the reference-patient scale, giving RFS 44/32/17% at 6/12/36 months).

**5. BAT.** Protocol: observation, HMA, venetoclax and/or LDAC, resumable at reduced dose. Evidence for maintenance benefit: QUAZAR (aza vs placebo, CR1): RFS 10.2 vs 4.8 months, HR 0.65; HOVON97: 12-month DFS 64% vs 42%; VIALE-M stopped for futility. So BAT = a relapse-hazard multiplier (`bat_eff`) on the ~75% on active therapy.

**6. GPS.** Phase 2 CR1 (22 patients, median age 64): immune response in 9/14 (64%), CD8 in 6/7 HLA-A*02 patients (41% of patients carry it), CD4 in 4/9; DFS 16.9 months; 3-year OS 47%. An earlier mixed AML/MDS pilot found T-cell responses in 3/10 and OS 7.4 months. Hashii (pediatric, post-transplant): CTL peak at week 12, responders 91% vs 40% at 3 years. OCV-501 (randomised, HLA class II peptide) failed (DFS HR 0.93). Immunotherapy trials often show 3-6 months of delay before separation. So GPS = immune response, then durable or not; durable patients get a 95% relapse-hazard reduction after about 3 months (median onset), everyone else behaves like observation. `gps_eff` = durable fraction.

## 3. Results (relapse baseline from the 40/17/2 pattern; counts tolerance 1 event; interim in the likelihood)

BAT arm predicted from the biology alone (knob 0.75, no REGAL data): 12 / 24 / 36-month OS 0.61 / 0.38 / 0.27, median 16.4 months, slightly above the van der Maas bridge (0.47-0.55 / 0.25-0.34 / 0.13-0.22; 8.4-14.4 months).

| GPS-knob prior | bat_eff | gps_eff | BAT 3-yr OS | GPS 3-yr OS | HR (noise-free) | P(success) protocol (stratified) | unstratified |
|---|---|---|---|---|---|---|---|
| flat | 0.60 (0.45-0.65) | 0.95 (0.80-1.00) | 31% | 48% | 0.69 | **36%** | 45% |
| Phase 2-centred | 0.55 (0.40-0.65) | 0.90 (0.71-1.00) | 32% | 46% | 0.74 | **27%** | 36% |

Relaxing the count tolerance to 5.4 events (the actual sampling spread of the counts) leaves the knobs unchanged (BAT 0.65, GPS 0.95) and gives 42% (stratified) / 51% (unstratified).

Across 16 biology variants (relapse tail, post-relapse median, transplant fraction, frailty, enrollment shape, observation share) the knobs always land at bat_eff 0.40-0.65 and gps_eff 0.95-1.00, with P(success) 0.51-0.77 (older kernel-based table in `results_v2/bio_scenarios*.pkl`). Adding reporting lag **worsens** the fit (log-evidence -17.6 with no lag, -21.3 at BAT lag 60 days, -26.8 at 120, -46 at 240, -69 at 360).

## 4. What it means
- With biology fixed from the literature, the 60/72/78 counts can only be reproduced if BAT maintenance is stronger than QUAZAR (relapse HR about 0.5-0.65 vs 0.65, closer to HOVON97) **and** nearly every GPS patient is durable (0.8-1.0, above the 64-80% immune-response rates). Neither knob is comfortable; the two trade off along a thin ridge (stronger BAT needs less GPS).
- So the biology we pinned cannot by itself produce enough long survivors; either the true BAT arm or GPS is better than the evidence behind our fixed values, or something not modelled (later enrollment, more transplant, a heavier relapse-free tail) supplies the survivors. Reporting lag is not the explanation (it makes the fit worse).
- P(success) is about 27-45% under the protocol's stratified analysis for the central biology, rising when GPS is allowed to be near-cure and BAT strong, and falling about 8 points relative to an unstratified test.
