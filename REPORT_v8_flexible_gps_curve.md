# REGAL v8: flexible GPS survival curve, literature-grounded BAT

Code: `smc_bio.py` (parameters, likelihood, SMC), `sb_report.py`, `regal_bio.py` (simulator). Figure: `results_v2/fig_flexible_gps.png`. Biology and design: see `BIO_MODEL.md` and `REPORT_v7_biology_pinned_two_knobs.md`.

## 1. How the arms are parameterized

**GPS: a mixture-cure curve on the relapse clock, with broad priors (the data move these)**

| parameter | meaning | prior |
|---|---|---|
| `pi` | cure fraction: share of GPS patients with a durable, cure-like effect | Uniform(0, 1) |
| `th_nc` | relapse-hazard multiplier after onset for the non-cured (partial benefit; 1 = none) | Uniform(0.4, 1.2) |
| `th_d` | residual relapse-hazard multiplier of cured patients (0 = true cure, leak above 0) | log-uniform(0.01, 0.3) |
| `L` | median onset delay of the effect, months (per-patient sd 0.4 on the log scale) | lognormal, median 3, sd 0.7 |

Everyone has the same relapse clock before onset; after onset a cured patient's relapse hazard falls to `th_d` times baseline and a non-cured patient's to `th_nc`. Post-relapse death is unchanged, so OS gain is compressed relative to relapse-free gain. The survival curve is an output (a plateau emerges when `pi` is high and `th_d` small).

**BAT: literature-grounded (informative priors plus literature data)**

| parameter | prior | source |
|---|---|---|
| `bat_eff` relapse HR of active BAT | N(0.75, 0.15) | QUAZAR 0.65; HOVON97 about 0.5; VIALE-M no benefit |
| baseline relapse scale | lognormal sd 0.2 around the calibrated observation arm | QUAZAR placebo RFS, 40/17/2 pattern |
| post-relapse median | N(5.5, 0.7) months | 5.3-6.8 months |
| **BAT outcome data**: population BAT 12/24/36-month OS | 0.52 ± 0.06, 0.30 ± 0.06, 0.18 ± 0.05 | van der Maas bridge, entering as a likelihood |

Everything else is fixed biology and the protocol design (stratified permuted-block randomisation, stratified Cox primary analysis); see `BIO_MODEL.md`.

**Likelihood:** expected reported counts at 60/72/78 within 1 event, P(80th not reported by 11 Aug), a soft interim term (P(IDMC continued), averaged over three futility bounds), and the BAT literature data. Tempered SMC, 2,000 particles, 7 free parameters.

## 2. Results

| Run | GPS cure fraction | onset (mo) | BAT 3-yr OS | GPS 3-yr OS | noise-free HR | P(success) protocol (stratified) | unstratified |
|---|---|---|---|---|---|---|---|
| **BAT data as above (main)** | 0.92 (0.71-0.99) | 1.5 | 25% | 54% | 0.47 | **82%** | 91% |
| BAT data twice as loose | 0.88 (0.59-0.99) | 1.8 | 27% | 52% | 0.54 | 71% | 81% |
| BAT data tighter (x0.6) | 0.94 (0.76-1.00) | 1.3 | 23% | 56% | 0.43 | 88% | 95% |
| no BAT outcome data (priors only) | 0.85 (0.50-0.99) | 2.2 | 28% | 51% | 0.57 | 64% | 74% |
| Phase 2-informed cure prior N(0.5, 0.2) | 0.80 (0.59-0.97) | 1.4 | 25% | 53% | 0.49 | 78% | 87% |

Main run: GPS OS 71% / 60% / 54% / 48% at 12 / 24 / 36 / 60 months vs BAT 58% / 35% / 25% / 16%. Partial benefit for the non-cured is modest (HR 0.78, interval 0.44-1.16), residual HR of the cured is about 0.02, BAT active relapse HR 0.84 (0.65-1.03). Expected counts 59.7 / 73.9 / 76.9 against 60 / 72 / 78.

**The likelihood moves GPS more than BAT**: mean shift 0.68 prior SDs with the posterior SD contracting to 45% for the GPS parameters, versus 0.42 and 76% for the BAT parameters.

## 3. Why this differs from the two-knob result (36%)

The two-knob model fixed GPS onset at 3 months, the cured residual hazard at 5%, and gave the non-cured no benefit, and its only BAT freedom was the efficacy knob, which the counts pushed to a strong 0.6. With a free GPS curve and BAT held to the literature curve, the counts are met by a nearly complete, fast, almost-leak-free GPS effect and an ordinary BAT arm (3-year OS 25%).

## 4. Plausibility flags (read before using the headline)

- The cure fraction piles up against its upper bound (5th-95th percentile 0.71-0.99): the data want almost every GPS patient cured. Phase 2 does not support that (3-year OS 47% in CR1, 64% any immune response). The posterior GPS 3-year OS of 54% in CR2 exceeds the Phase 2 CR1 figure.
- The onset posterior (median 1.5 months, lower prior bound 1 month) says the effect would have to start almost immediately; WT1 T-cell responses peak around week 12 and immunotherapy trials usually separate after 3-6 months.
- So the 71-88% depends on a GPS curve more favourable than any prior GPS evidence, which the broad prior permits because we deliberately did not restrict it. A prior centred on Phase 2 lowers it to 78%; fixing the earlier restrictions lowers it to 36%.
- P(success) is also about 8-9 points lower under the protocol's stratified test than an unstratified one.
