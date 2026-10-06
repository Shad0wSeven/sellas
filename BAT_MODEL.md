# BAT survival inputs and evidence audit

## Recommended parameterization for trial-level sweeps

Use overall survival from randomization as the BAT input, represented by a
two-parameter Weibull curve:

`S_BAT(t) = exp(-log(2) * (t/M)^k)`

`k = log(-log(S36)/log(2)) / log(36/M)`

Here `M` is the BAT median OS in months and `S36` is BAT OS at 36 months. These
are cohort-level inputs: they should describe patients like REGAL's BAT arm,
including any later salvage treatment or transplant after randomization. This
lets the curve be used directly in patient-level Monte Carlo without adding a
separate transplant survival benefit and counting it twice. The earlier
OS-level model already implements this in `regal_model.py`; `bat_sweep.py` now
runs it conditionally over fixed `(M, S36)` pairs and writes weighted results
to `results/bat_weibull_sweep.csv`.

The shape must be positive. When `M < 36`, require `S36 < 0.5`; when `M > 36`,
require `S36 > 0.5`. If `M == 36`, the two inputs do not identify a Weibull
shape. For example, `(M=12.5 months, S36=0.18)` implies `k` about 0.79. A
Weibull is a compact sensitivity model, not a claim that AML hazards truly
follow a Weibull law. It is preferable here to the current bottom-up BAT
construction because it lets the user set the survival quantities of interest
directly.

Run from `regal_model/`:

```bash
python3 bat_sweep.py --n-sim 100000
```

To hold BAT median OS and 3-year OS fixed at a single pair (zero variance on
both inputs), use:

```bash
python3 bat_sweep.py --lock --bat-median 12.5 --bat-os-36 0.18 --n-sim 100000
```

With `--lock`, the script evaluates only that pair. Without it, the existing
`BAT_GRID` scenarios are run; within each scenario the BAT values are still
fixed while the other Monte Carlo inputs vary.

Reporting pipeline delays can be set by arm independently with
`--bat-pipe-med` and `--gps-pipe-med`. For example, the latest sensitivity
run used BAT median pipeline delay 0.75 months and GPS 0.25 months under S1,
S2, and S3. The pipeline delay is lognormal with log-SD 0.6; S2/S3 also retain
their scheduled-call and real-world discovery assumptions. These are model
scenarios, not measured REGAL reporting delays. The result CSVs record both
arm-specific medians.

Edit `BAT_GRID` in `bat_sweep.py` to change the tested medians and 3-year OS
values. The script weights draws against the public 60/72/78 death counts and
the 11-Aug-2026 event-count stall; it reports sequential success (interim
boundary or final boundary) and final success conditional on not crossing the
interim efficacy boundary. It also reports weighted GPS-parameter quantiles
and Monte Carlo Cox-HR quantiles. The legacy model uses randomized 64/63
allocation and a simpler unstratified log-rank analysis, so this sweep is a
sensitivity analysis until the same BAT curve is wired into the
protocol-stratified v8 analysis.

Enrollment is also an explicit sweep input:

```bash
python3 bat_sweep.py --enroll-midpoint 31 --enroll-steepness 0.22
```

The first 105 patients are generated from a logistic density truncated to the
trial start–Nov 29, 2023 interval; the remaining 22 are spread uniformly from
that milestone through Apr 29, 2024. The model sorts dates within each phase,
then randomizes patients to arms. Midpoint is trial-months from Feb 8, 2021;
steepness is per month. A midpoint of 31 and steepness 0.22 yields patient 64
around July 2023 (about trial month 29), consistent with Confident-Web's
corrected estimate. This is an explicit community-model sensitivity input,
not a literature-derived prior or a disclosed patient-level SLS enrollment
curve. The precise late-phase interval and within-phase dates are unknown.
Compare at least the earlier midpoint 28 scenario and a wider midpoint /
steepness grid before treating downstream survival outputs as robust.

**Public-data bounds:** the ClinicalTrials.gov record gives study start Feb 8,
2021 and final enrollment of 127. SELLAS reported exceeding 105 ex-China
patients in Q4 2023 and completing enrollment in March 2024; the April 29,
2024 release reviewed all 127 patients. Thus public information alone only
places patient 64 between first enrollment and the 105-patient milestone.
The June/July 2023 median is curve-dependent. The repository's older March 26
123-patient anchor is not used by the new logistic profile because we did not
locate a primary source for that exact count/date pair.

## What the literature does and does not establish

| Question | Evidence | Modeling consequence |
|---|---|---|
| Which options count as BAT? | REGAL design paper: observation (hydroxyurea allowed), azacitidine or decitabine, venetoclax, and/or low-dose cytarabine; investigator chooses. | Protocol defines the menu, not the frequency of each choice. |
| Is 75% of BAT on active drug verified? | The design paper says the total in each BAT-therapy stratum would remain approximately equal (14–15 patients per stratum). With four mutually exclusive treatment strata and observation as one stratum, that implies roughly 25% observation / 75% active therapy. The paper does not clearly define how combinations are classified, and no realized arm-level regimen counts were located. | `p_act=0.75` has a plausible protocol-based rationale, but remains an inference about planned strata, not a verified observed percentage. It is unnecessary when using marginal BAT OS. |
| What is a reasonable survival anchor? | REGAL's design paper powered the trial assuming BAT median OS 8.0 months. QUAZAR placebo in AML CR1 had median OS 14.8 months and OS 56%/37% at 12/24 months. The HOVON-SAKK relapse cohort reported from first relapse, a different clock and population. | Sweep several `(M, S36)` scenarios; do not treat a CR1 or first-relapse study as a direct REGAL BAT estimate. |
| How many patients later receive transplant? | In QUAZAR AML-001, subsequent HSCT occurred in 13.7% of placebo patients and 6.3% of oral-azacitidine patients. This was CR1, age ≥55, and not REGAL. | These numbers are an external reference only. REGAL excludes patients with immediately planned allo-HCT, but a post-randomization escape fraction for REGAL has not been publicly reported. |
| What is REGAL's age distribution? | Public design/registry materials specify adult eligibility (≥18) and transplant-ineligible/unable-to-undergo status, but no enrolled-patient median or age-bin counts were located. | Existing `N(66,9)` is not verified SLS/REGAL data. Keep age as a scenario input if using age-conditional survival; do not describe it as the observed trial distribution. |

QUAZAR is a useful age-profile comparator, not a matched population:
participants were 55–86 years old with mean age 67.9 (SD 5.66), 28.4% aged
55–64, 60.6% aged 65–74, and about 11% aged 75 or older. REGAL permits adults
younger than 55, and CR2/CRp2 plus transplant ineligibility further changes
the case mix. Therefore QUAZAR's age table cannot identify REGAL ages.

The closest public age analogue by disease state is the small Moffitt CR2
comparison used in SELLAS materials: median age 74 in the 10 GPS patients and
73 in 15 contemporaneous controls. It is a small, nonrandomized historical
comparison, not REGAL's age distribution. For future age-conditional model
checks, 68 and 74 are defensible reference-age scenarios; actual REGAL ages
remain unknown.

The 2025 van der Maas paper's HOVON-SAKK development cohort had median age 58
at relapse; among the 348 patients reaching CR2, 159 (46%) received allo-HCT.
That is a selected, intensive-reinduction CR2 group, unlike REGAL's
transplant-ineligible-at-entry population. The often-cited no-transplant CR2
curve (60 patients; 3-year OS 18.6%) in project notes comes from an
author-communication relayed in an online discussion, not a published table
we could independently verify. It should be a labeled sensitivity scenario,
not a confirmed point estimate.

## Sources

- REGAL design paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11760237/
- REGAL sponsor filing (adult CR2+ and allo-HSCT eligibility): https://www.sec.gov/Archives/edgar/data/1390478/000139047826000003/sls-20251231.htm
- QUAZAR CADTH review, baseline age distribution: https://www.ncbi.nlm.nih.gov/books/NBK595374/
- QUAZAR CADTH review, OS estimates and subsequent-HSCT proportions: https://www.ncbi.nlm.nih.gov/books/NBK595373/
- SELLAS SEC-filed presentation summarizing the Moffitt CR2 cohort age: https://www.sec.gov/Archives/edgar/data/1390478/000139047817000052/sellasgalenamergerslides.htm
- van der Maas et al., Blood Advances 2025: https://doi.org/10.1182/bloodadvances.2025015797
- ClinicalTrials.gov, NCT04229979 (start date and final enrollment): https://clinicaltrials.gov/study/NCT04229979
- SELLAS Q4 2023 enrollment update (105 ex-China and planned backfill): https://ir.sellaslifesciences.com/news/News-Details/2024/SELLAS-Provides-Corporate-Updates-and-Highlights-Key-Upcoming-Milestones/default.aspx
- SELLAS April 29, 2024 release (n=127 and enrollment concluded): https://ir.sellaslifesciences.com/news/News-Details/2024/SELLAS-Life-Sciences-Announces-Positive-Recommendation-of-Independent-Data-Monitoring-Committee-Following-Completion-of-Enrollment-in-REGAL-Phase-3-Study/default.aspx
- Confident-Web-7118 corrected logistic enrollment estimate (community assumption): https://www.reddit.com/r/sellaslifesciences/comments/1tk5ke9/slightly_corrected_staggered_enrollment_makes_the/
