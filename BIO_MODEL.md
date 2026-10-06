# Biology-pinned model: knobs, fixed biology, sources (`regal_bio.py`)

## The two knobs (the only things varied)
| knob | meaning | literature range |
|---|---|---|
| `bat_eff` | relapse-hazard multiplier of active BAT (HMA / venetoclax / LDAC, about 75% of BAT) vs observation; 1.0 = no benefit | QUAZAR oral aza vs placebo RFS HR 0.65; HOVON97 aza vs observation 12-mo DFS 64% vs 42% (HR about 0.5); VIALE-M ven+aza maintenance stopped for futility. Prior N(0.75, 0.15) |
| `gps_eff` | fraction of GPS patients with a durable, cure-like immune effect = P(immune response) x P(durable given response) | immune response: 64% any (9/14), CD8 86% in HLA-A*02 (41% of patients) in Phase 2 CR1; 30% in the earlier mixed AML/MDS pilot; 80% in a REGAL random sample (company PR). A value above about 0.8 is outside the evidence |

## Fixed biology (not fitted to the REGAL counts)
| name | value |
|---|---|
| `Mr` | 0.5843 |
| `kr` | 0.3927 |
| `sigma_f` | 0.5 |
| `c_b` | 0.06 |
| `hr_long` | 0.55 |
| `hr_poor` | 1.6 |
| `hr_mrd` | 1.7 |
| `hr_crp` | 1.3 |
| `hr_age10` | 1.1 |
| `p_long` | 0.45 |
| `p_poor` | 0.3 |
| `p_mrd` | 0.4 |
| `p_crp` | 0.25 |
| `age_mean` | 66.0 |
| `age_sd` | 9.0 |
| `u_mean` | 2.5 |
| `m_p` | 5.5 |
| `hr_pr_age10` | 1.15 |
| `h65` | 0.0013 |
| `gomp` | 0.09 |
| `m_bg` | 1.5 |
| `p_sct` | 0.1 |
| `t_sct` | 8.0 |
| `sct_os36` | 0.5 |
| `p_act` | 0.75 |
| `th_dur` | 0.05 |
| `onset_med` | 3.0 |
| `onset_sd` | 0.4 |
| `gamma` | 1.9 |
| `block` | 4 |

Mr and kr are calibrated so that an observation patient's relapse-free survival from randomisation is about 44% / 32% / 17% at 6 / 12 / 36 months (targets 45 / 27 / 22%).

## Sources and reasoning
| item | source |
|---|---|
| Mr, kr | calibrated so the simulated OBSERVATION arm has relapse-free survival 45% at 6 mo and 27% at 12 mo from randomisation (QUAZAR placebo: median 4.8 mo; CR2 durations 3-14 mo, Leopold/Remarkable-Big; relapse 40/17/2% in years 1-3) |
| hr_long 0.55 | CR1 >18 mo vs <=6 mo: 1-yr OS 57% vs 14% (Breems 2005); Thalhammer 1996 as quoted on Reddit (unverified) |
| hr_poor 1.6, hr_crp 1.3 | assumptions (generic AML prognosis; CRp/CRi carries more early failure) |
| hr_mrd 1.7 | 2-yr relapse 40% vs 24% for MRD+ vs MRD- in CR2 (EBMT, ln(0.60)/ln(0.76)) |
| hr_age10 1.10 | van der Maas validation cohort age 69 vs 58, 4-yr OS 9% vs 16% |
| p_long .45, p_poor .30, p_mrd .40, p_crp .25 | assumptions (stratification factors; only age counts are public: 50 aged 18-64, 66 aged >=65) |
| age N(66, 9) | EU register age counts |
| u_mean 2.5 (truncated at 6) | protocol: consent within 6 months of CR2 |
| m_p 5.5 mo | post-relapse survival 5.3 (Bataller), ven R/R 5.5-6.8, re-treatment 6.8 months |
| h65 .0013/mo, +9%/yr, x1.5 | US life table (approximate, unverified) x excess mortality for AML survivors |
| p_sct 0.10, post-SCT 3-yr OS 50% | QUAZAR placebo 13.7%, oral aza 6.3% transplanted (as quoted on Reddit); transplant leaks in both arms (ITT) |
| p_act .75 | EU protocol allows observation / HMA / venetoclax / LDAC; ~25% observation per community estimate (unverified) |
| th_dur .05, onset 3 mo | durable responders keep 5% of relapse hazard; WT1 CTL peak by week 12 (Hashii 2026), sipuleucel-T delayed separation ~6 mo, 80% of immunotherapy trials show >=3 mo delay |
| gamma 1.9 | enrollment-curve shape implied by 20 by Apr 2022 and 38 by Oct 2022 (Reddit, unverified) with 105 by Nov 2023 (company PR) |
| block 4 | IWRS permuted block size not published; 4 assumed |

## Design implemented (from the REGAL design paper, PMC11760237)
- IWRS 1:1 randomisation **stratified by CR1 duration (>=12 vs <12 mo), cytogenetics (poor vs other), CR2 vs CRp2, MRD (pos vs neg)**: 16 strata, sequential permuted blocks (block size not published; 4 assumed).
- Primary analysis: **Cox model stratified by the same factors, treatment only**, full analysis set, one-sided 0.025; one interim at 60 deaths with Lan-DeMets O'Brien-Fleming; futility rule not described in the paper (the IDMC found GPS exceeded predetermined futility criteria).
- GPS: 6 doses q2w, 6 q4w, 3 q6w (15 doses over year 1), treatment ends at relapse; BAT: observation (incl. hydroxyurea), HMA, venetoclax and/or LDAC, physician may resume HMA/LDAC+venetoclax at reduced dose.
