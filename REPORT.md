# REGAL (SELLAS GPS vs BAT in AML CR2): a model of how death-reporting lag affects the BAT arm

Built 2026-10-05. Everything here is reproducible from `regal_model.py` → `run_all.py` → `analyze.py` / `figures.py` / `prior_grid.py` / `bat_map.py`. This is a research model, not investment advice. Its inputs include unverified claims from Reddit (see §6), and one of the sources is an author who disclosed a large long position.

## 1. Bottom line

1. **Reporting lag is real but second-order for what we know about BAT.** Under the protocol schedule (survival calls quarterly to week 156, then yearly), the model puts about **1–2 deaths unreported at the 78-event update (60–65% of them BAT)** and **about 2–3 today**. Correcting for that lowers the inferred BAT 3-yr OS by only **about 0.5–1 point** (25.4% → 24.3–24.8%) and BAT median OS by **about 0.5 month** (13.7 → 13.0–13.3).
2. **Where lag does matter is timing.** With protocol-literal lag, the *true* 80th death had already happened by the Q2 PR (11 Aug) with probability **~0.63–0.79**, and by now with **~0.83–0.90**. The *reported* 80th is still pending. The lag explains part of the stall since 78 events (11 May), but only weakly. Bayes factors for lag over no lag are 1.1–1.4, so the stall alone cannot tell the scenarios apart.
3. **Delay is not the dangerous mechanism; permanent non-ascertainment is.** If a death is only discovered late, it is back-dated and the final analysis is fine. If BAT deaths are *never* found before cutoff (patient censored as alive), the hazard ratio is pulled toward 1. In the model each 10% of BAT deaths lost costs about 5 points of P(success) (61% → 56%), and 20% costs about 10 points.
4. **The bigger driver of the answer is the BAT prior and the GPS prior, not lag.** The public event counts pin down roughly how many of the ~127 patients are alive (about 47–49). They do not say which arm. Literature-informed BAT (3-yr OS ≈ 18%) gives P(success) ≈ 60%. A flat BAT prior gives ≈ 40%. Whether the final HR clears 0.636 depends on how many of the survivors are BAT.
5. **BAT estimate from literature + counts + lag:** 3-yr OS from randomization **≈ 24–25% (90% interval 17–32%)**, median OS from randomization **≈ 13 months (10–16)**. The literature alone gives ≈ 13–22% and 8–14 months (§3).

## 2. What I took from Reddit

**Sources.** The "REGAL facts and assumptions" thread is by u/Same-Ad-4215. The user history is u/Confident-Web-7118 ("CW"): 35 submissions and the most recent 100 comments recovered from a PullPush archive (the comment pull was capped at 100, and the archive may be incomplete). The 16.8-month / 18.6% thread is u/Real_Philosopher_831. Reddit blocks the in-app browser, so I used the RSS feed and the archive.

Timeline used (public counts): first patient 8 Feb 2021; 105 enrolled by 29 Nov 2023; 123 by 26 Mar 2024; 127 by 29 Apr 2024; 60 events 10 Dec 2024; 72 events 26 Dec 2025; 78 events 11 May 2026; 80th "approaching" in the 11 Aug 2026 Q2 PR. I confirmed the Nov 2022 PR (events cut 105→80 and 80→60, pooled median "approximately two-fold" longer than expected) and the Q2 PR. I also confirmed that a 23–24 Aug AktienCheck article claiming the 80th was unconfirmed by the company.

**CW's modeling, and where this model agrees or differs**

| CW claim | What I found |
|---|---|
| An 8-week reporting lag barely moves HR. | **Agree for symmetric lag** (S1: 0.4 event gap). His lag is a uniform shift; the protocol's annual cadence after 36 months is arm-differential and larger (S2: 1.9 events at the 78 update; BAT share ≈ 65%). Still modest for HR. |
| Deaths are back-dated to the true date, so lag has no negative effect. | **Right for the final analysis if every death is eventually found.** It misses (a) the 80th-event *trigger* is the discovery date, and (b) deaths that are never found (§1.3). |
| Randomization window inflates BAT median; HR unaffected. | **Partly right.** Left truncation lifts the from-randomization curve, but the relapse→CR2 clock shift pushes the other way. Net effect on 3-yr OS is only ±2 points (§3). |
| BAT 3-yr OS 13–19% biologically; worst case 25–28%; success ≈ 97–99%. | My posterior BAT 3-yr OS is **25% (17–32%)**, i.e. at his worst-case edge, because the slow event accrual forces survivors somewhere. With the same data I get P(success) ≈ 60%, not 97%+ (§5). |
| 80th event is likely "soon". | Reported-80th forecast: median late Nov 2026, 10–90% mid-Oct 2026 to Apr 2027 (§5). |

**Provenance caveats.** The loss-of-contact schedule ("every 3 months by phone to week 156, then yearly") is quoted by CW from the EU protocol. The CTR register page I fetched does not show it; it says OS is assessed "at every contact… continuously". The design paper (PMC11760237) is silent too. Treat the schedule as **plausible, unverified**. CW's 3-yr OS inputs from Kugler (MD Anderson) and the Tsirigotis email (median 16 months in "a recent randomized trial") I could not verify. The email is anecdotal.

## 3. Literature behind the BAT prior

| Source | Setting | Number used | Caveat |
|---|---|---|---|
| **van der Maas / Versluis, Blood Adv 2025;9:3853** (verified) | HOVON-SAKK 2000–2018, 943 relapsed, median age at relapse 58, intensive re-induction; CR2 in 50%; 46% of CR2 had allo-HCT | Untransplanted CR2 (n=60, via author email relayed on Reddit): 1y 52.4%, 2y 33.2%, **3y 18.6% (10.7–32.2)**; clock = first relapse | Pre-venetoclax, 15 years of data, age 58 vs REGAL ~67. The AML18 validation cohort (age 69) has 4-yr OS 9% vs 16% in the age-58 cohort. |
| QUAZAR AML-001 (verified for medians) | CR1, ≥55y, placebo | Placebo mOS 14.8 mo | CR1, not CR2: upper bound for BAT |
| Estey 2013 (acute leukemia, AML+ALL); Breems 2005 (AML ≤60y) (verified) | First relapse | 5-yr OS ≈ 6–12% (intermediate/poor); CR2 reached in 30–50% | Mixed treatment, mostly ≤60y |
| Design paper (verified) | REGAL itself | Design assumed BAT median 8.0 mo; Nov 2022 PR: pooled median ~2× longer than expected | BAT only 8 mo at design |
| Kugler LIT+Ven, Kurosawa, PETHEMA, Tsirigotis | — | As quoted by CW / commenters | **Not verified** |

**VDM bridge** (`vdm_bridge.py`). VDM is measured from relapse; REGAL from randomization (≤6 months after CR2). Two opposite clock effects: relapse→CR2 (d0 = 1.5–2.5 mo) shortens survival measured from CR2; CR2→randomization (d1 = 0–4 mo) lifts it, because patients dying earlier never enter. Applying both plus an age hazard multiplier φ = 1.0–1.2:

| | S36 from randomization | median from randomization |
|---|---|---|
| φ=1.0 | 18–22% | 10.7–14.4 mo |
| φ=1.2 (older population) | 13–16% | 8.4–11.7 mo |
| Using the 3-yr CI bounds (10.7% / 32.2%) | low bound ≈ 8–12%; high bound ≈ 22–42% | median ≈ 9–12 mo (low) to 7–25 mo (high) |

Two corrections to the thread's arguments. A smooth fit through the three VDM Kaplan-Meier points gives a median near 12–13 months, not 16.8, so the 16.8 is probably a KM step effect. And "subtract 1–3 months from VDM" is only half the story, because the truncation lift offsets it. The literature BAT prior I used is **3-yr OS ≈ 18% (lognormal, ×/÷ 1.5), median ≈ 12.5 ± 2.5 mo, Weibull shape ≈ 0.85 ± 0.25**.

## 4. The model (`regal_model.py`)

Patient-level Monte Carlo, 127 patients (64 BAT / 63 GPS), 3 million simulations per scenario, calibrated by kernel-ABC (σ = 1 event) to the *reported* counts 60/72/78 plus "≤79 reported at 11 Aug 2026".

- **Enrollment:** anchored to the disclosed counts; curve shape uncertain (back-loaded).
- **BAT survival:** Weibull from randomization, parameterized by (median, 3-yr OS).
- **GPS survival:** a fraction c of durable responders (BAT hazard until onset lag L, then a low leak hazard); the rest follow BAT with a hazard ratio θ after L. Priors are broad, or informed by Phase 2 (GPS 3-yr OS ≈ 0.50 ± 0.15).
- **Reporting:** `arrival = death + pipeline delay (median 0.75–1 mo)` + discovery delay. BAT patients have no dosing visits, so a death waits for the next scheduled call (uniform over 3 months before month 36 and 12 months after) unless the site learns it sooner (probability q, "real-world discovery"). GPS patients on dosing are known immediately. Optional "silent" deaths (never ascertained before cutoff, patient censored as alive).
- **Scenarios:** S0 no lag; S1 symmetric admin lag; S2 protocol-literal calls (q=0); S3 calls + 50% real-world discovery (base); S4 BAT-adverse (q_BAT = 0.2); S5–S7 S3 plus 5/10/20% of BAT deaths (post-12 months) silent.
- **Analysis:** Cox HR (Newton) and log-rank Z at the cutoff of the reported 80th event. Lan-DeMets O'Brien-Fleming boundaries at 60/80 events (Z = 2.34 interim, 2.01 final; HR boundaries 0.547 / 0.638, matching the design HR 0.636). I condition on "IDMC continued unmodified", i.e. no efficacy stop at the interim. "Swept" = every discovered death is back-dated; "reported-only" = snapshot with unreported deaths censored.

## 5. Results

**Table 1. Scenarios, literature BAT prior, flat GPS prior, conditioned on IA continuing** (`results/summary_table.csv`; figures in `results/`)

| Scenario | BAT 3-yr OS | BAT median (mo) | Unreported at 78 update (BAT) | P(true 80th by 11 Aug) | HR median [90%] | P(success) swept / reported-only | BF vs S0 |
|---|---|---|---|---|---|---|---|
| S0 no lag | 25.4% | 13.7 | 0 | – | 0.59 [0.42–0.90] | 61% / 61% | 1 |
| S1 admin lag | 25.4% | 13.3 | 0.4 (0.2) | 0.30 | 0.60 [0.44–0.91] | 60% / 59% | 1.1 |
| S2 protocol calls | 24.3% | 13.0 | 1.9 (1.2) | 0.79 | 0.60 [0.43–0.92] | 60% / 56% | 1.4 |
| S3 calls + real-world | 24.8% | 13.3 | 1.1 (0.7) | 0.63 | 0.60 [0.44–0.92] | 61% / 58% | 1.2 |
| S4 BAT-adverse | 24.8% | 13.1 | 1.6 (1.2) | 0.75 | 0.61 [0.44–0.93] | 58% / 53% | 1.2 |
| S5 +5% silent | 24.2% | 13.3 | 2.3 (1.8) | 0.84 | 0.61 [0.45–0.96] | 58% / 56% | 1.4 |
| S6 +10% silent | 23.4% | 13.2 | 3.5 (2.8) | 0.92 | 0.61 [0.45–0.94] | 56% / 54% | 1.6 |
| S7 +20% silent | 21.6% | 13.1 | 6.1 (5.2) | 0.99 | 0.63 [0.45–0.97] | 51% / 49% | 2.1 |

**Sweep over BAT real-world discovery probability** (q = 0 → 1, `results/sweep_table.csv`): unreported-at-78 falls from 1.7 to 0.6 events, BAT 3-yr OS posterior moves 24.5% → 24.9%, P(success) ranges 60–64%. Real-world discovery of deaths is the main lever on timing.

**Table 2. Prior sensitivity (S3)** (`results/prior_grid.csv`). P(success):

| BAT prior \ GPS 3-yr OS prior | 0.35 | 0.50 | 0.60 | 0.70 |
|---|---|---|---|---|
| Literature (18%) | 51% | 63% | 71% | 77% |
| Wide (25%) | 39% | 51% | 60% | 67% |
| Flat | 29% | 40% | 48% | 56% |

Posterior BAT 3-yr OS stays at about 25–31% in every cell: the event counts demand survivors.

**Table 3. Outcome by *true* BAT 3-yr OS (S3, flat priors, `results/bat_map.csv`)**

| True BAT 3-yr OS | 15–20% | 20–25% | 25–30% | 30–35% | ≥35% |
|---|---|---|---|---|---|
| Median HR | 0.52 | 0.58 | 0.65 | 0.72 | 0.82 |
| P(success) | 94% | 73% | 47% | 29% | 14% |

Break-even true BAT 3-yr OS is about 27% (about 23% if 10% of BAT deaths are silent). Interim tension: the posterior says the IA would have crossed the efficacy boundary about half the time yet the IDMC continued. CW flags the same oddity.

**Timing forecast** (reported 80th, conditional on not reported by 5 Oct): median late Nov 2026, 10–90% mid-Oct 2026 to Apr 2027; P(reported by 31 Dec 2026) ≈ 0.65–0.70.

**Why not CW's 97%+?** (i) I treat survivor counts as random, not as the deterministic expectation of a fitted curve. (ii) The model's GPS 3-yr OS is ≈ 0.48 versus 0.55–0.66 in CW's fits (his GPS family is solved to hit the counts). (iii) The BAT posterior is ≥ his central estimates. A GPS 3-yr OS of 0.70 ± 0.10 with the literature BAT prior reaches 77%.

## 6. Limitations

- Single Weibull for BAT, no explicit transplant tail; GPS structure fixed; log-rank unstratified; SAP weighting unknown.
- Enrollment shape and reporting pipeline parameters are assumptions; the loss-of-contact schedule is unverified.
- The 5 Oct non-announcement is used only in the forecast, not as a calibration constraint; the 23–24 Aug 80th report is unconfirmed.
- VDM numbers come via an author email relayed on Reddit (n=60, wide CI). The cohort is 2000–2018 and age 58.
- ABC has σ = 1 event tolerance; posteriors for small-mass cells (BAT 3-yr OS < 15%) rest on 40–120 effective samples.
- The Reddit author whose model I replicate declared a large position; independent re-derivation was the point of this exercise.

**What would sharpen it:** the protocol's actual follow-up wording; the Kugler/CR2-untransplanted KM curves; the date of the 80th from the company; any BAT-arm treatment mix; stratified SAP details.

Note: `results/scenarios.pkl` and `sweep.pkl` (~850 MB of simulation draws) were deleted to save space; `python run_all.py 3000000` (~10 min) regenerates them, then run `analyze.py`, `figures.py`, `prior_grid.py`, `bat_map.py`.
