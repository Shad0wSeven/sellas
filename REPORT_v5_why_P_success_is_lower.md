# Why our P(success) is lower than the other Reddit models

Experiments: `run_diag.sh`, `run_diag2.sh`, `diag_report.py`; outputs in `results_v2/smc3_X*.pkl` (git-ignored) and the table below.
All runs: lean patient model, tolerance 1 event, soft interim likelihood, lag 0. "P(success)" = share of realised trials (conditioned on the 60/72/78 counts and the interim continuing) whose log-rank test crosses the final boundary (HR about 0.64). "Population HR" is the noise-free Cox HR from the same simulated patients.

**Correction first:** earlier I said the posterior GPS onset delay was about 11 months. That was a misreading; the table already reports months: the median onset is about **2.4 months**. The delay is not the main reason GPS tops out near 48%.

## 1. Experiments

| Run | what changes | counts RMS gap | BAT 3-yr OS | GPS 3-yr OS | population HR | P(pop HR < 0.636) | P(success) |
|---|---|---|---|---|---|---|---|
| X0 base | nothing | 1.59 | 33% | 48% | 0.71 | 23% | **46%** |
| X6 | soft literature pin on BAT 3-yr OS: 19% ± 7 | 1.59 | 31% | 49% | 0.65 | 45% | 57% |
| X1 | pin BAT 3-yr OS 19% ± 4 | 1.61 | 28% | 52% | 0.56 | 82% | **72%** |
| X9 | pin BAT 3-yr OS 19% ± 2 | 1.70 | 24% | 55% | 0.45 | 100% | **92%** |
| X3 | no GPS onset delay | 1.58 | 32% | 49% | 0.63 | 52% | 53% |
| X4 | no delay + pin 19% ± 4 | 1.63 | 27% | 53% | 0.52 | 93% | 78% |
| X5 | no delay + pin 19% ± 4 + GPS as strong as allowed (90% responders, 95% durable) | 1.57 | 24% | 57% | 0.43 | 100% | 92% |

(Wider pins: ± 10 → 52%, ± 15 → 51%; a pin at 25% ± 4 → 59%.)

## 2. What explains the gap

1. **How BAT is anchored.** The 60/72/78 counts fit equally well in every run (RMS gap about 1.6 events in all of them), so the counts do not separate the models. What separates them is how BAT 3-yr OS is constrained. In our base model the literature enters only through mechanistic priors, so the data can drag BAT to 33%, far above the 13–22% VDM-bridge range, and the 33% posterior is *cheaper* in prior probability than forcing GPS above 55%. The Reddit models (CW, Thetamancer, JustThatGuy) fix BAT as an outcome (median and 3-yr OS) and let GPS absorb the rest. That is the same as our pin with a very small SD (X9: 92%).
2. **The conflict is visible but not resolved by pinning.** Even with the pin at ± 2 points, the posterior BAT sits at 24%, not 19%, and the count fit worsens slightly (RMS 1.70). The data and the BAT literature genuinely disagree: about 38% of patients alive at about 41 months of mean follow-up requires either a strong BAT arm or a GPS arm well above Phase 2.
3. **The GPS ceiling.** Our GPS needs an immune response, then durability, and it acts only on relapse. Relapse hazard is front-loaded (a declining hazard fitted at shape 0.54 from CR2), so about a quarter of GPS patients relapse in the first months, before the effect starts; those then follow a post-relapse course. The other models apply a cure fraction directly to OS from the start. Removing the onset delay adds about 7 points (X3: 53%); allowing GPS its maximum adds the rest (X5).
4. **Sampling noise.** The noise-free population HR clears 0.636 more often than the realised trial does (X1: 82% vs 72%; X4: 93% vs 78%), because only 80 deaths are observed. Some posts quote the noise-free number (JustThatGuy 85% underlying vs 74% realised).
5. **Not drivers:** the interim term (42% vs 46% without it), the count tolerance, and reporting lag (BAT lag N(120, 60) days changed nothing).

## 3. How to read it

P(success) is essentially a dial on **how much weight the BAT literature gets versus the pooled counts**: 46% with no outcome anchor, 57% at ± 7 points, 72% at ± 4, 92% at ± 2. The 90%+ models implicitly use the tightest pin and a strong GPS. Which is right depends on whether you trust the van der Maas bridge as a tight estimate of BAT (its confidence interval alone spans about 8–42% at 3 years, so a ± 2 pin overstates what that one study knows) and whether GPS can exceed its Phase 2 results (3-yr OS 47% in CR1).
