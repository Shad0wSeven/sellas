# BAT literature-anchored, GPS-flexible inference

Idea: keep BAT tied to the literature as *data* (van der Maas bridge: BAT 12 / 24 / 36-month OS about 0.52 / 0.30 / 0.18), give GPS broad priors, and let the 60/72/78 counts and the interim move GPS far more than BAT.
Code: `PM_PIN='{"bat12":[0.52,0.06],"bat24":[0.30,0.06],"bat36":[0.18,0.05]}'` (soft likelihood on population BAT survival) with `PM_OVERRIDE` widening the GPS priors; analysis `bl_report.py`.

| run | BAT anchor | GPS priors | BAT 3-yr OS (post) | GPS 3-yr OS (post) | GPS durable fraction (p_resp x f_dur) | HR given counts & interim | P(success) |
|---|---|---|---|---|---|---|---|
| BL0 | none | Phase 2-centred | 0.33 | 0.48 | 0.59 (0.39-0.81) | 0.64 | 46% |
| BL5 | S36 18% ±5 | Phase 2-centred | 0.27 | 0.52 | 0.78 (0.62-0.90) | 0.56 | 78% |
| BL6 | S36 27% ±5 | Phase 2-centred | 0.31 | 0.50 | 0.69 (0.50-0.85) | 0.61 | 59% |
| BL2 | S36 18% ±8 | broad | 0.28 | 0.53 | 0.80 (0.59-0.96) | 0.56 | 77% |
| BL1 | S36 18% ±5 | broad | 0.25 | 0.55 | 0.87 (0.69-0.97) | 0.53 | 87% |
| BL4 | S36 18% ±3 | broad | 0.23 | 0.57 | 0.90 (0.79-0.98) | 0.50 | 94% |
| BL3 | S36 27% ±5 | broad | 0.29 | 0.52 | 0.75 (0.54-0.94) | 0.58 | 71% |

Data movement (mean shift in prior SDs, posterior SD / prior SD): with BAT anchored and GPS broad, BAT parameters move 0.6 SD (contraction 0.87) and GPS parameters 1.3 SD (contraction 0.28), i.e. the data now mainly update GPS.

## Reading it

- Conditional on BAT matching the literature, the counts require GPS to be at or near its ceiling: about 70-90% of GPS patients durable (responder and durable), GPS 3-yr OS about 0.52-0.57, onset about 1.5-2.5 months. With Phase 2-centred priors the posterior still lands at responders 0.85 and durable-among-responders 0.94.
- That is well above what Phase 2 shows (3-yr OS 47% in CR1, 64% any immune response), so the model is telling you *what GPS would have to be* if the BAT literature is right: P(success) is high (about 0.71-0.94) only because GPS is forced to near-cure. If BAT is instead anchored at the higher Kugler-type reading (3-yr OS 27%), GPS needs less and P(success) falls to 0.59-0.71 (0.59 with Phase 2-centred GPS priors).
- P(success) therefore depends on the BAT anchor (centre and width) far more than on anything else: 0.59 to 0.94 across anchors.
- The posterior piles up against the upper bounds of `p_resp` and `f_dur`; read those as lower bounds on what is needed, not as estimates.
