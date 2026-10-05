import pickle, numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import regal_model as r
import analyze_v2 as A

scen = A.scen
BASE = "V3 calls to wk156 + real-world 50%"
def P(res, **kw):
    d = A.summ(res, **kw); return None if d is None else d["P_success"]
base = P(scen[BASE])
items = []
def add(label, val, group):
    if val is not None: items.append((label, val, group))
# IA treatments
for k in ["none (ignore IDMC)", "futility pass: IA HR<=1 & no stop", "Stochasty: GPS IA deaths < BAT", "Thetamancer: IA HR .50-.90 & BAT>=32/60"]:
    add(f"IA: {k}", P(scen[BASE], ia=k), "IDMC interim treatment")
# BAT & GPS priors
add("BAT prior flat (no literature)", P(scen[BASE], prior=lambda x: np.ones_like(x["w"])), "BAT prior")
for mu in (0.35, 0.65):
    add(f"GPS 3-yr OS prior {mu:.2f}±0.10", P(scen[BASE], prior=lambda x, mu=mu: A.w_lit(x) * norm.pdf(x["gps_S36"], mu, 0.10)), "GPS prior")
# structure / mechanism
add("GPS non-responders not worse than BAT (theta<=1)", P(scen[BASE], mask=scen[BASE]["theta"] <= 1.0), "GPS structure")
add("add 13.5-mo median follow-up target", P(scen[BASE], use_fu=True), "calibration")
add("two-stage (RFS + post-relapse) structure", P(scen["TL3 two-stage (lit-compatible), calls+real-world"]), "GPS structure")
# reporting
for nm, lab in [("V2 calls to wk156, no real-world", "reporting: no real-world discovery"), ("V8 calls to wk91 then yearly, rw 50%", "reporting: calls quarterly to wk91 then yearly"),
                ("V9 quarterly throughout, rw 50%", "reporting: quarterly throughout"), ("V10 V3 + LTFU BAT 6%/GPS 2% per yr", "withdrawal: BAT 6%/yr, GPS 2%/yr"),
                ("V11 V3 + LTFU symmetric 3% per yr", "withdrawal: symmetric 3%/yr"), ("V12 V10 + 10% BAT silent deaths", "withdrawal 6/2% + 10% BAT silent deaths")]:
    add(lab, P(scen[nm]), "reporting / censoring")
add("analysis on reported-only snapshot", A.summ(scen[BASE])["P_success_reported_only"], "reporting / censoring")
items.sort(key=lambda x: x[1])
col = {"IDMC interim treatment": "#c0392b", "BAT prior": "#2a78c8", "GPS prior": "#8e44ad", "GPS structure": "#e08a00", "calibration": "#6b7280", "reporting / censoring": "#1b998b"}
fig, ax = plt.subplots(figsize=(10, 6.2))
for i, (lab, v, g) in enumerate(items):
    ax.barh(i, v - base, left=base, color=col[g]); ax.text(v + (0.01 if v >= base else -0.01), i, f"{v:.0%}", va="center", ha="left" if v >= base else "right", fontsize=8)
ax.axvline(base, color="k", lw=1); ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=8)
ax.set_xlim(0.25, 0.9); ax.set_title(f"P(success) when one model choice changes (baseline V3, literature BAT prior = {base:.0%})")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=g) for g, c in col.items()], fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{A.R2}/fig_tornado_v2.png", dpi=150); plt.close()
with open(f"{A.R2}/tornado.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["choice", "P_success", "group"]); w.writerow(["BASELINE V3 lit BAT", round(base, 3), ""])
    for lab, v, g in sorted(items, key=lambda x: -x[1]): w.writerow([lab, round(v, 3), g])
print(open(f"{A.R2}/tornado.csv").read())
