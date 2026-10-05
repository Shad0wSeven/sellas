import pickle, datetime as dt, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r
from analyze import PRIORS, scen, sweep, table, date

R = "results"
C = {"S0 no lag": "#6b7280", "S1 admin lag only (symmetric)": "#2a78c8", "S2 protocol-literal calls": "#c0392b",
     "S3 protocol + real-world (50%)": "#e08a00", "S4 BAT-adverse": "#8e44ad",
     "S5 S3 + 5% BAT silent deaths": "#1b998b", "S6 S3 + 10% BAT silent deaths": "#0b6e63", "S7 S3 + 20% BAT silent deaths": "#064e46"}
SHORT = {"S0": "S0 no lag", "S1": "S1 admin lag (sym.)", "S2": "S2 protocol calls", "S3": "S3 calls + real-world", "S4": "S4 BAT-adverse",
         "S5": "S5 +5% BAT silent", "S6": "S6 +10% BAT silent", "S7": "S7 +20% BAT silent"}
short = lambda s: SHORT[s.split(" ")[0]]
PK = "literature BAT, flat GPS"; pf = PRIORS[PK]
grid_dates = [r.T0 + dt.timedelta(days=float(g) * 30.4375) for g in r.GRID]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. accrual: true vs reported (posterior mean) + anchors
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for sn in ["S0 no lag", "S2 protocol-literal calls", "S3 protocol + real-world (50%)", "S4 BAT-adverse"]:
    res = scen[sn]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
    ax[0].plot(grid_dates, (res["gr"] * w[:, None]).sum(0) / w.sum(), color=C[sn], lw=2, label=f"{short(sn)} reported")
    if sn != "S0 no lag": ax[0].plot(grid_dates, (res["gt"] * w[:, None]).sum(0) / w.sum(), color=C[sn], lw=1.2, ls="--", label=f"{short(sn)} TRUE")
anch = [(dt.date(2024, 12, 10), 60), (dt.date(2025, 12, 26), 72), (dt.date(2026, 5, 11), 78)]
ax[0].scatter([a for a, _ in anch], [b for _, b in anch], color="k", zorder=5, label="public counts")
ax[0].axhline(80, color="k", lw=.6, ls=":"); ax[0].set_xlim(dt.date(2024, 6, 1), dt.date(2027, 3, 1)); ax[0].set_ylim(40, 95)
ax[0].set_title("Deaths: reported vs true (posterior mean)"); ax[0].legend(fontsize=7, ncol=2); ax[0].set_ylabel("cumulative deaths")
for sn in ["S2 protocol-literal calls", "S3 protocol + real-world (50%)", "S4 BAT-adverse"]:
    res = scen[sn]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
    ax[1].plot(grid_dates, ((res["gtB"] - res["grB"]) * w[:, None]).sum(0) / w.sum(), color=C[sn], lw=2, label=f"{short(sn)}: BAT")
    ax[1].plot(grid_dates, (((res["gt"] - res["gtB"]) - (res["gr"] - res["grB"])) * w[:, None]).sum(0) / w.sum(), color=C[sn], lw=1.2, ls=":", label=f"{short(sn)}: GPS")
ax[1].set_xlim(dt.date(2024, 6, 1), dt.date(2027, 3, 1)); ax[1].set_title("Unreported deaths at each date, by arm"); ax[1].legend(fontsize=7)
ax[1].set_ylabel("true minus reported deaths")
plt.tight_layout(); plt.savefig(f"{R}/fig1_reported_vs_true.png", dpi=150); plt.close()

# 2. BAT posterior S36 and median by scenario
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
names = list(scen.keys())
for i, sn in enumerate(names):
    for j, (key, lab) in enumerate([("S36", "BAT 3-yr OS from randomisation"), ("M", "BAT median OS from randomisation (months)")]):
        res = scen[sn]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
        q = r.wq(res[key], w, [.05, .25, .5, .75, .95])
        ax[j].plot([q[0], q[4]], [i, i], color=C[sn], lw=1.2); ax[j].plot([q[1], q[3]], [i, i], color=C[sn], lw=6, solid_capstyle="butt")
        ax[j].plot(q[2], i, "o", color="white", ms=4)
for j, lab in enumerate(["BAT 3-yr OS from randomisation", "BAT median OS from randomisation (months)"]):
    ax[j].set_yticks(range(len(names))); ax[j].set_yticklabels([short(n) for n in names] if j == 0 else []); ax[j].invert_yaxis(); ax[j].set_title(lab)
ax[0].axvspan(.107, .322, color="#2a78c8", alpha=.08); ax[0].text(.108, -0.7, "VDM 95% CI (from relapse)", fontsize=7, color="#2a78c8")
plt.suptitle("Posterior for BAT (literature-informed prior; bar = 50% interval, line = 90%)", y=1.0)
plt.tight_layout(); plt.savefig(f"{R}/fig2_bat_posterior.png", dpi=150); plt.close()

# 3. outcome by scenario: HR and P(success)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for i, sn in enumerate(names):
    res = scen[sn]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
    for off, key, ls in [(-0.12, "hr_final", "-"), (0.12, "hr_final_unswept", ":")]:
        q = r.wq(res[key], w, [.05, .25, .5, .75, .95])
        ax[0].plot([q[0], q[4]], [i + off, i + off], color=C[sn], lw=1); ax[0].plot([q[1], q[3]], [i + off, i + off], color=C[sn], lw=5 if off < 0 else 3, solid_capstyle="butt", alpha=1 if off < 0 else .55)
    ps = table[(PK, sn)]["P_success"]; pu = table[(PK, sn)]["P_success_unswept"]
    ax[1].barh(i - 0.18, ps, 0.34, color=C[sn]); ax[1].barh(i + 0.18, pu, 0.34, color=C[sn], alpha=.45)
    ax[1].text(ps + .01, i - .18, f"{ps:.0%}", va="center", fontsize=8); ax[1].text(pu + .01, i + .18, f"{pu:.0%}", va="center", fontsize=8)
ax[0].axvline(.636, color="k", lw=.8, ls="--"); ax[0].text(.64, -0.6, "design HR 0.636", fontsize=7)
for a in ax: a.set_yticks(range(len(names))); a.invert_yaxis()
ax[0].set_yticklabels([short(n) for n in names]); ax[1].set_yticklabels([]); ax[0].set_title("Final HR (solid: statuses swept; faded: reported-only snapshot)")
ax[1].set_title("P(success | IA continued)  solid: swept, faded: reported-only"); ax[1].set_xlim(0, .85)
plt.tight_layout(); plt.savefig(f"{R}/fig3_outcome.png", dpi=150); plt.close()

# 4. timing: true vs reported 80th
fig, ax = plt.subplots(figsize=(8.5, 4))
bins = np.arange(r.m(dt.date(2026, 3, 1)), r.m(dt.date(2027, 9, 1)), 1.0)
for sn in ["S0 no lag", "S2 protocol-literal calls", "S3 protocol + real-world (50%)"]:
    res = scen[sn]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
    h, _ = np.histogram(res["T80rep"], bins=bins, weights=w); h = h / w.sum()
    ax.step([r.T0 + dt.timedelta(days=float(b) * 30.4375) for b in bins[:-1]], h, where="post", color=C[sn], lw=2, label=f"{short(sn)} reported 80th")
    if sn != "S0 no lag":
        h2, _ = np.histogram(res["T80true"], bins=bins, weights=w); h2 = h2 / w.sum()
        ax.step([r.T0 + dt.timedelta(days=float(b) * 30.4375) for b in bins[:-1]], h2, where="post", color=C[sn], lw=1.2, ls="--", label=f"{short(sn)} TRUE 80th")
ax.axvline(dt.date(2026, 8, 11), color="k", lw=.7, ls=":"); ax.axvline(dt.date(2026, 10, 5), color="k", lw=.7, ls="--")
ax.text(dt.date(2026, 8, 12), ax.get_ylim()[1] * .95, "Q2 PR", fontsize=7); ax.text(dt.date(2026, 10, 6), ax.get_ylim()[1] * .85, "today", fontsize=7)
ax.set_title("When does the 80th death happen vs when is it reported?"); ax.legend(fontsize=7); ax.set_ylabel("posterior probability per month")
plt.tight_layout(); plt.savefig(f"{R}/fig4_timing.png", dpi=150); plt.close()

# 5. sweep
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
qs = sorted(sweep.keys())
vals = {k: [] for k in ["gap78", "S36", "ps", "psu"]}
for q in qs:
    res = sweep[q]; w = res["w"] * pf(res) * (res["z_int"] > -r.Z1)
    vals["gap78"].append(r.wmean(res["true_78"] - res["rep_78"], w)); vals["S36"].append(r.wq(res["S36"], w, [.5])[0])
    vals["ps"].append(r.wmean((res["z_final"] <= -r.ZF).astype(float), w)); vals["psu"].append(r.wmean((res["z_final_unswept"] <= -r.ZF).astype(float), w))
ax[0].plot(qs, vals["gap78"], "o-"); ax[0].set_title("Unreported deaths at the 78-event update"); ax[0].set_xlabel("P(BAT death learned immediately)")
ax[1].plot(qs, vals["S36"], "o-"); ax[1].set_title("Posterior median BAT 3-yr OS"); ax[1].set_xlabel("P(BAT death learned immediately)")
ax[2].plot(qs, vals["ps"], "o-", label="swept"); ax[2].plot(qs, vals["psu"], "s--", label="reported-only"); ax[2].legend(); ax[2].set_title("P(success)"); ax[2].set_xlabel("P(BAT death learned immediately)")
plt.tight_layout(); plt.savefig(f"{R}/fig5_qbat_sweep.png", dpi=150); plt.close()
print("figures done")
