import pickle, csv, numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, regal_model_v2 as v, synth_lik as S

data = pickle.load(open("results_v2/synth.pkl", "rb"))
SC = {"S0 no lag": dict(lag=False), "V3 calls to wk156 + real-world 50%": v.SCENARIOS_V2["V3 calls to wk156 + real-world 50%"]}
BINS = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.38)]
R_REF = 3000
rows = []; store = {}
for nm, (th, valid, mu, cov, pst, fu, sdfu) in data.items():
    ll = S.loglik(mu, cov, pst, use_stall=False, use_fu=False)       # counts-only synthetic likelihood
    near = (np.abs(S.OBS[None] - mu).max(1) <= 1.0) & valid & (pst >= 0.15)
    print(f"\n=== {nm}: {near.sum()} of {len(ll)} parameter sets put all three predictive means within 1 event of 60/72/78")
    for lo, hi in BINS:
        m = near & (th["S36"] >= lo) & (th["S36"] < hi)
        if not m.any(): print(f"  BAT S36 {lo:.2f}-{hi:.2f}: none"); continue
        j = np.where(m)[0][np.argmax(ll[m])]
        p = {k: np.full(R_REF, th[k][j]) for k in th}
        rng = np.random.default_rng(900 + j)
        with np.errstate(all="ignore"):
            a, arm, T, resp = S.sim_from_params(p, rng); d = a + T
            arrive, gap, W = v.add_reporting(R_REF, rng, SC[nm], a, arm, T, d, resp)
            C = np.stack([(arrive <= t).sum(1) for t in S.DATES], 1).astype(float)
            mu2, cov2 = C.mean(0), np.cov(C.T)
            mv = multivariate_normal(mu2, cov2 + 0.5 * np.eye(3))
            llr = mv.logpdf(S.OBS)
            stall = ((arrive <= r.D["q2"]).sum(1) <= 79).mean()
            T80 = np.partition(arrive, 79, axis=1)[:, 79]
            cut = np.minimum(T80, r.m(__import__("datetime").date(2030, 1, 1)))
            (zc, lc), _ = v.analysis2(cut, a, arm, T, d, arrive, gap, W, True)
            (zi, li), _ = v.analysis2(np.full(R_REF, r.D["e60"]), a, arm, T, d, arrive, gap, W, True)
        succ = (zc <= -r.ZF); cont = zi > -r.Z1
        pct = [float((C[:, i] <= S.OBS[i]).mean()) for i in range(3)]
        row = dict(scenario=nm, BAT_S36=round(float(th["S36"][j]), 3), BAT_M=round(float(th["M"][j]), 1), k=round(float(th["k"][j]), 2),
                   c=round(float(th["c"][j]), 2), theta=round(float(th["theta"][j]), 2), L=round(float(th["L"][j]), 1), leak=round(float(th["leak"][j]), 4), gamma=round(float(th["gamma"][j]), 2),
                   mu60=round(mu2[0], 1), mu72=round(mu2[1], 1), mu78=round(mu2[2], 1), sd60=round(np.sqrt(cov2[0, 0]), 1), sd72=round(np.sqrt(cov2[1, 1]), 1), sd78=round(np.sqrt(cov2[2, 2]), 1),
                   corr_60_78=round(cov2[0, 2] / np.sqrt(cov2[0, 0] * cov2[2, 2]), 2), loglik_counts=round(float(llr), 2),
                   pctile_obs=[round(x, 2) for x in pct], P_stall=round(float(stall), 2),
                   HR_med=round(float(np.median(np.exp(lc[cont]))), 3), P_success_all=round(float(succ.mean()), 3), P_IA_continues=round(float(cont.mean()), 2), P_success_given_IA_cont=round(float(succ[cont].mean()), 3))
        rows.append(row); store[(nm, lo)] = (C, row)
        print(" ", {k: row[k] for k in ["BAT_S36", "BAT_M", "k", "c", "theta", "L", "leak", "gamma"]}, "| mu", [row["mu60"], row["mu72"], row["mu78"]], "sd", [row["sd60"], row["sd72"], row["sd78"]], "ll", row["loglik_counts"], "pctile", row["pctile_obs"], "stall", row["P_stall"], "| HR", row["HR_med"], "Psucc|IA", row["P_success_given_IA_cont"])
with open("results_v2/peak_parameter_sets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# figure: predictive pmf vs normal for main scenario, 3 BAT levels
nm = "V3 calls to wk156 + real-world 50%"
picks = [k for k in store if k[0] == nm][1:4]
fig, axes = plt.subplots(len(picks), 3, figsize=(12, 3 * len(picks)))
for ri, key in enumerate(picks):
    C, row = store[key]
    for ci in range(3):
        ax = axes[ri, ci]; x = np.arange(S.OBS[ci] - 18, S.OBS[ci] + 18)
        ax.hist(C[:, ci], bins=np.arange(S.OBS[ci] - 18.5, S.OBS[ci] + 18.5), density=True, color="#9ec5ee", edgecolor="white")
        mu, sd = C[:, ci].mean(), C[:, ci].std()
        ax.plot(x, norm.pdf(x, mu, sd), color="#c0392b", lw=2); ax.axvline(S.OBS[ci], color="k", ls="--")
        ax.set_title(f"BAT 3y OS {row['BAT_S36']:.2f}: N at {['Dec 24','Dec 25','May 26'][ci]}  mu {mu:.1f} sd {sd:.1f}", fontsize=8)
plt.tight_layout(); plt.savefig("results_v2/fig_peak_predictive.png", dpi=150); plt.close()
