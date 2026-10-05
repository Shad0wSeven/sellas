import csv, datetime as dt, numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm, multivariate_normal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, regal_model_v2 as v, synth_lik as S

V3 = v.SCENARIOS_V2["V3 calls to wk156 + real-world 50%"]
SC = {"S0 no lag": dict(lag=False), "S0 + holiday lag at 72": dict(lag=False, holiday=True), "V3 calls to wk156 + real-world 50%": V3, "V3 + holiday lag at 72": dict(V3, holiday=True)}
LN2 = np.log(2)
def mk_params(S36, kshape, x, n):
    c, th, L, leak, g, kshape = x
    M = 36.0 / (-np.log(S36) / LN2) ** (1 / kshape)
    p = dict(M=M, S36=S36, k=kshape, c=c, theta=th, L=L, leak=leak, gamma=g)
    return {k: np.full(n, float(vv)) for k, vv in p.items()}, M

def counts(p, scen, seed, with_extra=False):
    n = len(p["M"]); rng = np.random.default_rng(seed)
    with np.errstate(all="ignore"):
        a, arm, T, resp = S.sim_from_params(p, rng); d = a + T
        arrive, gap, W = v.add_reporting(n, rng, scen, a, arm, T, d, resp)
        C = np.stack([(arrive <= t).sum(1) for t in S.DATES], 1).astype(float)
    return (C, a, arm, T, d, arrive, gap, W) if with_extra else C

def objective(x, S36, kshape, scen, R=300):
    p, _ = mk_params(S36, kshape, x, R)
    C = counts(p, scen, seed=123)                                  # common random numbers => smooth in x
    mu = C.mean(0)
    stall = (np.stack([(0 * C[:, 0])], 1)[:, 0])                   # placeholder (stall handled after)
    # mild regularisation toward prior centres so the 5 free parameters are not wild
    reg = 0.02 * (((x[0] - 0.35) / 0.2) ** 2 + ((x[1] - 0.6) / 0.3) ** 2 + ((x[2] - 3) / 2) ** 2)
    return float(((mu - S.OBS) ** 2).sum() + reg)

BOUNDS = [(0.0, 0.65), (0.3, 1.3), (1.0, 8.0), (0.002, 0.009), (1.4, 2.8), (0.6, 1.4)]
rows = []; store = []
R_REF = 4000
for nm, scen in SC.items():
    for kshape in (None,):
        for S36 in (0.12, 0.16, 0.20, 0.25, 0.30, 0.35):
            res = differential_evolution(objective, BOUNDS, args=(S36, 0.85, scen), seed=3, maxiter=45, popsize=10, tol=1e-3, polish=False)
            x = res.x; p, M = mk_params(S36, kshape, x, R_REF); kshape = x[5]
            C, a, arm, T, d, arrive, gap, W = counts(p, scen, seed=777, with_extra=True)
            mu, cov = C.mean(0), np.cov(C.T)
            llr = multivariate_normal(mu, cov + 0.5 * np.eye(3)).logpdf(S.OBS)
            stall = float(((arrive <= r.D["q2"]).sum(1) <= 79).mean())
            T80 = np.partition(arrive, 79, axis=1)[:, 79]
            cut = np.minimum(T80, r.m(dt.date(2030, 1, 1)))
            (zc, lc), _ = v.analysis2(cut, a, arm, T, d, arrive, gap, W, True)
            (zi, li), _ = v.analysis2(np.full(R_REF, r.D["e60"]), a, arm, T, d, arrive, gap, W, True)
            cont = zi > -r.Z1; succ = zc <= -r.ZF
            # GPS S36 analytic
            H = lambda t: LN2 * (t / M) ** x[5]; Hl = H(x[2])
            unc = np.exp(-(Hl + x[1] * (H(36.0) - Hl))); rs = np.exp(-Hl - x[3] * (36 - x[2]))
            g36 = x[0] * rs + (1 - x[0]) * unc
            row = dict(scenario=nm, BAT_S36=S36, BAT_M=round(M, 1), k=round(float(kshape), 2), c=round(x[0], 2), theta=round(x[1], 2), L_months=round(x[2], 1), leak_per_mo=round(x[3], 4), gamma=round(x[4], 2),
                       GPS_S36=round(float(g36), 2), mu60=round(mu[0], 1), mu72=round(mu[1], 1), mu78=round(mu[2], 1),
                       sd60=round(float(np.sqrt(cov[0, 0])), 1), sd72=round(float(np.sqrt(cov[1, 1])), 1), sd78=round(float(np.sqrt(cov[2, 2])), 1),
                       loglik_counts=round(float(llr), 2), P_stall_80th_not_by_Aug11=round(stall, 2), HR_median=round(float(np.median(np.exp(lc))), 3),
                       P_IA_continues=round(float(cont.mean()), 2), P_success_all=round(float(succ.mean()), 3), P_success_given_IA_cont=round(float(succ[cont].mean()) if cont.any() else float("nan"), 3))
            rows.append(row); store.append((nm, S36, C, row)); print({k: row[k] for k in ["scenario", "BAT_S36", "BAT_M", "c", "theta", "L_months", "leak_per_mo", "gamma", "GPS_S36", "mu60", "mu72", "mu78", "sd78", "loglik_counts", "P_stall_80th_not_by_Aug11", "HR_median", "P_success_given_IA_cont"]}, flush=True)
with open("results_v2/peak_parameter_sets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
sel = [s for s in store if s[0] == "V3 + holiday lag at 72" and s[1] in (0.16, 0.25, 0.35)]
for ri, (nm, S36, C, row) in enumerate(sel):
    for ci in range(3):
        ax = axes[ri, ci]; o = S.OBS[ci]; x = np.arange(o - 18, o + 18)
        ax.hist(C[:, ci], bins=np.arange(o - 18.5, o + 18.5), density=True, color="#9ec5ee", edgecolor="white")
        ax.plot(x, norm.pdf(x, C[:, ci].mean(), C[:, ci].std()), color="#c0392b", lw=2); ax.axvline(o, color="k", ls="--")
        ax.set_title(f"BAT 3y OS {S36:.2f}: N at {['10 Dec 24','26 Dec 25','11 May 26'][ci]}  mean {C[:, ci].mean():.1f}, sd {C[:, ci].std():.1f}", fontsize=8)
plt.tight_layout(); plt.savefig("results_v2/fig_peak_predictive.png", dpi=150); plt.close()
