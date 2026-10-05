import pickle, numpy as np, csv
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, synth_lik as S

data = pickle.load(open("results_v2/synth.pkl", "rb"))
def lit(th):
    ln = np.exp(-0.5 * ((np.log(th["S36"]) - np.log(0.18)) / 0.38) ** 2) / th["S36"]
    return ln * norm.pdf(th["M"], 12.5, 2.5) * norm.pdf(th["k"], 0.85, 0.25)
def gps_s36(th):
    H = lambda t: np.log(2) * (t / th["M"]) ** th["k"]; Hl = H(th["L"])
    unc = np.exp(-np.where(36 > th["L"], Hl + th["theta"] * (H(36.0) - Hl), H(36.0)))
    resp = np.exp(-Hl - th["leak"] * np.clip(36 - th["L"], 0, None))
    return th["c"] * resp + (1 - th["c"]) * unc

rows = []; best = {}
for nm, (th, valid, mu, cov, pst, fu, sdfu) in data.items():
    ll = S.loglik(mu, cov, pst, fu, sdfu, use_stall=True, use_fu=True); ll[~valid] = -np.inf
    th["gps_S36"] = gps_s36(th)
    sd = np.sqrt(np.stack([cov[:, i, i] for i in range(3)], 1))
    z = (S.OBS[None] - mu) / sd
    # global best
    i = int(np.argmax(ll))
    print(f"\n=== {nm}: MLE theta (max synthetic loglik {ll[i]:.2f})")
    print({k: round(float(th[k][i]), 3) for k in ["M", "S36", "k", "c", "theta", "L", "leak", "gamma"]}, "GPS S36", round(float(th["gps_S36"][i]), 3))
    print("  predictive mean", mu[i].round(1), "sd", sd[i].round(2), "z of obs", z[i].round(2), "P(<=79 at Q2)", round(float(pst[i]), 2), "FU med", round(float(fu[i]), 1))
    # posterior under flat and literature BAT prior
    for pn, pw in [("flat", np.ones(len(ll))), ("lit BAT", lit(th))]:
        w = np.exp(ll - ll.max()) * pw; w[~valid] = 0
        ess = w.sum() ** 2 / (w ** 2).sum()
        q = lambda x: r.wq(x, w, [.05, .5, .95]).round(3)
        print(f"  [{pn}] ESS {ess:.0f}  BAT S36 {q(th['S36'])} M {q(th['M'])} k {q(th['k'])}  GPS S36 {q(th['gps_S36'])}  c {q(th['c'])} theta {q(th['theta'])} L {q(th['L'])}")
        print(f"        posterior-mean predictive mu {(w[:,None]*mu).sum(0)/w.sum()}  mean |z| of obs {(w[:,None]*np.abs(z)).sum(0)/w.sum()}")
        rows.append([nm, pn, round(ess), *[f"{x[1]:.3g} [{x[0]:.3g},{x[2]:.3g}]" for x in [r.wq(th['S36'], w, [.05, .5, .95]), r.wq(th['M'], w, [.05, .5, .95]), r.wq(th['gps_S36'], w, [.05, .5, .95]), r.wq(th['c'], w, [.05, .5, .95]), r.wq(th['theta'], w, [.05, .5, .95]), r.wq(th['L'], w, [.05, .5, .95])]]])
    # profile over BAT S36
    print("  profile: best-fit parameters for each BAT 3-yr OS bin (max loglik within bin)")
    prof = []
    for lo, hi in [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.45)]:
        m = (th["S36"] >= lo) & (th["S36"] < hi) & valid
        if not m.any(): continue
        j = np.where(m)[0][np.argmax(ll[m])]
        prof.append((lo, hi, j, ll[j]))
        print(f"   BAT S36 {lo:.2f}-{hi:.2f}: ll {ll[j]:.2f} | S36 {th['S36'][j]:.3f} M {th['M'][j]:.1f} k {th['k'][j]:.2f} | c {th['c'][j]:.2f} theta {th['theta'][j]:.2f} L {th['L'][j]:.1f} leak {th['leak'][j]:.4f} | GPS S36 {th['gps_S36'][j]:.2f} | mu {mu[j].round(1)} sd {sd[j].round(1)}")
    best[nm] = (th, mu, sd, ll, prof, pst)
with open("results_v2/synth_posterior.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "prior", "ESS", "BAT_S36", "BAT_median", "GPS_S36", "c", "theta", "L"]); w.writerows(rows)

# figure: predictive normals at 3 profile points for the no-lag and V3 scenarios
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
cols = ["#2a78c8", "#e08a00", "#c0392b"]
for ri, nm in enumerate(best):
    th, mu, sd, ll, prof, pst = best[nm]
    pick = [p for p in prof if abs(p[0] - 0.15) < 1e-6 or abs(p[0] - 0.20) < 1e-6 or abs(p[0] - 0.25) < 1e-6 or abs(p[0] - 0.30) < 1e-6]
    for ci, d in enumerate(range(3)):
        ax = axes[ri, ci]; x = np.linspace(45, 90, 400)
        for (lo, hi, j, l), c in zip(pick, cols + ["#6b7280"]):
            ax.plot(x, norm.pdf(x, mu[j, d], sd[j, d]), color=c, lw=2, label=f"BAT 3y OS {lo:.2f}-{hi:.2f} (ll {l:.1f})")
        ax.axvline(S.OBS[d], color="k", ls="--"); ax.set_title(f"{nm[:20]} | N at {['10 Dec 24','26 Dec 25','11 May 26'][d]}", fontsize=9)
        ax.set_xlim(S.OBS[d] - 14, S.OBS[d] + 14)
        if ci == 0 and ri == 0: ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig("results_v2/fig_synth_predictive.png", dpi=150); plt.close()

# profile likelihood curve
fig, ax = plt.subplots(figsize=(7, 4))
for nm, c in zip(best, ["#6b7280", "#e08a00"]):
    th, mu, sd, ll, prof, pst = best[nm]
    ax.plot([(p[0] + p[1]) / 2 for p in prof], [p[3] - max(q[3] for q in prof) for q in prof][:len(prof)] if False else [p[3] - max(q[3] for q in prof) for p in prof], "o-", color=c, label=nm[:22])
ax.set_xlabel("BAT 3-yr OS (bin centre)"); ax.set_ylabel("profile log-likelihood (relative)"); ax.legend(); ax.set_title("How well each BAT level can reproduce 60/72/78")
plt.tight_layout(); plt.savefig("results_v2/fig_synth_profile.png", dpi=150); plt.close()
