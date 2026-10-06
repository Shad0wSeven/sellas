import pickle, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, smc_bio as sb
runs = {f.split("sb_")[1][:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_v2/sb_F*.pkl"))}
tr = lambda n, v: np.exp(v) if n in ("lth_d", "lL", "llam_b") else v
nm = {"pi": "cure fraction", "th_nc": "partial-benefit HR (non-cured)", "lth_d": "residual HR of cured", "lL": "onset median (mo)", "bat_eff": "BAT active relapse HR", "llam_b": "baseline relapse scale", "m_p": "post-relapse median (mo)"}
GPSn = ["pi", "th_nc", "lth_d", "lL"]; BATn = ["bat_eff", "llam_b", "m_p"]
f2 = lambda a: f"{a[1]:.2f} ({a[0]:.2f}-{a[2]:.2f})"
for tag, R in runs.items():
    fit, rel = R["fit"], R["rel"]; sb.PRIOR_MODE = R["prior_mode"]; s = sb.summarize(fit, rel); X = fit["X"]; Xp = sb.sample_prior(4000, np.random.default_rng(0))
    C = rel["C"]; mv = lambda names: (np.mean([abs(np.median(tr(n, X[:, sb.IDX[n]])) - np.median(tr(n, Xp[:, sb.IDX[n]]))) / tr(n, Xp[:, sb.IDX[n]]).std() for n in names]),
                                         np.mean([tr(n, X[:, sb.IDX[n]]).std() / tr(n, Xp[:, sb.IDX[n]]).std() for n in names]))
    sg, cg = mv(GPSn); sbt, cbt = mv(BATn); last = fit["hist"][-1]
    print(f"\n=== {tag}  (tau {fit['tau']}, BAT-data sd x{fit['bat_sd_mult']}, GPS prior {R['prior_mode']})  counts {np.round(last['mu'],1)} rmse {last['rmse']:.2f}")
    print("  GPS curve params:  " + "; ".join(f"{nm[n]} {f2(np.percentile(tr(n, X[:, sb.IDX[n]]), [5, 50, 95]))}" for n in GPSn))
    print("  BAT params:        " + "; ".join(f"{nm[n]} {f2(np.percentile(tr(n, X[:, sb.IDX[n]]), [5, 50, 95]))}" for n in BATn))
    print(f"  data movement (mean |shift| in prior SDs / posterior-SD contraction): GPS {sg:.2f} / {cg:.2f}   BAT {sbt:.2f} / {cbt:.2f}")
    ig = {12: 2, 24: 4, 36: 6, 60: 10}
    print("  survival  BAT: " + "  ".join(f"{t}m {s['BAT'][i][1]:.2f}" for t, i in ig.items()) + "   |  GPS: " + "  ".join(f"{t}m {s['GPS'][i][1]:.2f}" for t, i in ig.items()))
    print(f"  noise-free HR {f2(s['popHR'])}  realised HR given counts&IA {f2(s['HR'])}  P(success) protocol {s['P_succ']:.3f}  unstratified {s['P_succ_unstrat']:.3f}  (ESS {s['ESS']:.0f})")
# figure for the main run
R = runs["F1_main"]; fit, rel = R["fit"], R["rel"]; s = sb.summarize(fit, rel); sb.PRIOR_MODE = "broad"; Xp = sb.sample_prior(4000, np.random.default_rng(0)); X = fit["X"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
t = sb.TGRID
for key, c, lab in [("BAT", "#2a78c8", "BAT"), ("GPS", "#c0392b", "GPS")]:
    arr = np.array([s[key][i] for i in range(len(t))]); ax[0].plot(t, arr[:, 1], color=c, lw=2, label=lab); ax[0].fill_between(t, arr[:, 0], arr[:, 2], color=c, alpha=.2)
ax[0].errorbar([12, 24, 36], [.52, .30, .18], yerr=[.06, .06, .05], fmt="ks", capsize=3, label="van der Maas bridge (BAT)"); ax[0].set_xlabel("months from randomisation"); ax[0].set_ylabel("overall survival"); ax[0].set_title("posterior OS curves (population, 90% bands)"); ax[0].legend(fontsize=8)
for k, (n, a) in enumerate([("pi", ax[1]), ("lL", ax[2])]):
    pr, po = tr(n, Xp[:, sb.IDX[n]]), tr(n, X[:, sb.IDX[n]]); bins = np.linspace(min(pr.min(), po.min()), max(pr.max(), po.max()), 40)
    a.hist(pr, bins, density=True, alpha=.3, color="#6b7280", label="prior"); a.hist(po, bins, density=True, histtype="step", lw=2, color="#c0392b", label="calibrated"); a.set_title(nm[n]); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results_v2/fig_flexible_gps.png", dpi=150); plt.close()
