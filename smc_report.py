import pickle, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, smc_fit as m

runs = {f.split("smc_")[1][:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_v2/smc_*.pkl"))}
q = lambda x, w=None: (np.percentile(x, [5, 50, 95]) if w is None else r.wq(x, w, [.05, .5, .95]))
fm = lambda a: f"{a[1]:.3g} [{a[0]:.3g}, {a[2]:.3g}]"

def derived(Xs):
    return dict(BAT_3y_OS=Xs[:, 0], BAT_median=Xs[:, 1], GPS_3y_OS=m.gps_S36(Xs), c_durable=Xs[:, 2], theta_nonresp=Xs[:, 3], onset_L=Xs[:, 4], leak=Xs[:, 5], enroll_gamma=Xs[:, 6], pull_lag_72=Xs[:, 7])

rows = []
for tag, R in runs.items():
    fit, rel, s = R["fit"], R["rel"], R["summ"]
    C = rel["C"]; kern = np.exp(-0.5 * (((C - m.OBS[None]) / 2.0) ** 2).sum(1)); ia = rel["z_ia"] > -r.Z1
    w = kern * ia; Xt = rel["theta"][rel["part"]]
    post_smc = derived(fit["X"]); post_ia = {k: r.wq(v, w, [.05, .5, .95]) for k, v in derived(Xt).items()}
    rows.append(dict(config=tag, BAT3y_calibrated=fm(q(post_smc["BAT_3y_OS"])), BAT3y_after_IA=fm(post_ia["BAT_3y_OS"]), BATmed_after_IA=fm(post_ia["BAT_median"]),
                     GPS3y_after_IA=fm(post_ia["GPS_3y_OS"]), c_after_IA=fm(post_ia["c_durable"]), theta_after_IA=fm(post_ia["theta_nonresp"]), onsetL_after_IA=fm(post_ia["onset_L"]), pull72_after_IA=fm(post_ia["pull_lag_72"]),
                     expected_counts=np.round(s["expected_counts_mean"], 1).tolist(), realised_sd=np.round(s["realised_counts_sd"], 1).tolist(),
                     HR_unconditional=fm(s["HR_unconditional"]), HR_population=fm(s["pop_HR"]), HR_given_counts_and_IA=fm(s["HR_given_counts_and_IA"]),
                     P_success_uncond=round(s["P_success_uncond"], 3), P_IA_continues=round(s["P_IA_continues_uncond"], 3), P_success_given_counts_and_IA=round(s["P_success_given_counts_and_IA"], 3)))
with open("results_v2/smc_summary.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
for rw in rows: print(rw["config"], "|", {k: rw[k] for k in ["BAT3y_after_IA", "BATmed_after_IA", "GPS3y_after_IA", "expected_counts", "HR_unconditional", "HR_population", "HR_given_counts_and_IA", "P_success_given_counts_and_IA"]})

# main run figures
main = runs["V3_tau1.0_agn"]; fit, rel = main["fit"], main["rel"]
# prior sample (same SIR as in smc)
rng = np.random.default_rng(7); X0 = m.LO + (m.HI - m.LO) * rng.random((60000, 8)); lp0 = m.log_prior(X0, False)
w0 = np.exp(lp0 - lp0.max()); Xp = X0[m.systematic(w0, rng)[::40][:1500]]
C = rel["C"]; kern = np.exp(-0.5 * (((C - m.OBS[None]) / 2.0) ** 2).sum(1)); ia = rel["z_ia"] > -r.Z1; w = kern * ia
Xt = rel["theta"][rel["part"]]
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
H = fit["hist"]; st = range(len(H))
for i, (lab, c) in enumerate(zip(["10 Dec 24 (60)", "26 Dec 25 (72)", "11 May 26 (78)"], ["#2a78c8", "#e08a00", "#c0392b"])):
    ax[0, 0].plot(st, [h["mu_mean"][i] for h in H], "o-", color=c, label=lab); ax[0, 0].axhline(m.OBS[i], color=c, ls=":")
ax[0, 0].set_title("SMC: model's expected counts move to the data"); ax[0, 0].set_xlabel("tempering stage (0 = prior)"); ax[0, 0].legend(fontsize=7)
ax[0, 1].plot(st, [h["S36"][1] for h in H], "o-", label="BAT 3-yr OS"); ax[0, 1].plot(st, [h["gps36"][1] for h in H], "s-", label="GPS 3-yr OS")
ax[0, 1].set_title("parameters move (particle medians)"); ax[0, 1].legend(fontsize=7); ax[0, 1].set_xlabel("stage")
ax[0, 2].plot(st, [h["rmse"] for h in H], "o-", color="#6b7280"); ax[0, 2].set_title("RMS gap: expected counts vs 60/72/78 (events)"); ax[0, 2].set_xlabel("stage")
bins = np.linspace(0, 1, 41)
for key, lab, col in [(0, "BAT 3-yr OS", "#2a78c8"), (None, "GPS 3-yr OS", "#8e44ad")]:
    px = Xp[:, 0] if key == 0 else m.gps_S36(Xp); ox = fit["X"][:, 0] if key == 0 else m.gps_S36(fit["X"]); ax[1, 0].hist(px, bins, density=True, alpha=.25, color=col); ax[1, 0].hist(ox, bins, density=True, histtype="step", lw=2, color=col, label=lab)
ax[1, 0].set_title("prior (shaded) vs calibrated posterior (line)"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlim(0, .9)
for i, (lab, c) in enumerate(zip(["N 10 Dec 24", "N 26 Dec 25", "N 11 May 26"], ["#2a78c8", "#e08a00", "#c0392b"])):
    ax[1, 1].hist(C[:, i], bins=np.arange(30, 100), density=True, histtype="step", color=c, lw=2, label=lab); ax[1, 1].axvline(m.OBS[i], color=c, ls="--")
ax[1, 1].set_title("realised reported counts from calibrated model (dashed = observed)"); ax[1, 1].legend(fontsize=7)
hb = np.linspace(0.1, 1.6, 60)
ax[1, 2].hist(rel["hr"], hb, density=True, alpha=.35, color="#6b7280", label="all realised trials")
ax[1, 2].hist(rel["hr"], hb, weights=w, density=True, histtype="step", lw=2, color="#c0392b", label="given counts & interim continued")
ax[1, 2].axvline(0.636, color="k", ls="--"); ax[1, 2].set_title("Cox HR at the reported 80th event"); ax[1, 2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results_v2/fig_smc_main.png", dpi=150); plt.close()
