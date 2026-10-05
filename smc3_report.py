import pickle, glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, patient_model as pm

runs = {f.split("smc3_")[1][:-4]: pickle.load(open(f, "rb")) for f in sorted(glob.glob("results_v2/smc3_*_*.pkl")) if "test" not in f}
rng = np.random.default_rng(0); Xp = pm.sample_prior(3000, rng)
OBS = np.array([60., 72., 78.]); q = lambda x, w=None: (np.percentile(x, [5, 50, 95]) if w is None else r.wq(x, w, [.05, .5, .95]))
fm = lambda a, d=3: f"{a[1]:.{d}g} [{a[0]:.{d}g}, {a[2]:.{d}g}]"
disp = {"Mr": "median relapse-free time, reference pt (mo)", "kr": "relapse hazard shape", "c_b": "never-relapse fraction", "b_long": "log HR long CR1", "b_poor": "log HR poor cytogenetics",
        "b_mrd": "log HR MRD+", "b_age": "log HR per 10 y age", "sig_f": "frailty SD", "p_long": "share long CR1", "p_poor": "share poor cytogenetics", "p_mrd": "share MRD+",
        "u_mean": "CR2->randomisation delay (mo)", "m_p": "post-relapse median (mo)", "b_prage": "log HR post-relapse per 10 y", "m_bg": "background mortality multiplier", "p_sct": "transplant fraction (ITT)",
        "p_act": "BAT on active therapy", "th_B": "relapse HR, BAT active vs observation", "p_resp": "GPS immune-responder fraction", "f_dur": "fraction of responders with durable (cure-like) effect", "lth_r": "log relapse HR, GPS responders after onset",
        "th_nr": "relapse HR, GPS non-responders vs observation", "lL": "log onset delay (median mo = exp)", "gamma": "enrollment shape", "z_b": "interim efficacy z bound", "h_f": "interim futility HR bound"}

def trial_weights(rel, sigma=2.0):
    C = rel["C"]; kern = np.exp(-0.5 * (((C - OBS[None]) / sigma) ** 2).sum(1)); return kern, kern * rel["ia_cont"]

summ = {}
for tag, R in runs.items():
    fit, rel = R["fit"], R["rel"]; X = fit["X"]; kern, w = trial_weights(rel)
    succ = rel["z"] <= -r.ZF
    # posterior over particles after trial-level conditioning
    pw = np.bincount(rel["part"], weights=w, minlength=len(rel["theta"])); pop_idx = np.arange(len(rel["theta"]))
    Th = rel["theta"]
    d = dict(
        tau=fit["tau"], w_ia=fit["w_ia"], expected_counts=fit["ev"][0].mean(0).round(1), rmse=float(np.sqrt(((OBS[None] - fit["ev"][0]) ** 2).mean())),
        realised_sd=rel["C"].std(0).round(1), P_IA_cont_particles=float(fit["ev"][3].mean()), P_eff_stop=float(fit["ev"][4].mean()), P_fut_stop=float(fit["ev"][5].mean()),
        BAT_S12=q(rel["pop_bat_S12"], pw), BAT_S36=q(rel["pop_bat_S36"], pw), BAT_med=q(rel["pop_bat_med"], pw), GPS_S12=q(rel["pop_gps_S12"], pw), GPS_S36=q(rel["pop_gps_S36"], pw), GPS_med=q(rel["pop_gps_med"], pw),
        HR_all=q(rel["hr"]), P_succ_all=float(succ.mean()), P_IA_cont_trials=float(rel["ia_cont"].mean()),
        HR_cond=r.wq(rel["hr"], w, [.05, .5, .95]), P_succ_cond=float((w * succ).sum() / w.sum()), P_HRlt636_cond=float((w * (rel["hr"] < .636)).sum() / w.sum()),
        HR_cond_counts_only=r.wq(rel["hr"], kern, [.05, .5, .95]), P_succ_cond_counts_only=float((kern * succ).sum() / kern.sum()),
        ESS_w=float(w.sum() ** 2 / (w ** 2).sum()), P_stall=float(rel["stall"].mean()))
    summ[tag] = (d, Th, pw)
with open("results_v2/smc3_summary.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["run", "tau", "w_ia", "expected counts", "rmse", "P(IA continue) particles", "P(eff stop)", "P(fut stop)", "BAT 3y OS", "BAT median", "GPS 3y OS", "GPS median", "HR (all trials)", "HR (given counts & IA)", "P(success | counts & IA)", "P(success all)", "ESS"])
    for tag, (d, Th, pw) in summ.items():
        wr.writerow([tag, d["tau"], d["w_ia"], list(d["expected_counts"]), round(d["rmse"], 2), round(d["P_IA_cont_particles"], 2), round(d["P_eff_stop"], 3), round(d["P_fut_stop"], 3), fm(d["BAT_S36"]), fm(d["BAT_med"], 3), fm(d["GPS_S36"]), fm(d["GPS_med"], 3), fm(d["HR_all"]), fm(d["HR_cond"]), round(d["P_succ_cond"], 3), round(d["P_succ_all"], 3), round(d["ESS_w"])])
for tag, (d, Th, pw) in summ.items():
    print(f"\n=== {tag}: tau {d['tau']} w_ia {d['w_ia']} | expected counts {d['expected_counts']} (rmse {d['rmse']:.2f}) realised sd {d['realised_sd']}")
    print(f"  IA: P(continue) {d['P_IA_cont_particles']:.2f}  P(eff stop) {d['P_eff_stop']:.3f}  P(fut stop) {d['P_fut_stop']:.3f}")
    print(f"  BAT: S12 {fm(d['BAT_S12'])}  S36 {fm(d['BAT_S36'])}  median {fm(d['BAT_med'],3)}   | GPS: S12 {fm(d['GPS_S12'])}  S36 {fm(d['GPS_S36'])}  median {fm(d['GPS_med'],3)}")
    print(f"  HR all trials {fm(d['HR_all'])}  P(succ) {d['P_succ_all']:.3f} | given counts & IA: HR {fm(d['HR_cond'])}  P(HR<.636) {d['P_HRlt636_cond']:.3f}  P(success) {d['P_succ_cond']:.3f} (ESS {d['ESS_w']:.0f}) | given counts only: HR {fm(d['HR_cond_counts_only'])} P(succ) {d['P_succ_cond_counts_only']:.3f}")

# parameter table for run A
import os
KEY = os.environ.get("PM_REPORT_RUN", "L0_lean")
d, Th, pw = summ[KEY]; fit = runs[KEY]["fit"]
rows = []
for j, n in enumerate(pm.NAMES):
    pr = q(Xp[:, j]); po = q(fit["X"][:, j]); pa = r.wq(Th[:, j], pw + 1e-12, [.05, .5, .95])
    tr = (lambda v_: np.exp(v_)) if n in ("lth_r", "lL") else (lambda v_: v_)
    rows.append([n, disp[n], fm(tr(pr)), fm(tr(po)), fm(tr(pa)), round(float((po[1] - pr[1]) / (pr[2] - pr[0]) * 3.29), 2)])
with open("results_v2/smc3_params.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["param", "meaning", "prior median [5,95]", "calibrated posterior", "after trial-level conditioning (counts+IA)", "shift (prior SDs)"]); wr.writerows(rows)
print(f"\nPARAMETERS (run {KEY}) — prior vs posterior")
for rw in rows: print(f"  {rw[0]:8s} {rw[1][:46]:46s} prior {rw[2]:24s} post {rw[4]:24s} shift {rw[5]:+.2f}")
pickle.dump({k: v for k, v in summ.items()}, open("results_v2/smc3_summ.pkl", "wb"))
