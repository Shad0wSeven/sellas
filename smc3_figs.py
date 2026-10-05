import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, patient_model as pm
OBS = np.array([60., 72., 78.])
A = pickle.load(open("results_v2/smc3_A_tau1_softIA.pkl", "rb")); G = pickle.load(open("results_v2/smc3_G_strongGPS_resp80.pkl", "rb"))
Xp = pm.sample_prior(3000, np.random.default_rng(0))
fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
for k, (j, lab, tr) in enumerate([("kr", "relapse hazard shape", lambda x: x), ("m_p", "post-relapse median (mo)", lambda x: x), ("lth_r", "GPS responder relapse HR", np.exp),
                                  ("gamma", "enrollment shape", lambda x: x)]):
    a = ax[0, k]; i = pm.IDX[j]
    lo, hi = tr(pm.LO[i]), tr(pm.HI[i]); bins = np.linspace(lo, hi, 40)
    a.hist(tr(Xp[:, i]), bins, density=True, alpha=.3, color="#6b7280", label="prior")
    a.hist(tr(A["fit"]["X"][:, i]), bins, density=True, histtype="step", lw=2, color="#2a78c8", label="calibrated (A)")
    a.hist(tr(G["fit"]["X"][:, i]), bins, density=True, histtype="step", lw=2, color="#c0392b", label="strong-GPS prior (G)")
    a.set_title(lab, fontsize=9)
    if k == 0: a.legend(fontsize=7)
for k, (nm, key) in enumerate([("BAT 3-yr OS", "pop_bat_S36"), ("GPS 3-yr OS", "pop_gps_S36")]):
    a = ax[1, k]
    for R_, c, lab in [(A, "#2a78c8", "A: Ph2-centred GPS prior"), (G, "#c0392b", "G: stronger GPS prior")]:
        a.hist(R_["rel"][key], np.linspace(0.1, 0.65, 40), density=True, histtype="step", lw=2, color=c, label=lab)
    if k == 0: a.axvspan(.13, .22, color="#2a78c8", alpha=.12); a.text(.135, a.get_ylim()[1] * .9, "VDM-bridge\nrange", fontsize=7)
    else: a.axvline(.47, color="k", ls=":"); a.text(.475, a.get_ylim()[1] * .85, "Ph2 CR1\n47%", fontsize=7)
    a.set_title(nm + " (population, from simulated patients)", fontsize=9); a.legend(fontsize=7)
a = ax[1, 2]
for R_, c, lab in [(A, "#2a78c8", "A"), (G, "#c0392b", "G")]:
    rel = R_["rel"]; C = rel["C"]; w = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1)) * rel["ia_cont"]
    a.hist(rel["hr"], np.linspace(.15, 1.5, 55), weights=w, density=True, histtype="step", lw=2, color=c, label=f"{lab}: given counts & interim continued")
a.axvline(.636, color="k", ls="--"); a.set_title("Cox HR at the reported 80th event", fontsize=9); a.legend(fontsize=7)
a = ax[1, 3]
for i, c in enumerate(["#2a78c8", "#e08a00", "#c0392b"]):
    a.hist(A["rel"]["C"][:, i], np.arange(35, 100), density=True, histtype="step", lw=2, color=c); a.axvline(OBS[i], color=c, ls="--")
a.set_title("realised reported counts (A) vs observed (dashed)", fontsize=9)
ax[0, 3].legend().set_visible(False)
plt.tight_layout(); plt.savefig("results_v2/fig_smc3_patient_model.png", dpi=150); plt.close()
