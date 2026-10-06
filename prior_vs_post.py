import os, json, pickle, numpy as np
import regal_model as r, patient_model as pm, smc3
OBS = np.array([60., 72., 78.])
def summ(rel, tag):
    C = rel["C"]; kern = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1)); ia = rel["ia_cont"]; succ = rel["z"] <= -r.ZF; pidx = rel["part"]
    out = {}
    for nm, w in [("no data", np.ones(len(kern))), ("counts only", kern), ("counts + interim", kern * ia)]:
        w = w + 1e-300
        out[nm] = dict(ess=w.sum() ** 2 / (w ** 2).sum(), P_succ=(w * succ).sum() / w.sum(), HR=r.wq(rel["hr"], w, [.05, .5, .95]), popHR=r.wq(rel["pop_hr"], w, [.5])[0],
                       BAT36=np.average(rel["pop_bat_S36"][pidx], weights=w), GPS36=np.average(rel["pop_gps_S36"][pidx], weights=w))
    return out
rng = np.random.default_rng(123)
Xp = pm.sample_prior(1500, rng)
rel = smc3.realised(Xp, R2=100, nsel=1500, seed=5)
res = summ(rel, "prior")
print("PRIOR PREDICTIVE (lean model, no calibration; 1500 prior draws x 100 trials)")
for k, v in res.items():
    print(f"  {k:18s} ESS {v['ess']:8.0f}  BAT3y {v['BAT36']:.2f}  GPS3y {v['GPS36']:.2f}  popHR {v['popHR']:.2f}  realised HR {v['HR'][1]:.2f} [{v['HR'][0]:.2f},{v['HR'][2]:.2f}]  P(success) {v['P_succ']:.3f}")
pickle.dump(res, open("results_v2/prior_pred_summary.pkl", "wb"))
