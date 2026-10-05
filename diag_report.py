import pickle, glob, numpy as np
import regal_model as r, patient_model as pm
OBS = np.array([60., 72., 78.]); fm = lambda a: f"{a[1]:.3g} [{a[0]:.3g},{a[2]:.3g}]"
print(f"{'run':32s} {'rmse':>5s} {'BAT3y':>6s} {'GPS3y':>6s} {'popHR':>6s} {'P(popHR<.636)':>13s} {'HR real|cond':>12s} {'P(succ)|cond':>12s} {'P(IA cont)':>10s}")
for f in sorted(glob.glob("results_v2/smc3_X*.pkl")):
    R = pickle.load(open(f, "rb")); fit, rel = R["fit"], R["rel"]; tag = f.split("smc3_")[1][:-4]
    C = rel["C"]; kern = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1)); w = kern * rel["ia_cont"]
    pw = np.bincount(rel["part"], weights=w, minlength=len(rel["theta"])); pidx = rel["part"]
    # per-trial weights -> particle weights for population quantities
    popw = w
    succ = rel["z"] <= -r.ZF
    rmse = np.sqrt(((OBS[None] - fit["ev"][0]) ** 2).mean())
    print(f"{tag:32s} {rmse:5.2f} {np.average(rel['pop_bat_S36'][pidx], weights=popw+1e-12):6.3f} {np.average(rel['pop_gps_S36'][pidx], weights=popw+1e-12):6.3f} {r.wq(rel['pop_hr'], popw, [.5])[0]:6.3f} {(popw*(rel['pop_hr']<.636)).sum()/popw.sum():13.3f} {r.wq(rel['hr'], w, [.5])[0]:12.3f} {(w*succ).sum()/w.sum():12.3f} {rel['ia_cont'].mean():10.2f}")
