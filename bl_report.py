import os, json, pickle, importlib, numpy as np
import regal_model as r
OBS = np.array([60., 72., 78.])
GPSW = {"p_resp": [0.65, 0.25, 0.2, 1.0], "f_dur": [0.5, 0.35, 0.0, 1.0], "lL": [1.1, 0.6, 0.0, 2.5]}
RUNS = {"BL0_base_for_ref": ({}, "baseline: no BAT outcome anchor, original GPS priors"),
        "BL1_batLit_gpsBroad": (GPSW, "BAT anchored (S12 .52±.06, S24 .30±.06, S36 .18±.05), GPS broad"),
        "BL2_batLit_wider": (GPSW, "BAT anchor wider (±.09/.09/.08), GPS broad"),
        "BL3_batLitKugler": (GPSW, "BAT anchored higher (S12 .58, S24 .40, S36 .27: Kugler-type reading), GPS broad"),
        "BL5_batLit_gpsPh2": ({}, "BAT anchored (S36 .18), GPS priors centred on Phase 2 (resp .70, durable .55, onset 3 mo)"),
        "BL6_batKugler_gpsPh2": ({}, "BAT anchored high (S36 .27), GPS Phase 2-centred priors"),
        "BL4_batLit_gpsBroad_tight": (GPSW, "BAT anchor tight (±.04/.04/.03), GPS broad")}
BAT_P = ["Mr", "kr", "c_b", "b_long", "sig_f", "m_p", "th_B"]; GPS_P = ["p_resp", "f_dur", "lL"]
tr = lambda n, v: np.exp(v) if n == "lL" else v
print(f"{'run':30s} {'rmse':>5s} | BAT S12/S24/S36 (post)     | GPS S12/S36/S60         | popHR  P(popHR<.636) | HR real  P(succ)")
for tag, (ov, desc) in RUNS.items():
    os.environ["PM_OVERRIDE"] = json.dumps(ov); import patient_model as pm; importlib.reload(pm)
    R = pickle.load(open(f"results_v2/smc3_{tag}.pkl", "rb")); fit, rel = R["fit"], R["rel"]
    C = rel["C"]; kern = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1)); w = kern * rel["ia_cont"]; pidx = rel["part"]
    q = lambda k: np.average(rel[k][pidx], weights=w + 1e-12)
    succ = rel["z"] <= -r.ZF; rmse = np.sqrt(((OBS[None] - fit["ev"][0]) ** 2).mean())
    print(f"{tag:30s} {rmse:5.2f} | {q('pop_bat_S12'):.2f} / {q('pop_bat_S24') if 'pop_bat_S24' in rel else float('nan'):.2f} / {q('pop_bat_S36'):.2f} | {q('pop_gps_S12'):.2f} / {q('pop_gps_S36'):.2f} / {q('pop_gps_S60'):.2f} | {r.wq(rel['pop_hr'], w, [.5])[0]:.3f}  {(w*(rel['pop_hr']<.636)).sum()/w.sum():.3f} | {r.wq(rel['hr'], w, [.5])[0]:.3f}  {(w*succ).sum()/w.sum():.3f}")
    Xp = pm.sample_prior(4000, np.random.default_rng(0)); X = fit["X"]
    def mv(names):
        sh, ct = [], []
        for n in names:
            j = pm.IDX[n]; a, b = tr(n, Xp[:, j]), tr(n, X[:, j]); sh.append(abs(np.median(b) - np.median(a)) / a.std()); ct.append(b.std() / a.std())
        return np.mean(sh), np.mean(ct)
    sb, cb = mv(BAT_P); sg, cg = mv(GPS_P)
    print(f"   {desc}\n   data movement (mean |shift| in prior SDs / posterior-SD contraction): BAT params {sb:.2f} / {cb:.2f}   GPS params {sg:.2f} / {cg:.2f}")
    print("   GPS posterior: p_resp %.2f [%.2f,%.2f]  f_dur %.2f [%.2f,%.2f]  onset median %.1f mo [%.1f,%.1f]" % tuple(np.concatenate([np.percentile(tr(n, X[:, pm.IDX[n]]), [50, 5, 95]) for n in GPS_P])))
