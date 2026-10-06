import json, pickle, numpy as np
import regal_model as r, regal_bio as b, bio_scan as bs

TH = np.linspace(0.4, 1.1, 15); G = np.linspace(0.0, 1.0, 21)
def logZ(sc, th_prior=(0.75, 0.15), g_prior=(0.5, 0.25)):
    lp = -0.5 * ((sc["th"] - th_prior[0]) / th_prior[1]) ** 2 - 0.5 * ((sc["g"] - g_prior[0]) / g_prior[1]) ** 2
    w = np.exp(lp - lp.max()); w /= w.sum()
    m = sc["ll"].max(); return m + np.log((w * np.exp(sc["ll"] - m)).sum()), w * np.exp(sc["ll"] - m) / (w * np.exp(sc["ll"] - m)).sum()
def run(name, bio_over=None, lag=None, recal=True, rfs_extra=None):
    bio = dict(b.BIO, **(bio_over or {}))
    b.LAG.update(dict(mu_bat=0., sd_bat=0., mu_gps=0., sd_gps=0.)); b.LAG.update(lag or {})
    if recal:
        saved = dict(b.RFS_TARGET)
        if rfs_extra: b.RFS_TARGET.clear(); b.RFS_TARGET.update(rfs_extra)
        bio["Mr"], bio["kr"], _ = b.calibrate_relapse(bio)
        b.RFS_TARGET.clear(); b.RFS_TARGET.update(saved)
    sc = bs.scan(bio, R=300, th_grid=TH, g_grid=G, with_pop=False)
    lz, post = logZ(sc)
    q = lambda x: r.wq(x, post, [.05, .5, .95]); i = np.argmax(sc["ll"])
    out = dict(name=name, logZ=lz, bat_eff=q(sc["th"]), gps_eff=q(sc["g"]), BAT36=q(sc["bat36"]), GPS36=q(sc["gps36"]), P=float((post * sc["psucc"]).sum()), HR=float(np.average(sc["hrmed"], weights=post)),
               ll_max=float(sc["ll"].max()), best=(float(sc["th"][i]), float(sc["g"][i])), mu_best=sc["mu"][i], Mr=bio["Mr"], kr=bio["kr"])
    print(f"{name:34s} logZ {lz:7.2f} | best knobs bat {out['best'][0]:.2f} gps {out['best'][1]:.2f} ll {out['ll_max']:6.2f} counts {np.round(out['mu_best'],1)} | post bat_eff {out['bat_eff'][1]:.2f} gps_eff {out['gps_eff'][1]:.2f} BAT3y {out['BAT36'][1]:.2f} GPS3y {out['GPS36'][1]:.2f} HR {out['HR']:.2f} P(succ) {out['P']:.2f}", flush=True)
    return out
res = []
res.append(run("base (lag 0)", recal=False))
b.BIO["Mr"], b.BIO["kr"] = json.load(open("calib_bio.json"))["Mr"], json.load(open("calib_bio.json"))["kr"]
for nm, lag in [("BAT lag N(60,30) d, GPS N(15,10)", dict(mu_bat=60, sd_bat=30, mu_gps=15, sd_gps=10)), ("BAT lag N(120,60) d", dict(mu_bat=120, sd_bat=60, mu_gps=15, sd_gps=10)),
                ("BAT lag N(240,120) d", dict(mu_bat=240, sd_bat=120, mu_gps=15, sd_gps=10)), ("BAT lag N(360,180) d", dict(mu_bat=360, sd_bat=180, mu_gps=15, sd_gps=10)),
                ("symmetric lag N(120,60) d both", dict(mu_bat=120, sd_bat=60, mu_gps=120, sd_gps=60))]:
    res.append(run(nm, lag=lag, recal=False))
for nm, ov, ex in [("biology: never-relapse 15%", dict(c_b=0.15), None), ("biology: heavier relapse tail (3-yr RFS 14%)", {}, {6: .45, 12: .27, 36: .14}),
                   ("biology: post-relapse median 7.5 mo", dict(m_p=7.5), None), ("biology: transplant fraction 20%", dict(p_sct=0.20), None),
                   ("biology: frailty SD 0.9", dict(sigma_f=0.9), None), ("biology: CR2 relapse worse (RFS 40/22%)", {}, {6: .40, 12: .22})]:
    res.append(run(nm, bio_over=ov, rfs_extra=ex))
pickle.dump(res, open("results_v2/bio_scenarios.pkl", "wb"))
