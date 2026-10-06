"""Two-knob scan of the biology-pinned model: bat_eff x gps_eff -> likelihood of 60/72/78 + interim, HR, P(success)."""
import sys, json, pickle, datetime as dt
import numpy as np
import regal_model as r, regal_bio as b

OBS = np.array([60., 72., 78.])
TAU = 1.0
FUT_BOUNDS = (0.78, 0.83, 0.89)                 # futility HR bounds spanning 10% conditional power (current trend .74-.78), design-effect (.83), HSD (.89)


def scan(bio, R=400, th_grid=None, g_grid=None, seed=7, chunk=12, with_pop=True):
    th_grid = np.linspace(0.4, 1.1, 15) if th_grid is None else th_grid
    g_grid = np.linspace(0.0, 1.0, 21) if g_grid is None else g_grid
    TH, G = np.meshgrid(th_grid, g_grid, indexing="ij"); th = TH.ravel(); g = G.ravel(); nk = len(th)
    Z = b.make_Z(R, seed); coh = b.cohort(Z, bio); a_dates = b.enrollment_dates(bio["gamma"])
    mu = np.zeros((nk, 3)); pst = np.zeros(nk); pc = np.zeros(nk); pe = np.zeros(nk); pf = np.zeros(nk); popHR = np.zeros(nk)
    bat12 = np.zeros(nk); bat24 = np.zeros(nk); bat36 = np.zeros(nk); gps12 = np.zeros(nk); gps36 = np.zeros(nk); gps60 = np.zeros(nk); batmed = np.zeros(nk); gpsmed = np.zeros(nk)
    trial = dict(C=np.zeros((nk, R, 3), np.int16), z=np.zeros((nk, R), np.float32), zu=np.zeros((nk, R), np.float32), hr=np.zeros((nk, R), np.float32), ia=np.zeros((nk, R), np.float32))
    for s in range(0, nk, chunk):
        sl = slice(s, min(s + chunk, nk)); n = sl.stop - sl.start
        S = b.simulate(th[sl], g[sl], Z, bio, coh, a_dates)
        C = np.stack([(S["arrive"] <= t).sum(-1) for t in (r.D["e60"], r.D["e72"], r.D["e78"])], -1)
        trial["C"][sl] = C; mu[sl] = C.mean(1)
        pst[sl] = ((S["arrive"] <= r.D["q2"]).sum(-1) <= 79).mean(1)
        zi, hri, _ = b.analyse(S, r.D["e60"])
        z_eff = -zi
        ia = np.mean([((z_eff < r.Z1) & (hri <= h)).astype(float) for h in FUT_BOUNDS], axis=0)       # soft over the unknown futility bound
        trial["ia"][sl] = ia; pc[sl] = ia.mean(1); pe[sl] = (z_eff >= r.Z1).mean(1); pf[sl] = np.mean([(hri > h).mean(1) for h in FUT_BOUNDS], axis=0)
        T80 = np.partition(S["arrive"], 79, axis=-1)[..., 79]
        zf, hrf, _ = b.analyse(S, np.minimum(T80, r.m(dt.date(2030, 1, 1))))
        trial["z"][sl] = zf; trial["hr"][sl] = hrf
        zu, _, _ = b.analyse(dict(S, code=np.zeros_like(S["code"])), np.minimum(T80, r.m(dt.date(2030, 1, 1)))); trial["zu"][sl] = zu
        for i in range(n):
            arm = S["arm"][i]; Tt = S["T"][i]
            sv = lambda t, m: (Tt[m] > t).mean()
            bat12[s + i] = sv(12, ~arm); bat24[s + i] = sv(24, ~arm); bat36[s + i] = sv(36, ~arm); gps12[s + i] = sv(12, arm); gps36[s + i] = sv(36, arm); gps60[s + i] = sv(60, arm)
            batmed[s + i] = np.median(Tt[~arm]); gpsmed[s + i] = np.median(Tt[arm])
            if with_pop:
                tm = np.clip(np.minimum(Tt, r.D["q2"] - S["a"][i]), 0.01, None).reshape(1, -1); ev = (S["d"][i] <= r.D["q2"]).reshape(1, -1).astype(np.int8)
                _, be = b.strat_stats(tm, ev, arm.reshape(1, -1).astype(np.int8), S["code"][i].reshape(1, -1))
                popHR[s + i] = np.exp(be[0])
    ll = -0.5 * (((OBS[None] - mu) / TAU) ** 2).sum(1) + np.log(np.clip(pst, 0.05, 1)) + np.log(np.clip(pc, 0.02, 1))
    kern = np.exp(-0.5 * (((trial["C"] - OBS[None, None]) / 2.0) ** 2).sum(-1)) * trial["ia"]
    succ = trial["z"] <= -r.ZF; succ_un = trial["zu"] <= -r.ZF
    psucc = (kern * succ).sum(1) / np.maximum(kern.sum(1), 1e-300); psucc_un = (kern * succ_un).sum(1) / np.maximum(kern.sum(1), 1e-300)
    hrmed = np.array([r.wq(trial["hr"][i], kern[i] + 1e-300, [.5])[0] for i in range(nk)])
    return dict(th=th, g=g, th_grid=th_grid, g_grid=g_grid, mu=mu, pst=pst, pc=pc, pe=pe, pf=pf, ll=ll, psucc=psucc, hrmed=hrmed, popHR=popHR, kernsum=kern.sum(1),
                bat12=bat12, bat24=bat24, bat36=bat36, gps12=gps12, gps36=gps36, gps60=gps60, batmed=batmed, gpsmed=gpsmed, psucc_uncond=succ.mean(1), psucc_un=psucc_un, coh=coh)


def posterior(sc, g_prior="flat", th_prior=(0.75, 0.15)):
    lp = -0.5 * ((sc["th"] - th_prior[0]) / th_prior[1]) ** 2
    if g_prior == "ph2": lp = lp - 0.5 * ((sc["g"] - 0.5) / 0.2) ** 2
    w = np.exp(sc["ll"] - sc["ll"].max() + lp - lp.max()); return w / w.sum()


def summarize(sc, g_prior="flat"):
    w = posterior(sc, g_prior); q = lambda x: r.wq(x, w, [.05, .5, .95])
    return dict(best=(float(sc["th"][np.argmax(sc["ll"])]), float(sc["g"][np.argmax(sc["ll"])])), bat_eff=q(sc["th"]), gps_eff=q(sc["g"]), P_success=float((w * sc["psucc"]).sum()), P_success_unstrat=float((w * sc["psucc_un"]).sum()),
                BAT36=q(sc["bat36"]), GPS36=q(sc["gps36"]), popHR=q(sc["popHR"]), HR=float(np.average(sc["hrmed"], weights=w)), rmse=float(np.sqrt(((OBS[None] - sc["mu"]) ** 2).mean(1))[np.argmax(w)]),
                mu_at_mode=sc["mu"][np.argmax(w)])


if __name__ == "__main__":
    bio = dict(b.BIO)
    sc = scan(bio, R=400)
    pickle.dump({k: v for k, v in sc.items() if k != "coh"}, open("results_v2/bio_scan_base.pkl", "wb"))
    coh = sc["coh"]
    print("RANDOMISATION CHECK (R=400 simulated trials)")
    arm = coh["arm"]; print("  GPS patients per trial: mean %.2f  sd %.2f  min %d max %d" % (arm.sum(1).mean(), arm.sum(1).std(), arm.sum(1).min(), arm.sum(1).max()))
    imb = []
    for k in range(16):
        m = coh["code"] == k; ng = (arm & m).sum(1); nb = (~arm & m).sum(1); imb.append(np.abs(ng - nb).mean())
    print("  mean |GPS-BAT| imbalance within a stratum: %.2f patients (16 strata)" % np.mean(imb))
    half = arm[:, :64].mean(1); print("  GPS share among first 64 enrolled: %.3f (sd %.3f); last 63: %.3f" % (half.mean(), half.std(), arm[:, 64:].mean()))
    for nm in ("long", "poor", "mrd", "crp"):
        print("  %-5s share  BAT %.3f  GPS %.3f" % (nm, coh[nm][~arm].mean(), coh[nm][arm].mean()))
    print("\nBIOLOGY CHECK at BAT knob 0.75 (observation = 1.0): BAT arm predictions without using any REGAL count")
    i = np.argmin(np.abs(sc["th"] - 0.75) + np.abs(sc["g"] - 0.0))
    print("  BAT OS 12/24/36 mo: %.2f / %.2f / %.2f  median %.1f   | VDM-bridge targets 0.47-0.55 / 0.25-0.34 / 0.13-0.22, median 8.4-14.4" % (sc["bat12"][i], sc["bat24"][i], sc["bat36"][i], sc["batmed"][i]))
    for gp in ("flat", "ph2"):
        s = summarize(sc, gp)
        print(f"\nPOSTERIOR over the two knobs (GPS prior {gp}): bat_eff {s['bat_eff'].round(2)}  gps_eff {s['gps_eff'].round(2)}  | BAT 3y OS {s['BAT36'].round(3)}  GPS 3y OS {s['GPS36'].round(3)}  popHR {s['popHR'].round(2)}  | realised HR {s['HR']:.2f}  P(success) {s['P_success']:.3f}  | mode counts {s['mu_at_mode'].round(1)}")
