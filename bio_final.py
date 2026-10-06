import json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, regal_bio as b, bio_scan as bs
c = json.load(open("calib_bio.json")); bio = dict(b.BIO, Mr=c["Mr"], kr=c["kr"])
sc = bs.scan(bio, R=600)
pickle.dump({k: v for k, v in sc.items() if k != "coh"}, open("results_v2/bio_final.pkl", "wb"))
n = lambda a: np.round(a, 3)
print("BIOLOGY-ONLY PREDICTIONS (no REGAL counts used), BAT knob 0.75, observation/GPS off:")
i = np.argmin(np.abs(sc["th"] - 0.75) + np.abs(sc["g"])); print("  BAT OS 12/24/36 mo %.2f/%.2f/%.2f, median %.1f   (VDM bridge 0.47-0.55/0.25-0.34/0.13-0.22, 8.4-14.4)" % (sc["bat12"][i], sc["bat24"][i], sc["bat36"][i], sc["batmed"][i]))
for gp in ("flat", "ph2"):
    s = bs.summarize(sc, gp); w = bs.posterior(sc, gp)
    print(f"\nTWO-KNOB POSTERIOR (GPS-knob prior {gp}; BAT-knob prior N(.75,.15))")
    print(f"  best cell bat_eff {s['best'][0]:.2f} gps_eff {s['best'][1]:.2f}; posterior bat_eff {n(s['bat_eff'])}  gps_eff {n(s['gps_eff'])}")
    print(f"  predicted BAT 3y OS {n(s['BAT36'])}  GPS 3y OS {n(s['GPS36'])}  noise-free HR {n(s['popHR'])}  realised HR {s['HR']:.2f}")
    print(f"  P(success) stratified (protocol) {s['P_success']:.3f}   unstratified {s['P_success_unstrat']:.3f}   expected counts at mode {n(s['mu_at_mode'])} (obs 60/72/78)")
# P(success | cell) table
th_g, g_g = sc["th_grid"], sc["g_grid"]
P = sc["psucc"].reshape(len(th_g), len(g_g)); LL = sc["ll"].reshape(len(th_g), len(g_g))
print("\nP(success | knobs, counts, interim)  rows = bat_eff (relapse HR of active BAT), cols = gps_eff (durable fraction)")
cols = [0, 4, 8, 12, 14, 16, 18, 20]; print("bat\\gps  " + "  ".join(f"{g_g[j]:.2f}" for j in cols))
for ti in [0, 2, 4, 6, 8, 10, 12, 14]: print(f"{th_g[ti]:.2f}     " + "  ".join(f"{P[ti, j]:.2f}" for j in cols))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
ext = [g_g[0], g_g[-1], th_g[0], th_g[-1]]
for a, M, ttl, cm in [(ax[0], LL - LL.max(), "log-likelihood of 60/72/78 + interim", "viridis"), (ax[1], P, "P(success | knobs, counts, interim)", "RdYlGn"),
                      (ax[2], sc["bat36"].reshape(P.shape), "predicted BAT 3-yr OS", "Blues"), (ax[3], sc["gps36"].reshape(P.shape), "predicted GPS 3-yr OS", "Purples")]:
    im = a.imshow(M, origin="lower", aspect="auto", extent=ext, cmap=cm, vmin=(-15 if "log-lik" in ttl else None)); a.set_title(ttl, fontsize=10); a.set_xlabel("gps_eff: durable fraction of GPS patients"); a.set_ylabel("bat_eff: relapse HR of active BAT (lower = stronger)")
    plt.colorbar(im, ax=a, fraction=.046)
    w = bs.posterior(sc, "flat").reshape(P.shape); a.contour(g_g, th_g, w, levels=[w.max() * .5, w.max() * .1], colors="k", linewidths=1)
plt.tight_layout(); plt.savefig("results_v2/fig_bio_two_knobs.png", dpi=150); plt.close()
