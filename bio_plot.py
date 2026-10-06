import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regal_model as r, bio_scan as bs
sc = pickle.load(open("results_v2/bio_final.pkl", "rb"))
th_g, g_g = sc["th_grid"], sc["g_grid"]; shp = (len(th_g), len(g_g))
P = sc["psucc"].reshape(shp); LL = sc["ll"].reshape(shp); ok = LL >= LL.max() - 6
print("P(success | knobs, counts, interim); '--' where the cell cannot reproduce the data (log-lik more than 6 below the best)")
cols = [4, 8, 12, 14, 16, 18, 20]; print("bat_eff \\ gps_eff  " + "  ".join(f"{g_g[j]:.2f}" for j in cols))
for ti in range(0, len(th_g), 2): print(f"   {th_g[ti]:.2f}            " + "  ".join((f"{P[ti, j]:.2f}" if ok[ti, j] else " -- ") for j in cols))
print("\nlog-likelihood relative to best (same cells):")
for ti in range(0, len(th_g), 2): print(f"   {th_g[ti]:.2f}            " + "  ".join((f"{LL[ti, j]-LL.max():5.1f}") for j in cols))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4)); ext = [g_g[0], g_g[-1], th_g[0], th_g[-1]]
w = bs.posterior(sc, "flat").reshape(shp)
for a, M, ttl, cm, vm in [(ax[0], np.where(LL - LL.max() > -15, LL - LL.max(), -15), "log-likelihood of 60/72/78 + interim (relative)", "viridis", None),
                          (ax[1], np.where(ok, P, np.nan), "P(success | knobs, counts, interim)", "RdYlGn", (0, 1)),
                          (ax[2], sc["bat36"].reshape(shp), "predicted BAT 3-yr OS", "Blues", None), (ax[3], sc["gps36"].reshape(shp), "predicted GPS 3-yr OS", "Purples", None)]:
    im = a.imshow(M, origin="lower", aspect="auto", extent=ext, cmap=cm, vmin=None if vm is None else vm[0], vmax=None if vm is None else vm[1]); a.set_title(ttl, fontsize=10)
    a.set_xlabel("gps_eff: durable fraction of GPS patients"); a.set_ylabel("bat_eff: relapse HR of active BAT (lower = stronger)"); plt.colorbar(im, ax=a, fraction=.046)
    a.contour(g_g, th_g, w, levels=[w.max() * .1, w.max() * .5], colors="k", linewidths=1)
plt.tight_layout(); plt.savefig("results_v2/fig_bio_two_knobs.png", dpi=150); plt.close()
