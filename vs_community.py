import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# self-reported headline P(success) (low, mid, high) and expected/underlying HR from the posts I read (provenance: Reddit, unverified, several LLM-assisted)
others = [("CW (Kugler reconciliation title)", .82, .82, .82, .43), ("Khela (ABC + log-rank)", .855, .855, .855, .376), ("Stochasty (Markov ABC)", .781, .84, .898, .48),
          ("JustThatGuy (weighted fits)", .55, .741, .88, .537), ("Sveesvee777 (Schoenfeld)", .65, .65, .65, np.nan), ("MoAlbaek (browser tool, base)", .94, .94, .94, np.nan),
          ("Nasfact (summary)", .55, .58, .65, .526), ("neo2551 (if BAT 3y OS<25%)", .90, .90, .90, np.nan), ("Real_Philosopher (BAT<35%..<30%)", .73, .85, .96, np.nan),
          ("FitSet9837 (1M-run script)", .90, .93, .96, .45)]
mid = np.array([o[2] for o in others]); hr = np.array([o[4] for o in others]); hr = hr[~np.isnan(hr)]
print(f"community P(success): n={len(mid)}  median {np.median(mid):.2f}  mean {mid.mean():.2f}  range {mid.min():.2f}-{mid.max():.2f};  expected HR median {np.median(hr):.2f} (n={len(hr)})")
ours = [("ours: no BAT anchor (lean, calibrated)", .462, .71), ("ours: BAT S36 27%±5, Phase 2 GPS", .589, .64), ("ours: BAT S36 18%±8, broad GPS", .768, .54), ("ours: BAT S36 18%±5, Phase 2 GPS", .775, .54),
        ("ours: BAT S36 18%±5, broad GPS", .866, .48), ("ours: BAT S36 18%±3, broad GPS", .944, .42), ("ours: prior only (no data)", .088, .86)]
fig, ax = plt.subplots(figsize=(10, 6.2))
y = 0
labels = []
for nm, lo, m_, hi, h in others[::-1]:
    ax.plot([lo, hi], [y, y], color="#6b7280", lw=3 if hi > lo else 0); ax.plot(m_, y, "o", color="#6b7280"); labels.append(nm); y += 1
y += 0.5
for nm, p, ph in ours[::-1]:
    ax.plot(p, y, "D", color="#c0392b" if "no BAT anchor" in nm or "prior only" in nm else "#2a78c8"); labels.append(nm); y += 1
ys = list(range(len(others))) + [len(others) + 0.5 + i for i in range(len(ours))]
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8); ax.axvline(np.median(mid), color="k", ls="--", lw=1); ax.text(np.median(mid) + .005, y - 0.3, f"community median {np.median(mid):.0%}", fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("P(success) (self-reported by others; ours = share of calibrated realised trials crossing the final boundary)"); ax.set_title("REGAL P(success): community models vs this model")
plt.tight_layout(); plt.savefig("results_v2/fig_vs_community.png", dpi=150); plt.close()
print("\nOUR GAP vs community median:")
for nm, p, ph in ours:
    print(f"  {nm:44s} P(success) {p:.0%}  ({(p-np.median(mid))*100:+.0f} pts)   underlying HR {ph:.2f} ({ph-np.median(hr):+.2f} vs community {np.median(hr):.2f})")
import math
lg = lambda p: math.log(p / (1 - p))
print("\nLIKELIHOOD PUSH (log-odds / odds multiplier relative to our own prior-predictive 8.8%):")
for nm, p in [("counts only (rejection)", .177), ("counts + interim (rejection)", .332), ("full SMC calibration", .462), ("+ BAT literature anchor ±8, Phase2 GPS", .775), ("+ BAT anchor ±5, broad GPS", .866)]:
    print(f"  {nm:42s} P={p:.1%}  odds x{math.exp(lg(p)-lg(.088)):.1f}")
