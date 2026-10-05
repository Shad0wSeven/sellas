"""Bridge van der Maas/Versluis (VDM) untransplanted-CR2 survival (clock = first relapse) onto REGAL's clock
(randomisation, <=6 months after CR2).

Source numbers: author email relayed on r/sellaslifesciences (n=60; 1y 52.4%, 2y 33.2%, 3y 18.6% [10.7-32.2];
at risk 30/19/10; mOS 16.8 [12.3-27.4]).  Cohort: HOVON-SAKK 2000-2018, intensive re-induction, median age at
relapse 58 (Blood Adv 2025;9:3853).  Subset = achieved CR2, no transplant => no deaths between relapse and CR2
by construction.

Two separate clock effects (they push in OPPOSITE directions):
  d0 = relapse -> CR2 (1.5-3 mo): pure clock shift, SHORTENS survival measured from CR2 by d0.
  d1 = CR2 -> randomisation (0-6 mo): left truncation; patients who die before d1 never enter REGAL,
       so from-randomisation survival is S(t+d1)/S(d1) -> LIFTS survival when hazard declines (Weibull k<1).
phi = hazard multiplier for older/less fit population (VDM validation cohort age 69 vs 58: H(4y) ratio 1.3, H(1y) 1.08).
"""
import numpy as np, csv

KM = np.array([[12, .524], [24, .332], [36, .186]])

def fit_shifted(pts, d0):
    x = np.log(pts[:, 0] - d0); y = np.log(-np.log(pts[:, 1]))
    k, b = np.polyfit(x, y, 1)
    return k, np.exp(-b / k)

def S(t, k, lam, phi=1.0):
    return np.exp(-phi * (np.clip(t, 0, None) / lam) ** k)

def from_rand(k, lam, d1, phi):
    s1 = S(d1, k, lam, phi)
    f = lambda t: S(t + d1, k, lam, phi) / s1
    H = -np.log(0.5 * s1) / phi
    med = lam * H ** (1 / k) - d1
    return f(12), f(24), f(36), med, s1

rows = []
variants = {"point (S36 18.6%)": KM,
            "S36 low (10.7%)": np.array([[12, .524], [24, .332], [36, .107]]),
            "S36 high (32.2%)": np.array([[12, .524], [24, .332], [36, .322]])}
for vn, p in variants.items():
    for d0 in (1.5, 2.5):
        k, lam = fit_shifted(p, d0)
        for d1 in (0, 2, 4):
            for phi in (1.0, 1.2):
                s12, s24, s36, med, s1 = from_rand(k, lam, d1, phi)
                rows.append(dict(variant=vn, d0_relapse_to_CR2=d0, d1_CR2_to_rand=d1, phi_age=phi, weibull_k=round(k, 2),
                                 surv_to_rand=round(s1, 3), S12=round(s12, 3), S24=round(s24, 3),
                                 S36=round(s36, 3), median_from_rand=round(med, 1)))
with open("results/vdm_bridge.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__ == "__main__":
    for rw in rows:
        if rw["variant"].startswith("point"): print(rw)
