import numpy as np, csv
import regal_model as r
from analyze import scen, PRIORS
bins = [(0.03,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.25),(0.25,0.30),(0.30,0.35),(0.35,0.45)]
out = []
for sn in ["S0 no lag", "S3 protocol + real-world (50%)", "S6 S3 + 10% BAT silent deaths"]:
    res = scen[sn]
    w0 = res["w"] * (res["z_int"] > -r.Z1)               # flat prior, conditioned on IA continuing
    tot = w0.sum()
    for lo, hi in bins:
        m = (res["S36"] >= lo) & (res["S36"] < hi); w = w0 * m
        if w.sum() < 1e-12 or r.ess(w) < 15: out.append([sn, f"{lo:.2f}-{hi:.2f}", w.sum()/tot, "n/a"] + [""]*7); continue
        out.append([sn, f"{lo:.2f}-{hi:.2f}", round(w.sum()/tot,3), round(r.ess(w)),
                    round(r.wq(res["M"],w,[.5])[0],1), round(r.wq(res["gps_S36"],w,[.5])[0],3),
                    round(r.wq(res["aliveBAT_q2"],w,[.5])[0]), round(r.wq(res["aliveGPS_q2"],w,[.5])[0]),
                    round(r.wq(res["hr_final"],w,[.5])[0],3), round(r.wmean((res["z_final"]<=-r.ZF).astype(float),w),3),
                    round(r.wmean((res["z_final_unswept"]<=-r.ZF).astype(float),w),3)])
hdr = ["scenario","BAT 3y OS bin","posterior mass (flat prior)","ESS","BAT median","GPS S36","BAT alive@Q2","GPS alive@Q2","HR final","P(success)","P(success) reported-only"]
with open("results/bat_map.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(hdr); wr.writerows(out)
for o in out: print(o)
