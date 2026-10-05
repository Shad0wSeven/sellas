import numpy as np, csv
from scipy.stats import norm
import regal_model as r
from analyze import scen

BAT = {"BAT lit (S36~18%, M~12.5)": lambda x: np.exp(-0.5*((np.log(x["S36"])-np.log(0.18))/0.38)**2)/x["S36"]*norm.pdf(x["M"],12.5,2.5)*norm.pdf(x["k"],0.85,0.25),
       "BAT wide (S36~25%, M~14)":   lambda x: np.exp(-0.5*((np.log(x["S36"])-np.log(0.25))/0.45)**2)/x["S36"]*norm.pdf(x["M"],14,3.5)*norm.pdf(x["k"],0.85,0.30),
       "BAT flat":                   lambda x: np.ones_like(x["w"])}
GPS = {"GPS skeptical (S36 .35±.10)": (0.35,0.10), "GPS Ph2-like (.50±.10)": (0.50,0.10), "GPS strong (.60±.10)": (0.60,0.10), "GPS very strong (.70±.10)": (0.70,0.10)}
rows=[]
for sn in ["S0 no lag","S3 protocol + real-world (50%)","S6 S3 + 10% BAT silent deaths"]:
    res=scen[sn]; ia=(res["z_int"]>-r.Z1)
    for bn,bf in BAT.items():
        for gn,(mu,sd) in GPS.items():
            w=res["w"]*bf(res)*norm.pdf(res["gps_S36"],mu,sd)
            wi=w*ia
            if r.ess(wi)<40: rows.append([sn,bn,gn,"ESS<40"]+[""]*5); continue
            rows.append([sn,bn,gn,round(r.ess(wi)),
                         f'{r.wq(res["S36"],wi,[.5])[0]:.3f}', f'{r.wq(res["M"],wi,[.5])[0]:.1f}',
                         f'{r.wq(res["hr_final"],wi,[.5])[0]:.3f}', f'{r.wmean((res["z_final"]<=-r.ZF).astype(float),wi):.3f}',
                         f'{r.wmean(ia.astype(float),w):.3f}'])
hdr=["scenario","BAT prior","GPS prior","ESS","post BAT S36","post BAT median","HR final","P(success)","P(IA continues) [weight]"]
with open("results/prior_grid.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(hdr); wr.writerows(rows)
cur=None
for rw in rows:
    if rw[0]!=cur: cur=rw[0]; print("\n##",cur)
    print(" | ".join(map(str,rw[1:])))
