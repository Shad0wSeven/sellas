import pickle, datetime as dt, numpy as np, csv, json
import regal_model as r
from scipy.stats import norm

R = "results"
scen = pickle.load(open(f"{R}/scenarios.pkl", "rb"))
sweep = pickle.load(open(f"{R}/sweep.pkl", "rb"))

def date(mo): return (r.T0 + dt.timedelta(days=float(mo) * 30.4375)).isoformat()

# ---- priors (importance weights relative to the flat sampling prior)
def w_bat_lit(res):
    S36, M, k = res["S36"], res["M"], res["k"]
    ln = np.exp(-0.5 * ((np.log(S36) - np.log(0.18)) / 0.38) ** 2) / S36        # lognormal density
    return ln * norm.pdf(M, 12.5, 2.5) * norm.pdf(k, 0.85, 0.25)
def w_gps_inf(res):
    return norm.pdf(res["gps_S36"], 0.50, 0.15)

PRIORS = {
    "flat BAT, flat GPS": lambda res: np.ones_like(res["w"]),
    "literature BAT, flat GPS": lambda res: w_bat_lit(res),
    "literature BAT + Ph2-informed GPS": lambda res: w_bat_lit(res) * w_gps_inf(res),
}

def summarize(res, prior_fn, cond_ia=True):
    w = res["w"] * prior_fn(res)
    ev = w.sum() / (res["w"].sum() + 1e-300) * res["evidence"]            # prior-weighted evidence (relative)
    ia_cont = res["z_int"] > -r.Z1
    p_ia_stop = r.wmean((~ia_cont).astype(float), w)
    wc = w * ia_cont if cond_ia else w
    q = lambda x, ww=wc: r.wq(x, ww, [.05, .5, .95])
    wk = lambda x: r.wmean(x, wc)
    out = dict(
        ESS=round(r.ess(wc)), evidence=ev, P_IA_efficacy_stop=p_ia_stop,
        BAT_S36=q(res["S36"]), BAT_median=q(res["M"]), BAT_k=q(res["k"])[1], BAT_S12=q(res["bat_S12"])[1], BAT_S60=q(res["bat_S60"])[1],
        GPS_S36=q(res["gps_S36"]), cure_c=q(res["c"])[1],
        gap72=wk(res["true_72"] - res["rep_72"]), gap78=wk(res["true_78"] - res["rep_78"]),
        gap78_BAT=wk(res["trueBAT_78"] - res["repBAT_78"]),
        gap_q2=wk(res["true_q2"] - res["rep_q2"]), gap_now=wk(res["true_now"] - res["rep_now"]),
        true78=wk(res["true_78"]), true_q2=wk(res["true_q2"]), true_now=wk(res["true_now"]),
        P_true80_by_q2=wk((res["T80true"] <= r.D["q2"]).astype(float)),
        P_true80_by_now=wk((res["T80true"] <= r.D["now"]).astype(float)),
        P_rep80_by_now=wk((res["T80rep"] <= r.D["now"]).astype(float)),
        aliveBAT_q2=q(res["aliveBAT_q2"])[1], aliveGPS_q2=q(res["aliveGPS_q2"])[1],
        HR_final=q(res["hr_final"]), HR_final_unswept=q(res["hr_final_unswept"]),
        events_final=wk(res["ev_final"]), events_final_unswept=wk(res["ev_final_unswept"]),
        P_success=wk((res["z_final"] <= -r.ZF).astype(float)),
        P_success_unswept=wk((res["z_final_unswept"] <= -r.ZF).astype(float)),
        HR_interim=q(res["hr_int"])[1],
    )
    # forecast of the *reported* 80th conditional on not reported by now
    nn = wc * (res["T80rep"] > r.D["now"])
    if nn.sum() > 0:
        t = r.wq(res["T80rep"], nn, [.1, .5, .9]); out["T80rep_cond_notyet"] = [date(x) for x in t]
        out["P_rep80_by_2026_12_31"] = r.wmean((res["T80rep"] <= r.m(dt.date(2026, 12, 31))).astype(float), nn)
    return out

def fmt(x):
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 3 and not isinstance(x[0], str):
        return f"{x[1]:.3g} [{x[0]:.3g}, {x[2]:.3g}]"
    if isinstance(x, float): return f"{x:.3g}"
    return str(x)

table = {}
for pn, pf in PRIORS.items():
    base_ev = None
    for sn, res in scen.items():
        s = summarize(res, pf)
        if sn.startswith("S0"): base_ev = s["evidence"]
        s["Bayes_factor_vs_S0"] = s["evidence"] / base_ev if base_ev else np.nan
        table[(pn, sn)] = s

json.dump({f"{p} | {s}": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in d.items()} for (p, s), d in table.items()},
          open(f"{R}/summary.json", "w"), indent=1, default=float)

keys = ["ESS", "Bayes_factor_vs_S0", "BAT_S36", "BAT_median", "BAT_k", "GPS_S36", "cure_c", "gap72", "gap78", "gap78_BAT", "gap_q2", "gap_now",
        "true78", "true_q2", "P_true80_by_q2", "P_true80_by_now", "P_rep80_by_now", "aliveBAT_q2", "aliveGPS_q2",
        "HR_final", "HR_final_unswept", "events_final", "events_final_unswept", "P_success", "P_success_unswept", "P_IA_efficacy_stop",
        "T80rep_cond_notyet", "P_rep80_by_2026_12_31"]
with open(f"{R}/summary_table.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["prior", "scenario"] + keys)
    for (p, s), d in table.items(): wr.writerow([p, s] + [fmt(d.get(k, "")) for k in keys])

# ---- sweep over q_bat
sw = []
for q, res in sweep.items():
    s = summarize(res, PRIORS["literature BAT, flat GPS"])
    sw.append((q, s))
with open(f"{R}/sweep_table.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["q_bat_immediate"] + keys)
    for q, s in sw: wr.writerow([q] + [fmt(s.get(k, "")) for k in keys])

pickle.dump(table, open(f"{R}/table.pkl", "wb"))
print(open(f"{R}/summary_table.csv").read())
print(open(f"{R}/sweep_table.csv").read())
