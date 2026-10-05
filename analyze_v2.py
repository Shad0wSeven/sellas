import pickle, csv, datetime as dt, numpy as np
from scipy.stats import norm
import regal_model as r

R2 = "results_v2"
scen = pickle.load(open(f"{R2}/scen_v2.pkl", "rb"))
def date(mo): return (r.T0 + dt.timedelta(days=float(mo) * 30.4375)).isoformat()

def w_lit(res):
    S36, M, k = res["S36"], res["M"], res["k"]
    ok = np.isfinite(M) & (S36 > 0)
    ln = np.exp(-0.5 * ((np.log(np.clip(S36, 1e-6, None)) - np.log(0.18)) / 0.38) ** 2) / np.clip(S36, 1e-6, None)
    out = ln * norm.pdf(np.where(ok, M, 0), 12.5, 2.5) * norm.pdf(k, 0.85, 0.25)
    return np.where(ok, out, 0.0)
def w_fu(res): return np.exp(-0.5 * ((res["fu_med_ia"] - 13.5) / 1.0) ** 2)
def w_gps(res): return norm.pdf(res["gps_S36"], 0.50, 0.15)

IA = {
    "none (ignore IDMC)":                      lambda x: np.ones_like(x["w"], bool),
    "no efficacy stop (z>-2.34) [v1]":         lambda x: x["z_int"] > -r.Z1,
    "futility pass: IA HR<=1 & no stop":       lambda x: (x["z_int"] > -r.Z1) & (x["hr_int"] <= 1.0),
    "Stochasty: GPS IA deaths < BAT":          lambda x: (x["ia_deaths"] - x["ia_deaths_bat"]) < x["ia_deaths_bat"],
    "Thetamancer: IA HR .50-.90 & BAT>=32/60": lambda x: (x["hr_int"] >= 0.5) & (x["hr_int"] <= 0.9) & (x["ia_deaths_bat"] >= 32),
    "Grand_Effort: IA HR >= 0.55":             lambda x: x["hr_int"] >= 0.55,
}

def summ(res, extra=None, ia="no efficacy stop (z>-2.34) [v1]", prior=w_lit, use_fu=False, mask=None):
    w = res["w"] * prior(res)
    if use_fu: w = w * w_fu(res)
    if mask is not None: w = w * mask
    w = w * IA[ia](res)
    if w.sum() <= 0 or r.ess(w) < 25: return None
    q = lambda x: r.wq(x, w, [.05, .5, .95]); f = lambda b: r.wmean(b.astype(float), w)
    d = dict(ESS=round(r.ess(w)), BAT_S36=q(res["S36"]), BAT_M=q(np.where(np.isfinite(res["M"]), res["M"], 99)), GPS_S36=q(res["gps_S36"]),
             fu_med_IA=q(res["fu_med_ia"])[1], enrol_med=date(q(res["enrol_median"])[1]),
             gap78=r.wmean(res["true_78"] - res["rep_78"], w), gap_now=r.wmean(res["true_now"] - res["rep_now"], w),
             P_true80_by_q2=f(res["T80true"] <= r.D["q2"]), P_true80_now=f(res["T80true"] <= r.D["now"]),
             HR=q(res["hr_final"]), P_success=f(res["z_final"] <= -r.ZF), P_success_reported_only=f(res["z_final_unswept"] <= -r.ZF),
             P_theta_gt1=f(res["theta"] > 1.0) if "theta" in res else np.nan,
             BAT_alive_q2=q(res["aliveBAT_q2"])[1], GPS_alive_q2=q(res["aliveGPS_q2"])[1],
             stall_lik=res["p_stall_given_78"])
    return d
fmt = lambda x: (f"{x[1]:.3g} [{x[0]:.3g},{x[2]:.3g}]" if isinstance(x, np.ndarray) else (f"{x:.3g}" if isinstance(x, float) else str(x)))

rows = []
for nm, res in scen.items():
    for pn, kw in [("lit BAT", {}), ("lit BAT + FU13.5 kernel", dict(use_fu=True)), ("flat BAT", dict(prior=lambda x: np.ones_like(x["w"])))]:
        d = summ(res, **kw)
        if d: rows.append(dict(scenario=nm, prior=pn, **{k: fmt(v) for k, v in d.items()}))
with open(f"{R2}/table_scenarios.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

# IA constraint variants
ia_rows = []
for nm in ["V0 no lag", "V3 calls to wk156 + real-world 50%", "TL3 two-stage (lit-compatible), calls+real-world"]:
    for iak in IA:
        d = summ(scen[nm], ia=iak)
        if d: ia_rows.append(dict(scenario=nm, IA_constraint=iak, ESS=d["ESS"], BAT_S36=fmt(d["BAT_S36"]), GPS_S36=fmt(d["GPS_S36"]), HR=fmt(d["HR"]), P_success=round(d["P_success"], 3)))
with open(f"{R2}/table_ia_variants.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=ia_rows[0].keys()); wr.writeheader(); wr.writerows(ia_rows)

# theta cap sensitivity (GPS non-responders allowed worse than BAT or not)
th_rows = []
for nm in ["V0 no lag", "V3 calls to wk156 + real-world 50%"]:
    for lab, mk in [("theta<=1.0 (v1)", scen[nm]["theta"] <= 1.0), ("theta<=1.3 (v2)", np.ones_like(scen[nm]["theta"], bool))]:
        d = summ(scen[nm], mask=mk)
        if d: th_rows.append(dict(scenario=nm, theta_prior=lab, BAT_S36=fmt(d["BAT_S36"]), HR=fmt(d["HR"]), P_success=round(d["P_success"], 3), P_theta_gt1=round(d["P_theta_gt1"], 3)))
with open(f"{R2}/table_theta.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=th_rows[0].keys()); wr.writeheader(); wr.writerows(th_rows)

pickle.dump(rows, open(f"{R2}/rows.pkl", "wb"))
for fn in ["table_scenarios.csv", "table_ia_variants.csv", "table_theta.csv"]:
    print("\n==", fn); print(open(f"{R2}/{fn}").read())
