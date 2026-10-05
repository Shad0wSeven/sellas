"""
REGAL (NCT04229979, SELLAS galinpepimut-S vs BAT in AML CR2) -- death-reporting-lag model.

Question: public event counts (60 / 72 / 78 of 80 "events") are *reported* deaths. If deaths in
the BAT arm are ascertained later than deaths in the GPS arm (BAT patients have no dosing visits,
only scheduled survival-status calls, quarterly to week 156 and yearly after), then
    reported(t) < true(t)  and the gap is concentrated in BAT.
Fitting survival curves to reported counts as if they were true counts then makes BAT look better
than it is. This script quantifies that with a patient-level Monte Carlo + ABC calibration.

Generative model (time in months from first patient, 2021-02-08)
  enrollment   : anchored to disclosed cumulative counts, 64 BAT / 63 GPS
  BAT          : Weibull parameterised by (median from randomisation M, 3-yr OS S36)
  GPS          : fraction c "durable responders" (BAT hazard until lag L, then low leak hazard),
                 rest follow BAT with hazard ratio theta after lag L
  reporting    : arrival = death + pipeline delay (+ discovery delay set by follow-up schedule)
  calibration  : kernel-ABC on reported counts 60@2024-12-10, 72@2025-12-26, 78@2026-05-11,
                 and <=79 reported at 2026-08-11 (and optionally 2026-10-05)
  analysis     : log-rank Z, Lan-DeMets O'Brien-Fleming boundaries at 60/80 events
"""
import datetime as dt
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq

T0 = dt.date(2021, 2, 8)


def m(d):
    """calendar date -> months since first patient"""
    return (d - T0).days / 30.4375


D = dict(
    nov23=m(dt.date(2023, 11, 29)),   # 105 enrolled
    mar24=m(dt.date(2024, 3, 26)),    # 123 enrolled
    apr24=m(dt.date(2024, 4, 29)),    # 127 enrolled (full)
    e60=m(dt.date(2024, 12, 10)),
    e72=m(dt.date(2025, 12, 26)),
    e78=m(dt.date(2026, 5, 11)),
    q2=m(dt.date(2026, 8, 11)),       # Q2 PR: "approaching" 80th, not announced
    now=m(dt.date(2026, 10, 5)),
)
N, N_BAT = 127, 64
GRID = np.arange(m(dt.date(2023, 1, 1)), m(dt.date(2027, 4, 1)), 1.0)
ANCH = [(D["e60"], 60), (D["e72"], 72), (D["e78"], 78)]

# ---------------------------------------------------------------- OBF boundaries (Lan-DeMets)
def obf_bounds(alpha=0.025, t1=60 / 80):
    a = lambda t: 2 * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(t)))
    a1 = a(t1)
    z1 = norm.ppf(1 - a1)
    cov = [[1, np.sqrt(t1)], [np.sqrt(t1), 1]]
    # P(Z1 < z1, Z2 > zf) = P(Z1<z1) - P(Z1<z1, Z2<=zf) must equal the alpha left after the interim
    mv = multivariate_normal(mean=[0, 0], cov=cov)
    g = lambda zf: norm.cdf(z1) - mv.cdf([z1, zf]) - (alpha - a1)
    zf = brentq(g, 1.0, 3.0)
    return z1, zf


Z1, ZF = obf_bounds()


# ---------------------------------------------------------------- scenario definitions
# q_bat / q_gps_off : prob a death is learned immediately by the site (clinical care, family,
#                     hospice records) instead of waiting for the next scheduled call.
# gps_on_imm        : prob a *responder* GPS death after month 12 is immediate (still on 6-weekly dosing)
# pipe_med          : median site->EDC->CRO verification delay (months), lognormal sigma 0.6
SCENARIOS = {
    "S0 no lag":                      dict(lag=False),
    "S1 admin lag only (symmetric)":  dict(lag=True, pipe_med=0.75, calls=False, q_bat=1.0, q_gps_off=1.0, gps_on_imm=1.0),
    "S2 protocol-literal calls":      dict(lag=True, pipe_med=0.75, calls=True,  q_bat=0.0, q_gps_off=0.0, gps_on_imm=0.95),
    "S3 protocol + real-world (50%)": dict(lag=True, pipe_med=0.75, calls=True,  q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95),
    "S4 BAT-adverse":                 dict(lag=True, pipe_med=1.0,  calls=True,  q_bat=0.2, q_gps_off=0.5, gps_on_imm=0.95),
    "S5 S3 + 5% BAT silent deaths":   dict(lag=True, pipe_med=0.75, calls=True,  q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, silent_bat=0.05, silent_gps=0.01),
    "S6 S3 + 10% BAT silent deaths":  dict(lag=True, pipe_med=0.75, calls=True,  q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, silent_bat=0.10, silent_gps=0.02),
    "S7 S3 + 20% BAT silent deaths":  dict(lag=True, pipe_med=0.75, calls=True,  q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, silent_bat=0.20, silent_gps=0.04),
}


# ---------------------------------------------------------------- simulation
def enroll(n, gamma, rng):
    j = np.arange(1, N + 1)[None, :]
    a = np.empty((n, N))
    u = (j[:, :105] - 1) / 104.0
    a[:, :105] = D["nov23"] * u ** (1.0 / gamma[:, None])
    a[:, 105:123] = D["nov23"] + (j[:, 105:123] - 105 - 0.5) / 18 * (D["mar24"] - D["nov23"])
    a[:, 123:] = D["mar24"] + (j[:, 123:] - 123 - 0.5) / 4 * (D["apr24"] - D["mar24"])
    a += rng.normal(0, 0.15, a.shape)
    a = np.clip(a, 0, D["apr24"])
    arm = (np.argsort(rng.random((n, N)), axis=1) >= N_BAT).astype(np.int8)  # 0 BAT, 1 GPS
    return a, arm


def weib_k(M, S36):
    with np.errstate(all="ignore"):
        k = np.log(-np.log(S36) / np.log(2)) / np.log(36.0 / M)
    return k


def sim_chunk(n, rng, scen, prior):
  with np.errstate(all="ignore"):
    return _sim_chunk(n, rng, scen, prior)


def _sim_chunk(n, rng, scen, prior):
    p = {}
    p["M"] = rng.uniform(*prior["M"], n)
    p["S36"] = rng.uniform(*prior["S36"], n)
    p["c"] = rng.uniform(*prior["c"], n)
    p["theta"] = rng.uniform(*prior["theta"], n)
    p["L"] = rng.uniform(*prior["L"], n)
    p["leak"] = rng.uniform(*prior["leak"], n)
    p["gamma"] = rng.uniform(*prior["gamma"], n)
    k = weib_k(p["M"], p["S36"])
    valid = np.isfinite(k) & (k > 0.3) & (k < 2.0)
    k = np.where(valid, k, 1.0)
    p["k"] = k
    M, kk = p["M"][:, None], k[:, None]
    a, arm = enroll(n, p["gamma"], rng)

    Hl = np.log(2) * (p["L"][:, None] / M) ** kk
    E = rng.exponential(1.0, (n, N))
    t_bat = M * (E / np.log(2)) ** (1 / kk)
    # GPS path
    resp = rng.random((n, N)) < p["c"][:, None]
    th = p["theta"][:, None]
    H_after = Hl + (E - Hl) / th
    t_unc = np.where(E < Hl, t_bat, M * (H_after / np.log(2)) ** (1 / kk))
    t_resp = np.where(E < Hl, t_bat, p["L"][:, None] + rng.exponential(1.0 / p["leak"][:, None], (n, N)))
    t_gps = np.where(resp, t_resp, t_unc)
    T = np.where(arm == 1, t_gps, t_bat)
    is_resp = resp & (arm == 1)
    d = a + T

    # ---- reporting delay
    if scen["lag"]:
        P = np.exp(np.log(scen["pipe_med"]) + 0.6 * rng.standard_normal((n, N)))
        if scen["calls"]:
            interval = np.where(T < 36, 3.0, 12.0)
            disc = rng.random((n, N)) * interval
            gps_imm = np.where(T < 12, 1.0, np.where(is_resp, scen["gps_on_imm"], scen["q_gps_off"]))
            q = np.where(arm == 1, gps_imm, scen["q_bat"])
            imm = rng.random((n, N)) < q
            disc = np.where(imm, 0.0, disc)
        else:
            disc = np.zeros((n, N))
            interval = np.zeros((n, N))
        R = P + disc
        # gap between last known-alive contact and death (used if death never reported by cutoff)
        gap = rng.random((n, N)) * np.where(T < 36, 3.0, 12.0) if scen["calls"] else rng.random((n, N)) * 1.0
    else:
        R = np.zeros((n, N))
        gap = np.zeros((n, N))
    arrive = d + R
    if scen.get("silent_bat") is not None:
        sil = np.where(arm == 1, scen["silent_gps"], scen["silent_bat"])
        silent = (rng.random((n, N)) < sil) & (T >= 12)
        arrive = np.where(silent, np.inf, arrive)       # death never ascertained before any cutoff
    return p, valid, a, arm, T, d, arrive, gap, is_resp


def logrank(time, event, arm):
    """arm 1 = GPS. returns z (negative => GPS better) and approx log HR (GPS vs BAT)."""
    idx = np.argsort(time, axis=1)
    ev = np.take_along_axis(event, idx, 1).astype(float)
    ag = np.take_along_axis(arm, idx, 1).astype(float)
    n = time.shape[1]
    ng_tot = ag.sum(1, keepdims=True)
    ng_before = np.cumsum(ag, 1) - ag
    n_g = ng_tot - ng_before
    n_all = n - np.arange(n)[None, :]
    n_b = n_all - n_g
    O = (ev * ag).sum(1)
    Ee = (ev * n_g / n_all).sum(1)
    V = (ev * n_g * n_b / n_all ** 2).sum(1)
    V = np.maximum(V, 1e-9)
    beta = (O - Ee) / V
    for _ in range(8):                                   # Newton steps on the Cox partial likelihood
        e = np.exp(beta)[:, None]
        pr = n_g * e / np.maximum(n_b + n_g * e, 1e-12)
        U = (ev * (ag - pr)).sum(1)
        I = np.maximum((ev * pr * (1 - pr)).sum(1), 1e-9)
        beta = np.clip(beta + U / I, -5, 5)
    return (O - Ee) / np.sqrt(V), beta


def analysis(cut, a, arm, T, d, arrive, gap, swept=True):
    """cut: (n,) calendar cutoff. swept=True -> every death <=cut known (CRO status sweep + back-dating)."""
    cut = cut[:, None]
    dead = d <= cut
    if swept:
        known = dead & np.isfinite(arrive)               # discovered deaths are back-dated; silent ones stay censored
        event = known
        time = np.where(known, T, np.where(dead, np.maximum(T - gap, 0.01), cut - a))
    else:
        rep = arrive <= cut
        event = rep
        time = np.where(rep, T, np.where(dead, np.maximum(T - gap, 0.01), cut - a))
    time = np.clip(time, 0.01, None)
    return logrank(time, event.astype(np.int8), arm), event.sum(1)


def run_scenario(name, scen, n_sim=1_000_000, chunk=20_000, seed=1, sigma=1.0,
                 prior=None, require_not_80_at_now=False, keep_thr=1e-3):
    prior = prior or dict(M=(6, 20), S36=(0.03, 0.45), c=(0.0, 0.65), theta=(0.3, 1.0),
                          L=(1, 8), leak=(0.002, 0.009), gamma=(1.4, 2.8))
    rng = np.random.default_rng(seed)
    rows, w_all, n_done = [], [], 0
    ev_sum, n_valid = 0.0, 0
    while n_done < n_sim:
        n = min(chunk, n_sim - n_done)
        p, valid, a, arm, T, d, arrive, gap, is_resp = sim_chunk(n, rng, scen, prior)
        n_done += n
        rep = lambda t: (arrive <= t).sum(1)
        w = np.ones(n)
        for t, k in ANCH:
            w *= np.exp(-0.5 * ((rep(t) - k) / sigma) ** 2)
        w *= (rep(D["q2"]) <= 79)
        if require_not_80_at_now:
            w *= (rep(D["now"]) <= 79)
        w *= valid
        ev_sum += w.sum()
        n_valid += n
        keep = w > keep_thr * max(w.max(), 1e-12) if w.max() > 0 else np.zeros(n, bool)
        keep &= w > 1e-6
        if not keep.any():
            continue
        s = lambda x: x[keep]
        a_, arm_, T_, d_, ar_, g_ = s(a), s(arm), s(T), s(d), s(arrive), s(gap)
        wk = w[keep]
        # reported & true event counts at anchors
        out = {k_: s(v) for k_, v in p.items()}
        out["w"] = wk
        for tag, t in [("60", D["e60"]), ("72", D["e72"]), ("78", D["e78"]), ("q2", D["q2"]), ("now", D["now"])]:
            out[f"rep_{tag}"] = (ar_ <= t).sum(1)
            out[f"true_{tag}"] = (d_ <= t).sum(1)
            out[f"trueBAT_{tag}"] = ((d_ <= t) & (arm_ == 0)).sum(1)
            out[f"repBAT_{tag}"] = ((ar_ <= t) & (arm_ == 0)).sum(1)
        out["aliveBAT_q2"] = ((d_ > D["q2"]) & (arm_ == 0)).sum(1)
        out["aliveGPS_q2"] = ((d_ > D["q2"]) & (arm_ == 1)).sum(1)
        # time of 80th reported / true event (calendar months)
        out["T80rep"] = np.partition(ar_, 79, axis=1)[:, 79]
        out["T80true"] = np.partition(d_, 79, axis=1)[:, 79]
        # final analysis at reported-80th cutoff
        cut = np.minimum(out["T80rep"], m(dt.date(2030, 1, 1)))
        (zc, lc), nev_c = analysis(cut, a_, arm_, T_, d_, ar_, g_, swept=True)
        (zu, lu), nev_u = analysis(cut, a_, arm_, T_, d_, ar_, g_, swept=False)
        out.update(z_final=zc, hr_final=np.exp(lc), ev_final=nev_c,
                   z_final_unswept=zu, hr_final_unswept=np.exp(lu), ev_final_unswept=nev_u)
        # interim analysis at 2024-12-10 cutoff (data as available then)
        cut_i = np.full(len(wk), D["e60"])
        (zi, li), _ = analysis(cut_i, a_, arm_, T_, d_, ar_, g_, swept=True)
        (zir, lir), nev_ir = analysis(cut_i, a_, arm_, T_, d_, ar_, g_, swept=False)
        out.update(z_int=zi, hr_int=np.exp(li), z_int_rep=zir, hr_int_rep=np.exp(lir))
        # true BAT / GPS survival summaries from the curves (analytic for BAT)
        M_, k_ = out["M"], out["k"]
        out["bat_S12"] = np.exp(-np.log(2) * (12 / M_) ** k_)
        out["bat_S60"] = np.exp(-np.log(2) * (60 / M_) ** k_)
        H = lambda t: np.log(2) * (t / M_) ** k_
        Hl_ = H(out["L"])
        unc = np.exp(-np.where(36 > out["L"], Hl_ + out["theta"] * (H(36.0) - Hl_), H(36.0)))
        resp36 = np.exp(-Hl_ - out["leak"] * np.clip(36 - out["L"], 0, None))
        out["gps_S36"] = out["c"] * resp36 + (1 - out["c"]) * unc
        out["bat_S36"] = out["S36"]
        grid = GRID
        out["gt"] = np.stack([(d_ <= g).sum(1) for g in grid], 1)
        out["gr"] = np.stack([(ar_ <= g).sum(1) for g in grid], 1)
        out["gtB"] = np.stack([((d_ <= g) & (arm_ == 0)).sum(1) for g in grid], 1)
        out["grB"] = np.stack([((ar_ <= g) & (arm_ == 0)).sum(1) for g in grid], 1)
        out["pooled_T"] = None
        rows.append({k_: v for k_, v in out.items() if v is not None})
    ev = ev_sum / n_valid
    keys = rows[0].keys()
    res = {k_: np.concatenate([r[k_] for r in rows]) for k_ in keys}
    res["evidence"] = ev
    res["name"] = name
    return res


# ---------------------------------------------------------------- weighted helpers
def wq(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(qs, c, x)


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())
