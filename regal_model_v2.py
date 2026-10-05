"""
REGAL model v2: adds model choices surfaced in the r/sellaslifesciences community review.

New vs v1 (regal_model.py):
  1. GPS non-responders may do WORSE than BAT (theta up to 1.3): GPS arm gets no active maintenance, ~75% of BAT does.
  2. Withdrawal / loss-to-follow-up (WCLFU), arm-specific annual rates (open-label => BAT withdrawals higher).
  3. Alternative follow-up schedules (quarterly to week 91 [AverageUnited quote] vs to week 156 [CW quote]; quarterly forever).
  4. Two-stage structure (relapse-free time, then post-relapse survival identical in both arms): the effect on OS
     is compressed relative to the effect on RFS (Calm_Ad / Remarkable-Big critique). BAT OS is *derived*, not assumed.
  5. Extra stored diagnostics: median observed follow-up at IA (company reported 13.5 mo), IA deaths by arm
     (for alternative IA constraints used by Stochasty / Thetamancer / Grand_Effort), enrollment median,
     and a hold-out "stall" likelihood (calibrate on 60/72/78 only, then ask P(80th not reported by 11 Aug)).
"""
import datetime as dt
import numpy as np
from regal_model import (D, m, N, N_BAT, ANCH, Z1, ZF, GRID, T0, logrank, enroll, weib_k, wq, wmean, ess)

LN2 = np.log(2)

SCENARIOS_V2 = {
    "V0 no lag":                              dict(lag=False),
    "V3 calls to wk156 + real-world 50%":     dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95),
    "V2 calls to wk156, no real-world":       dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.0, q_gps_off=0.0, gps_on_imm=0.95),
    "V8 calls to wk91 then yearly, rw 50%":   dict(lag=True, pipe_med=0.75, calls=True, q_until=21, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95),
    "V9 quarterly throughout, rw 50%":        dict(lag=True, pipe_med=0.75, calls=True, q_until=999, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95),
    "V10 V3 + LTFU BAT 6%/GPS 2% per yr":     dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, ltfu_bat=0.06, ltfu_gps=0.02),
    "V11 V3 + LTFU symmetric 3% per yr":      dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, ltfu_bat=0.03, ltfu_gps=0.03),
    "V12 V10 + 10% BAT silent deaths":        dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, ltfu_bat=0.06, ltfu_gps=0.02, silent_bat=0.10, silent_gps=0.02),
}
# two-stage structure versions
SCENARIOS_2S_LIT = {
    "TL0 two-stage (lit-compatible prior), no lag":      dict(lag=False, structure="two", prior_name="lit"),
    "TL3 two-stage (lit-compatible), calls+real-world":  dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, structure="two", prior_name="lit"),
    "TL10 two-stage (lit-compatible) + LTFU 6%/2%":      dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, ltfu_bat=0.06, ltfu_gps=0.02, structure="two", prior_name="lit"),
}
SCENARIOS_2S = {
    "T0 two-stage, no lag":                   dict(lag=False, structure="two"),
    "T3 two-stage, calls+real-world":         dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, structure="two"),
    "T10 two-stage + LTFU BAT 6%/GPS 2%":     dict(lag=True, pipe_med=0.75, calls=True, q_until=36, q_bat=0.5, q_gps_off=0.5, gps_on_imm=0.95, ltfu_bat=0.06, ltfu_gps=0.02, structure="two"),
}

PRIOR_1S = dict(M=(6, 20), S36=(0.03, 0.45), c=(0.0, 0.65), theta=(0.3, 1.3), L=(1, 8), leak=(0.002, 0.009), gamma=(1.4, 2.8))
PRIOR_2S_LIT = dict(Mr=(3, 12), kr=(0.6, 1.4), c_b=(0.0, 0.15), c_g=(0.0, 0.65), theta=(0.3, 1.3), L=(1, 8),
                    m_p=(3.5, 7.0), h=(0.001, 0.006), gamma=(1.4, 2.8))
PRIOR_2S = dict(Mr=(5, 16), kr=(0.6, 1.4), c_b=(0.0, 0.25), c_g=(0.0, 0.65), theta=(0.3, 1.3), L=(1, 8),
                m_p=(3.5, 8.0), h=(0.001, 0.006), gamma=(1.4, 2.8))


# ------------------------------------------------------------------------ generators
def gen_one_stage(n, rng, prior):
    p = {k: rng.uniform(*prior[k], n) for k in ["M", "S36", "c", "theta", "L", "leak", "gamma"]}
    k = weib_k(p["M"], p["S36"])
    valid = np.isfinite(k) & (k > 0.3) & (k < 2.0)
    k = np.where(valid, k, 1.0)
    p["k"] = k
    M, kk = p["M"][:, None], k[:, None]
    a, arm = enroll(n, p["gamma"], rng)
    Hl = LN2 * (p["L"][:, None] / M) ** kk
    E = rng.exponential(1.0, (n, N))
    t_bat = M * (E / LN2) ** (1 / kk)
    resp = rng.random((n, N)) < p["c"][:, None]
    H_after = Hl + (E - Hl) / p["theta"][:, None]
    t_unc = np.where(E < Hl, t_bat, M * (H_after / LN2) ** (1 / kk))
    t_resp = np.where(E < Hl, t_bat, p["L"][:, None] + rng.exponential(1.0 / p["leak"][:, None], (n, N)))
    t_gps = np.where(resp, t_resp, t_unc)
    T = np.where(arm == 1, t_gps, t_bat)
    # responders = durable (c) component
    return p, valid, a, arm, T, resp & (arm == 1)


def gen_two_stage(n, rng, prior):
    p = {k: rng.uniform(*prior[k], n) for k in ["Mr", "kr", "c_b", "c_g", "theta", "L", "m_p", "h", "gamma"]}
    valid = np.ones(n, bool)
    a, arm = enroll(n, p["gamma"], rng)
    Mr, kr = p["Mr"][:, None], p["kr"][:, None]
    Hl = LN2 * (p["L"][:, None] / Mr) ** kr
    E = rng.exponential(1.0, (n, N))
    R_bat = Mr * (E / LN2) ** (1 / kr)
    H_after = Hl + (E - Hl) / p["theta"][:, None]
    R_unc_g = np.where(E < Hl, R_bat, Mr * (H_after / LN2) ** (1 / kr))
    cured_b = rng.random((n, N)) < p["c_b"][:, None]
    cured_g = rng.random((n, N)) < p["c_g"][:, None]
    R_g = np.where(cured_g & (E >= Hl), np.inf, R_unc_g)
    R_b = np.where(cured_b, np.inf, R_bat)
    R = np.where(arm == 1, R_g, R_b)
    D_rem = rng.exponential(1.0 / p["h"][:, None], (n, N))
    P = rng.exponential((p["m_p"] / LN2)[:, None], (n, N))
    T = np.where(D_rem < R, D_rem, R + P)
    is_resp = cured_g & (arm == 1)
    return p, valid, a, arm, T, is_resp


def two_stage_curves(p, which, t=None):
    """OS survival S(t) on a grid for a two-stage parameter set (vectorised over sims)."""
    if t is None:
        t = np.arange(0.0, 96.01, 0.5)
    dtt = t[1] - t[0]
    Mr, kr = p["Mr"][:, None], p["kr"][:, None]
    H = LN2 * (t[None, :] / Mr) ** kr
    h = p["h"][:, None]
    if which == "bat":
        SR = p["c_b"][:, None] + (1 - p["c_b"][:, None]) * np.exp(-H)
    else:
        L = p["L"][:, None]
        Hl = LN2 * (L / Mr) ** kr
        Hg = np.where(t[None, :] < L, H, Hl + p["theta"][:, None] * (H - Hl))
        cur = p["c_g"][:, None]
        S_c = np.where(t[None, :] < L, np.exp(-H), np.exp(-Hl))
        SR = cur * S_c + (1 - cur) * np.exp(-Hg)
    mass = np.maximum(SR[:, :-1] - SR[:, 1:], 0.0)            # relapse probability mass per step
    tau = (p["m_p"] / LN2)[:, None]
    G = np.zeros_like(SR)
    for i in range(SR.shape[1] - 1):
        tmid = t[i] + dtt / 2
        G[:, i + 1] = G[:, i] * np.exp(-dtt / tau[:, 0]) + mass[:, i] * np.exp(-h[:, 0] * tmid) * np.exp(-dtt / (2 * tau[:, 0]))
    return t, np.exp(-h * t[None, :]) * SR + G


def curve_stats(t, S):
    s12, s36, s60 = S[:, np.searchsorted(t, 12)], S[:, np.searchsorted(t, 36)], S[:, np.searchsorted(t, 60)]
    # median: first grid point where S <= 0.5 (linear interpolation)
    idx = np.argmax(S <= 0.5, axis=1)
    idx = np.clip(idx, 1, S.shape[1] - 1)
    s_hi, s_lo = S[np.arange(len(S)), idx - 1], S[np.arange(len(S)), idx]
    frac = np.where(s_hi > s_lo, (s_hi - 0.5) / np.maximum(s_hi - s_lo, 1e-12), 0)
    med = t[idx - 1] + frac * (t[1] - t[0])
    med = np.where(S[:, -1] > 0.5, np.inf, med)
    return s12, s36, s60, med


# ------------------------------------------------------------------------ reporting layer
def add_reporting(n, rng, scen, a, arm, T, d, is_resp):
    N_ = T.shape[1]
    W = np.full((n, N_), np.inf)
    if scen.get("ltfu_bat") is not None:
        rate = np.where(arm == 1, scen["ltfu_gps"], scen["ltfu_bat"])
        haz = -np.log(1 - rate) / 12.0
        W = rng.exponential(1.0, (n, N_)) / np.maximum(haz, 1e-12)
        W = np.where(haz > 0, W, np.inf)
    if scen["lag"]:
        P = np.exp(np.log(scen["pipe_med"]) + 0.6 * rng.standard_normal((n, N_)))
        if scen["calls"]:
            qu = scen.get("q_until", 36)
            interval = np.where(T < qu, 3.0, 12.0)
            disc = rng.random((n, N_)) * interval
            gps_imm = np.where(T < 12, 1.0, np.where(is_resp, scen["gps_on_imm"], scen["q_gps_off"]))
            q = np.where(arm == 1, gps_imm, scen["q_bat"])
            disc = np.where(rng.random((n, N_)) < q, 0.0, disc)
            gap = rng.random((n, N_)) * interval
        else:
            disc = np.zeros((n, N_)); gap = rng.random((n, N_)) * 1.0
        R = P + disc
    else:
        R = np.zeros((n, N_)); gap = np.zeros((n, N_))
    if scen.get("holiday"):                      # extra site/CRO delay for deaths around the Dec-2025 holidays (CW, Thetamancer)
        h0, h1 = m(dt.date(2025, 12, 10)), m(dt.date(2026, 1, 5))
        R = R + np.where((d >= h0) & (d <= h1), rng.uniform(0, 1.5, (n, N_)), 0.0)
    arrive = d + R
    lost = T > W
    arrive = np.where(lost, np.inf, arrive)
    if scen.get("silent_bat") is not None:
        sil = np.where(arm == 1, scen["silent_gps"], scen["silent_bat"])
        arrive = np.where((rng.random((n, N_)) < sil) & (T >= 12), np.inf, arrive)
    return arrive, gap, W


def analysis2(cut, a, arm, T, d, arrive, gap, W, swept=True):
    cut = cut[:, None]
    dead = d <= cut
    lost = T > W
    known = (dead & ~lost & np.isfinite(arrive)) if swept else ((arrive <= cut) & ~lost)
    dead_unknown = dead & ~lost & ~known
    time = np.where(known, T, np.where(lost, np.minimum(W, cut - a),
                    np.where(dead_unknown, np.maximum(T - gap, 0.01), cut - a)))
    time = np.clip(time, 0.01, None)
    return logrank(time, known.astype(np.int8), arm), known.sum(1)


# ------------------------------------------------------------------------ driver
def run_v2(name, scen, n_sim=3_000_000, chunk=20_000, seed=1, sigma=1.0, keep_thr=1e-3):
    two = scen.get("structure") == "two"
    prior = (PRIOR_2S_LIT if scen.get("prior_name") == "lit" else PRIOR_2S) if two else PRIOR_1S
    rng = np.random.default_rng(seed)
    rows = []
    sum_w_ns = sum_w_ns_stall = sum_w = 0.0
    n_done = n_tot = 0
    while n_done < n_sim:
        n = min(chunk, n_sim - n_done)
        with np.errstate(all="ignore"):
            gen = gen_two_stage if two else gen_one_stage
            res = gen(n, rng, prior)
            if two:
                p, valid, a, arm, T, is_resp = res
            else:
                p, valid, a, arm, T, is_resp = res
            d = a + T
            arrive, gap, W = add_reporting(n, rng, scen, a, arm, T, d, is_resp)
        n_done += n
        rep = lambda t: (arrive <= t).sum(1)
        w_ns = np.ones(n)
        for t, k in ANCH:
            w_ns *= np.exp(-0.5 * ((rep(t) - k) / sigma) ** 2)
        w_ns *= valid
        stall = rep(D["q2"]) <= 79
        w = w_ns * stall
        sum_w_ns += w_ns.sum(); sum_w_ns_stall += (w_ns * stall).sum(); n_tot += n
        keep = (w_ns > keep_thr * max(w_ns.max(), 1e-12)) & (w_ns > 1e-6)
        if not keep.any():
            continue
        s = lambda x: x[keep]
        a_, arm_, T_, d_, ar_, g_, W_ = s(a), s(arm), s(T), s(d), s(arrive), s(gap), s(W)
        out = {k_: s(v) for k_, v in p.items()}
        out["w"] = w[keep]; out["w_ns"] = w_ns[keep]
        for tag, t in [("60", D["e60"]), ("72", D["e72"]), ("78", D["e78"]), ("q2", D["q2"]), ("now", D["now"])]:
            out[f"rep_{tag}"] = (ar_ <= t).sum(1); out[f"true_{tag}"] = (d_ <= t).sum(1)
            out[f"trueBAT_{tag}"] = ((d_ <= t) & (arm_ == 0)).sum(1); out[f"repBAT_{tag}"] = ((ar_ <= t) & (arm_ == 0)).sum(1)
        out["aliveBAT_q2"] = ((d_ > D["q2"]) & (arm_ == 0)).sum(1)
        out["aliveGPS_q2"] = ((d_ > D["q2"]) & (arm_ == 1)).sum(1)
        out["T80rep"] = np.partition(ar_, 79, axis=1)[:, 79]
        out["T80true"] = np.partition(d_, 79, axis=1)[:, 79]
        out["enrol_median"] = np.median(a_, axis=1)
        cut = np.minimum(out["T80rep"], m(dt.date(2030, 1, 1)))
        (zc, lc), nev_c = analysis2(cut, a_, arm_, T_, d_, ar_, g_, W_, True)
        (zu, lu), nev_u = analysis2(cut, a_, arm_, T_, d_, ar_, g_, W_, False)
        out.update(z_final=zc, hr_final=np.exp(lc), ev_final=nev_c, z_final_unswept=zu, hr_final_unswept=np.exp(lu))
        cut_i = np.full(len(out["w"]), D["e60"])
        (zi, li), _ = analysis2(cut_i, a_, arm_, T_, d_, ar_, g_, W_, True)
        out.update(z_int=zi, hr_int=np.exp(li))
        # IA diagnostics (swept): deaths by arm, median observed follow-up (company reported 13.5 mo)
        dead_i = (d_ <= D["e60"]) & (T_ <= W_)
        out["ia_deaths"] = dead_i.sum(1); out["ia_deaths_bat"] = (dead_i & (arm_ == 0)).sum(1)
        obs_i = np.minimum(np.minimum(T_, W_), D["e60"] - a_)
        out["fu_med_ia"] = np.median(obs_i, axis=1)
        # truth curves
        if two:
            t, Sb = two_stage_curves(out, "bat"); _, Sg = two_stage_curves(out, "gps")
            s12, s36, s60, med = curve_stats(t, Sb)
            out["bat_S12"], out["S36"], out["bat_S60"], out["M"] = s12, s36, s60, med
            out["gps_S36"] = Sg[:, np.searchsorted(t, 36)]
            out["k"] = np.full(len(s36), 0.85); out["c"] = out["c_g"]
        else:
            M_, k_ = out["M"], out["k"]
            H = lambda tt: LN2 * (tt / M_) ** k_
            out["bat_S12"] = np.exp(-H(12.0)); out["bat_S60"] = np.exp(-H(60.0))
            Hl_ = H(out["L"])
            unc = np.exp(-np.where(36 > out["L"], Hl_ + out["theta"] * (H(36.0) - Hl_), H(36.0)))
            resp36 = np.exp(-Hl_ - out["leak"] * np.clip(36 - out["L"], 0, None))
            out["gps_S36"] = out["c"] * resp36 + (1 - out["c"]) * unc
        out["gt"] = np.stack([(d_ <= g).sum(1) for g in GRID], 1)
        out["gr"] = np.stack([(ar_ <= g).sum(1) for g in GRID], 1)
        rows.append(out)
    keys = rows[0].keys()
    res = {k_: np.concatenate([r[k_] for r in rows]) for k_ in keys if k_ in rows[0]}
    res["p_stall_given_78"] = sum_w_ns_stall / max(sum_w_ns, 1e-300)
    res["evidence_ns"] = sum_w_ns / n_tot
    res["name"] = name
    return res
