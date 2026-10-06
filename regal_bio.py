"""
REGAL biology-pinned model: two knobs, everything else from the literature.

KNOBS (the only things varied when running the model)
  bat_eff : relapse-hazard multiplier of ACTIVE BAT (azacitidine/decitabine/venetoclax/LDAC) vs observation. 1.0 = no benefit, 0.65 = QUAZAR-like
  gps_eff : fraction of GPS patients with a durable, cure-like immune effect (= P(immune response) x P(durable | response))

BIOLOGY (fixed, see BIO and BIO_SOURCES)  -- not fitted to the REGAL counts
  relapse process for an observation patient: declining Weibull hazard from CR2, calibrated to QUAZAR placebo relapse-free survival
  patient factors: CR1 duration, cytogenetics, CRp2 vs CR2, MRD, age, frailty
  CR2 -> randomisation delay (0-6 months, simulated left truncation), post-relapse survival, background mortality, transplant leakage
DESIGN (from the protocol)
  1:1 IWRS randomisation, STRATIFIED by CR1 duration x cytogenetics x CR2/CRp2 x MRD (16 strata), sequential permuted blocks
  primary analysis: Cox model stratified by the same factors, treatment only, one-sided 0.025, O'Brien-Fleming at 60/80 deaths
OBSERVATION: reported date = biological death date + max(0, N(mu, sd)) days per arm (default 0)
"""
import json, os, datetime as dt
import numpy as np
from scipy.optimize import least_squares
import regal_model as r

np.seterr(all="ignore")
LN2 = np.log(2)
N = r.N
NS = 16
BIO = dict(
    Mr=None, kr=None,                  # filled by calibrate_relapse()
    sigma_f=0.5, c_b=0.06,             # frailty SD; never-relapse fraction
    hr_long=0.55, hr_poor=1.6, hr_mrd=1.7, hr_crp=1.3, hr_age10=1.10,
    p_long=0.45, p_poor=0.30, p_mrd=0.40, p_crp=0.25, age_mean=66.0, age_sd=9.0,
    u_mean=2.5, m_p=5.5, hr_pr_age10=1.15, h65=0.0013, gomp=0.09, m_bg=1.5,
    p_sct=0.10, t_sct=8.0, sct_os36=0.5, p_act=0.75, th_dur=0.05, onset_med=3.0, onset_sd=0.4,
    gamma=1.9, block=4,
)
LAG = dict(mu_bat=0.0, sd_bat=0.0, mu_gps=0.0, sd_gps=0.0)       # days
RFS_TARGET = {6: 0.45, 12: 0.27}                                   # observation-arm relapse-free survival from randomisation (QUAZAR placebo)
BIO_SOURCES = {
    "Mr, kr": "calibrated so the simulated OBSERVATION arm has relapse-free survival 45% at 6 mo and 27% at 12 mo from randomisation (QUAZAR placebo: median 4.8 mo; CR2 durations 3-14 mo, Leopold/Remarkable-Big; relapse 40/17/2% in years 1-3)",
    "hr_long 0.55": "CR1 >18 mo vs <=6 mo: 1-yr OS 57% vs 14% (Breems 2005); Thalhammer 1996 as quoted on Reddit (unverified)",
    "hr_poor 1.6, hr_crp 1.3": "assumptions (generic AML prognosis; CRp/CRi carries more early failure)",
    "hr_mrd 1.7": "2-yr relapse 40% vs 24% for MRD+ vs MRD- in CR2 (EBMT, ln(0.60)/ln(0.76))",
    "hr_age10 1.10": "van der Maas validation cohort age 69 vs 58, 4-yr OS 9% vs 16%",
    "p_long .45, p_poor .30, p_mrd .40, p_crp .25": "assumptions (stratification factors; only age counts are public: 50 aged 18-64, 66 aged >=65)",
    "age N(66, 9)": "EU register age counts",
    "u_mean 2.5 (truncated at 6)": "protocol: consent within 6 months of CR2",
    "m_p 5.5 mo": "post-relapse survival 5.3 (Bataller), ven R/R 5.5-6.8, re-treatment 6.8 months",
    "h65 .0013/mo, +9%/yr, x1.5": "US life table (approximate, unverified) x excess mortality for AML survivors",
    "p_sct 0.10, post-SCT 3-yr OS 50%": "QUAZAR placebo 13.7%, oral aza 6.3% transplanted (as quoted on Reddit); transplant leaks in both arms (ITT)",
    "p_act .75": "EU protocol allows observation / HMA / venetoclax / LDAC; ~25% observation per community estimate (unverified)",
    "th_dur .05, onset 3 mo": "durable responders keep 5% of relapse hazard; WT1 CTL peak by week 12 (Hashii 2026), sipuleucel-T delayed separation ~6 mo, 80% of immunotherapy trials show >=3 mo delay",
    "gamma 1.9": "enrollment-curve shape implied by 20 by Apr 2022 and 38 by Oct 2022 (Reddit, unverified) with 105 by Nov 2023 (company PR)",
    "block 4": "IWRS permuted block size not published; 4 assumed",
}


# ---------------------------------------------------------------- common random numbers
def make_Z(R, seed=7):
    rng = np.random.default_rng(seed); sh = (R, N)
    Z = {k: rng.random(sh) for k in ["u_long", "u_poor", "u_mrd", "u_crp", "u_del", "u_cure", "u_sct", "u_act", "u_dur"]}
    Z.update({k: rng.standard_normal(sh) for k in ["z_age", "z_f", "z_L", "pipe"]})
    Z.update({k: rng.exponential(1.0, sh) for k in ["E_rel", "E_rem", "E_pr", "E_tsct", "E_post"]})
    Z["jit"] = rng.normal(0, 0.15, sh)
    Z["blockkeys"] = rng.random((R, NS, N // 2 + 2, 8))        # random keys for permuted blocks (up to block size 8)
    Z["R"] = R
    return Z


def enrollment_dates(gamma):
    j = np.arange(1, N + 1)
    a = np.empty(N); u = (j[:105] - 1) / 104.0
    a[:105] = r.D["nov23"] * u ** (1.0 / gamma)
    a[105:123] = r.D["nov23"] + (j[105:123] - 105 - 0.5) / 18 * (r.D["mar24"] - r.D["nov23"])
    a[123:] = r.D["mar24"] + (j[123:] - 123 - 0.5) / 4 * (r.D["apr24"] - r.D["mar24"])
    return a


def cohort(Z, bio):
    """patient covariates, strata, stratified permuted-block 1:1 arm assignment (all knob-independent), shapes (R,N)"""
    R = Z["R"]
    long_ = Z["u_long"] < bio["p_long"]; poor = Z["u_poor"] < bio["p_poor"]; mrd = Z["u_mrd"] < bio["p_mrd"]; crp = Z["u_crp"] < bio["p_crp"]
    age = np.clip(bio["age_mean"] + bio["age_sd"] * Z["z_age"], 30, 88)
    code = (long_ * 1 + poor * 2 + crp * 4 + mrd * 8).astype(int)                       # 16 strata
    oh = (code[:, :, None] == np.arange(NS)[None, None, :])
    k = (np.cumsum(oh, axis=1) - 1)
    k = np.take_along_axis(k, code[:, :, None], axis=2)[:, :, 0]                       # index of the patient within its stratum, in enrollment order
    B = bio["block"]; blk = k // B; pos = k % B
    keys = Z["blockkeys"][:, :, :, :B]                                                 # (R,NS,nblk,B)
    rank = np.argsort(np.argsort(keys, axis=-1), axis=-1)
    slot_gps = rank < (B // 2)
    ridx = np.arange(R)[:, None]
    arm = slot_gps[ridx, code, np.minimum(blk, keys.shape[2] - 1), pos]                # True = GPS
    return dict(long=long_, poor=poor, mrd=mrd, crp=crp, age=age, code=code, arm=arm)


# ---------------------------------------------------------------- simulation
def simulate(th_B, c_G, Z, bio, coh=None, a_dates=None, extra=None):
    """th_B, c_G: arrays (nk,). returns dict of (nk,R,N) arrays."""
    th_B = np.atleast_1d(np.asarray(th_B, float))[:, None, None]; c_G = np.atleast_1d(np.asarray(c_G, float))[:, None, None]
    ext = extra or {}
    g3 = lambda k, default: np.atleast_1d(np.asarray(ext.get(k, default), float))[:, None, None] if k in ext else default       # per-cell arrays or scalar default
    th_nc = g3("th_nc", 1.0); th_d = g3("th_d", bio["th_dur"]); onset = g3("onset_med", bio["onset_med"]); lam_b = g3("lam_b", 1.0); m_p = g3("m_p", bio["m_p"])
    coh = coh or cohort(Z, bio)
    ex = lambda x: x[None]
    arm = ex(coh["arm"]); age = ex(coh["age"])
    lam = 1.0 / bio["u_mean"]
    u = ex(-np.log(1 - Z["u_del"] * (1 - np.exp(-6 * lam))) / lam)
    Mr, kr = bio["Mr"], bio["kr"]
    H0 = lambda t: LN2 * (t / Mr) ** kr
    H0inv = lambda h: Mr * (np.maximum(h, 0) / LN2) ** (1 / kr)
    mult0 = np.exp(np.log(bio["hr_long"]) * (coh["long"] - bio["p_long"]) + np.log(bio["hr_poor"]) * (coh["poor"] - bio["p_poor"])
                   + np.log(bio["hr_mrd"]) * (coh["mrd"] - bio["p_mrd"]) + np.log(bio["hr_crp"]) * (coh["crp"] - bio["p_crp"])
                   + np.log(bio["hr_age10"]) * (coh["age"] - 66) / 10 + bio["sigma_f"] * Z["z_f"])
    active = (~arm) & ex(Z["u_act"] < bio["p_act"])
    mult = ex(mult0) * np.where(active, th_B, 1.0) * lam_b
    dur = arm & (ex(Z["u_dur"]) < c_G)
    E_tot = mult * H0(u) + ex(Z["E_rel"])
    b = u + onset * np.exp(bio["onset_sd"] * ex(Z["z_L"]))
    Hb = H0(b)
    t_plain = H0inv(E_tot / mult)
    th_r = np.where(dur, th_d, th_nc)                                                  # GPS effect after onset: durable -> th_d, otherwise partial -> th_nc
    t_gps = np.where(E_tot / mult < Hb, t_plain, H0inv(Hb + (E_tot / mult - Hb) / th_r))
    t_cr2 = np.where(arm, t_gps, t_plain)
    t_cr2 = np.where(ex(Z["u_cure"]) < bio["c_b"], np.inf, t_cr2)
    R_ = t_cr2 - u
    h_rem = bio["h65"] * np.exp(bio["gomp"] * (age - 65)) * bio["m_bg"]
    D_rem = ex(Z["E_rem"]) / h_rem
    P_post = ex(Z["E_pr"]) * (m_p / LN2) / np.exp(np.log(bio["hr_pr_age10"]) * (age - 66) / 10)
    died_rem = D_rem < R_
    T_nt = np.where(died_rem, D_rem, R_ + P_post)
    sct = ex(Z["u_sct"] < bio["p_sct"]); t_sct = ex(Z["E_tsct"]) * bio["t_sct"]
    sct_done = sct & (t_sct < T_nt)
    sct_rate = -np.log(bio["sct_os36"]) / 36.0
    T = np.where(sct_done, t_sct + ex(Z["E_post"]) / sct_rate, T_nt)
    a = np.clip((a_dates if a_dates is not None else enrollment_dates(bio["gamma"]))[None, None, :] + ex(Z["jit"]), 0, r.D["apr24"])
    d = a + T
    lag = np.where(arm, LAG["mu_gps"] + LAG["sd_gps"] * ex(Z["pipe"]), LAG["mu_bat"] + LAG["sd_bat"] * ex(Z["pipe"]))
    arrive = d + np.maximum(lag, 0) / 30.4375
    bt = lambda x: np.broadcast_to(x, T.shape)
    return dict(a=bt(a), arm=bt(arm), T=T, d=d, arrive=arrive, relapse_t=R_, died_rem=bt(died_rem), D_rem=bt(D_rem), sct=bt(sct_done), active=bt(active), dur=bt(dur), code=bt(ex(coh["code"])), age=bt(age))


# ---------------------------------------------------------------- stratified log-rank / Cox (protocol primary analysis)
def strat_stats(time, event, arm, strata, newton=8):
    """time,event,arm,strata: (n,N). returns z (negative = GPS better) and log HR from a Cox model stratified by `strata`, treatment only"""
    n = time.shape[0]
    idx = np.argsort(time, axis=1)
    ev = np.take_along_axis(event, idx, 1).astype(float); g = np.take_along_axis(arm, idx, 1).astype(float); s = np.take_along_axis(strata, idx, 1)
    NA = np.zeros(time.shape); NG = np.zeros(time.shape)
    for k in range(NS):
        mf = (s == k).astype(float)
        if not mf.any(): continue
        NA += mf * np.cumsum(mf[:, ::-1], axis=1)[:, ::-1]; NG += mf * np.cumsum((mf * g)[:, ::-1], axis=1)[:, ::-1]
    NB = NA - NG
    with np.errstate(all="ignore"):
        O = (ev * g).sum(1); E = (ev * NG / NA).sum(1); V = np.maximum((ev * NG * NB / NA ** 2).sum(1), 1e-9)
        beta = (O - E) / V
        for _ in range(newton):
            e = np.exp(beta)[:, None]; p = NG * e / np.maximum(NB + NG * e, 1e-12)
            U = (ev * (g - p)).sum(1); I = np.maximum((ev * p * (1 - p)).sum(1), 1e-9)
            beta = np.clip(beta + U / I, -5, 5)
    return (O - E) / np.sqrt(V), beta


def analyse(S, cut):
    """S from simulate(); cut: calendar cutoff scalar or (nk,R). returns z, hr, n_events, using deaths known by the cutoff"""
    nk, R, _ = S["T"].shape
    cut = np.broadcast_to(np.asarray(cut, float), (nk, R))[:, :, None]
    dead = S["d"] <= cut
    tm = np.clip(np.where(dead, S["T"], cut - S["a"]), 0.01, None)
    f = lambda x: x.reshape(nk * R, N)
    z, beta = strat_stats(f(tm), f(dead).astype(np.int8), f(S["arm"]).astype(np.int8), f(S["code"]))
    return z.reshape(nk, R), np.exp(beta).reshape(nk, R), dead.sum(-1)


# ---------------------------------------------------------------- calibration of the relapse process to observation-arm RFS
def calibrate_relapse(bio, R=1500, seed=3):
    Z = make_Z(R, seed); coh = cohort(Z, bio); a_dates = enrollment_dates(bio["gamma"])
    def rfs(x):
        b = dict(bio, Mr=x[0], kr=x[1])
        S = simulate(np.array([1.0]), np.array([0.0]), Z, b, coh, a_dates)       # observation patients only matter: use arm==BAT & not active
        mask = (~S["arm"][0]) & (~S["active"][0])
        relapse_or_death = np.minimum(S["relapse_t"][0], S["D_rem"][0])
        return np.array([(relapse_or_death[mask] > t).mean() for t in RFS_TARGET])
    res = least_squares(lambda x: rfs(x) - np.array(list(RFS_TARGET.values())), x0=[4.0, 0.7], bounds=([0.5, 0.3], [20, 1.5]), diff_step=0.05)
    return float(res.x[0]), float(res.x[1]), rfs(res.x)


_cal = "calib_bio.json"
if os.path.exists(_cal):
    BIO["Mr"], BIO["kr"] = json.load(open(_cal))["Mr"], json.load(open(_cal))["kr"]
