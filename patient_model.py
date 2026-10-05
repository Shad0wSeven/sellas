"""
REGAL patient-level model v3.

Each of the 127 patients is simulated as an individual:
  covariates  : CR1 duration long/short, poor cytogenetics, MRD+, age (REGAL registry: 50 aged 18-64, 66 aged >=65 -> mean ~66, sd ~9)
  randomisation: stratified 1:1 within strata (CR1 duration x cytogenetics x MRD) by alternation after sorting on stratum
  clock       : relapse-free time is measured from CR2; a patient enters REGAL u in [0,6] months after CR2 and must be relapse-free
                and alive at u (left truncation is simulated, not assumed)
  relapse     : baseline Weibull hazard (declining), patient multiplier exp(b.x + frailty); never-relapse fraction c_b
  BAT         : ~p_act of patients on active therapy (HMA/ven/LDAC) with relapse-hazard multiplier th_B; the rest observation
  GPS         : patient is an immune responder with prob p_resp; responders get relapse-hazard multiplier th_r after a
                patient-specific onset delay (median L, lognormal); non-responders get th_nr (~observation, no maintenance)
  death       : in remission (age-dependent background x m_bg) or after relapse (post-relapse survival, age-dependent)
  transplant  : prob p_sct in BOTH arms (ITT); after SCT survival has 3-yr OS 50%
  reporting   : GPS patients still in remission are on dosing visits (death known at once); others follow quarterly->yearly calls
                with a real-world-discovery probability; plus pipeline delay
Interim: soft likelihood term P(IDMC continues | theta) with uncertain efficacy bound z_b and futility HR bound h_f.
"""
import datetime as dt
import numpy as np
import regal_model as r
import regal_model_v2 as v

np.seterr(all="ignore")
LN2 = np.log(2)
N, N_BAT = r.N, r.N_BAT

# name, lo, hi, kind, a, b     kind 'n': normal(a,b); 'u': uniform; 'e': exponential(mean a)
PARAMS = [
    ("Mr", 1.5, 10, "n", 3.0, 0.9), ("kr", 0.5, 1.4, "n", 0.85, 0.2), ("c_b", 0.0, 0.3, "n", 0.08, 0.05),
    ("b_long", -1.6, 0.2, "n", np.log(0.55), 0.25), ("b_poor", -0.2, 1.2, "n", np.log(1.6), 0.25),
    ("b_mrd", -0.2, 1.0, "n", np.log(1.4), 0.25), ("b_age", -0.1, 0.5, "n", np.log(1.1), 0.1), ("sig_f", 0.1, 1.0, "n", 0.5, 0.2),
    ("p_long", 0.2, 0.7, "n", 0.45, 0.10), ("p_poor", 0.1, 0.5, "n", 0.30, 0.08), ("p_mrd", 0.15, 0.65, "n", 0.40, 0.10),
    ("u_mean", 1.0, 4.0, "n", 2.5, 0.8), ("m_p", 3.5, 8.0, "n", 5.5, 1.0), ("b_prage", -0.1, 0.5, "n", np.log(1.15), 0.1),
    ("m_bg", 0.8, 3.0, "n", 1.5, 0.4), ("p_sct", 0.02, 0.25, "n", 0.10, 0.04),
    ("p_act", 0.4, 0.95, "n", 0.75, 0.10), ("th_B", 0.4, 1.1, "n", 0.75, 0.15),
    ("p_resp", 0.35, 0.95, "n", 0.70, 0.10), ("lth_r", np.log(0.03), 0.0, "n", np.log(0.4), 0.6),
    ("th_nr", 0.7, 1.4, "n", 1.0, 0.15), ("lL", 0.0, np.log(8.0), "n", np.log(3.0), 0.3),
    ("gamma", 1.4, 2.8, "u", 0, 0), ("delta", 0.0, 2.5, "e", 1.0, 0), ("q_rw", 0.0, 1.0, "n", 0.5, 0.2),
    ("z_b", 2.2, 3.0, "n", 2.4, 0.15), ("h_f", 0.74, 0.92, "u", 0, 0),
]
import os, json
LAG = dict(mu_bat=0.0, sd_bat=0.0, mu_gps=0.0, sd_gps=0.0)       # reporting lag in DAYS added to the biological death date: N(mu, sd), clipped at 0
LAG.update(json.loads(os.environ.get("PM_LAG", "{}")))
_ov = json.loads(os.environ.get("PM_OVERRIDE", "{}"))           # e.g. {"lth_r": [-1.9, 0.6]} overrides prior (a, b) for sensitivity runs
PARAMS = [(n, lo, hi, k, *_ov.get(n, (a, b))) for (n, lo, hi, k, a, b) in PARAMS]
PARAMS = [p for p in PARAMS if p[0] not in ("q_rw", "delta")]      # no reporting nuisance parameters: lag is an explicit input (LAG)
NAMES = [p[0] for p in PARAMS]; IDX = {n: i for i, n in enumerate(NAMES)}; D = len(PARAMS)
LO = np.array([p[1] for p in PARAMS]); HI = np.array([p[2] for p in PARAMS])
KIND = [p[3] for p in PARAMS]; PA = np.array([p[4] for p in PARAMS], float); PB = np.array([p[5] for p in PARAMS], float)
H65 = 0.0013            # monthly background hazard at age 65 (~1.5%/yr, US life table); Gompertz slope 0.09/yr
SCT_RATE = -np.log(0.5) / 36.0   # post-transplant survival: 3-yr OS 50%


def log_prior(X):
    inbox = np.all((X >= LO) & (X <= HI), axis=1)
    lp = np.zeros(len(X))
    for j, k in enumerate(KIND):
        if k == "n": lp += -0.5 * ((X[:, j] - PA[j]) / PB[j]) ** 2
        elif k == "e": lp += -X[:, j] / PA[j]
    return np.where(inbox, lp, -np.inf)


def sample_prior(n, rng):
    """independent draws from the (truncated) prior"""
    X = np.empty((n, D))
    for j, k in enumerate(KIND):
        if k == "n":
            x = rng.normal(PA[j], PB[j], n * 4); x = x[(x >= LO[j]) & (x <= HI[j])][:n]
            while len(x) < n: x = np.concatenate([x, rng.normal(PA[j], PB[j], n)]); x = x[(x >= LO[j]) & (x <= HI[j])][:n]
        elif k == "e":
            x = rng.exponential(PA[j], n * 6); x = x[x <= HI[j]][:n]
        else:
            x = rng.uniform(LO[j], HI[j], n)
        X[:, j] = x
    return X


def make_Z(R, seed=2024):
    rng = np.random.default_rng(seed); sh = (R, N)
    keys = ["u_long", "u_poor", "u_mrd", "z_age", "u_del", "z_f", "E_rel", "u_cure", "E_rem", "E_pr", "u_sct", "E_tsct", "E_post",
            "u_act", "u_resp", "z_L", "u_imm", "u_disc", "key", "jit"]
    Z = {k: rng.random(sh) for k in keys}
    for k in ("z_age", "z_f", "z_L", "pipe"): Z[k] = rng.standard_normal(sh)
    for k in ("E_rel", "E_rem", "E_pr", "E_tsct", "E_post"): Z[k] = rng.exponential(1.0, sh)
    Z["jit"] = rng.normal(0, 0.15, sh); Z["flip"] = rng.random((R, 1)); Z["R"] = R
    return Z


def enroll_base(gamma):
    n = len(gamma); j = np.arange(1, N + 1)[None, :]
    a = np.empty((n, N)); u = (j[:, :105] - 1) / 104.0
    a[:, :105] = r.D["nov23"] * u ** (1.0 / gamma[:, None])
    a[:, 105:123] = r.D["nov23"] + (j[:, 105:123] - 105 - 0.5) / 18 * (r.D["mar24"] - r.D["nov23"])
    a[:, 123:] = r.D["mar24"] + (j[:, 123:] - 123 - 0.5) / 4 * (r.D["apr24"] - r.D["mar24"])
    return a


def sim_block(X, Z):
    """X (nb,D). returns dict of arrays (nb,R,N): a, arm, T, d, arrive, relapsed, ... (all months)"""
    nb = len(X); P = {n: X[:, i][:, None, None] for n, i in IDX.items()}
    ex = lambda k: Z[k][None]
    long_ = ex("u_long") < P["p_long"]; poor = ex("u_poor") < P["p_poor"]; mrd = ex("u_mrd") < P["p_mrd"]
    age = np.clip(66 + 9 * ex("z_age"), 30, 88)
    # stratified 1:1 assignment (alternation after sorting on stratum code + random key)
    code = long_ * 4 + poor * 2 + mrd
    order = np.argsort(code + 0.9 * ex("key"), axis=-1)
    pos = np.empty_like(order); np.put_along_axis(pos, order, np.arange(N)[None, None, :], axis=-1)
    flip = (Z["flip"][None] > 0.5)
    arm = ((pos + flip) % 2).astype(bool)               # True = GPS
    # clock
    lam = 1.0 / P["u_mean"]
    u = -np.log(1 - ex("u_del") * (1 - np.exp(-6 * lam))) / lam                       # truncated exponential on [0,6] months after CR2
    Mr, kr = P["Mr"], P["kr"]
    H0 = lambda t: LN2 * (t / Mr) ** kr
    H0inv = lambda h: Mr * (np.maximum(h, 0) / LN2) ** (1 / kr)
    mult = np.exp(P["b_long"] * (long_ - P["p_long"]) + P["b_poor"] * (poor - P["p_poor"]) + P["b_mrd"] * (mrd - P["p_mrd"])
                  + P["b_age"] * (age - 66) / 10 + P["sig_f"] * ex("z_f"))
    active = (~arm) & (ex("u_act") < P["p_act"])
    resp = arm & (ex("u_resp") < P["p_resp"])
    mult = mult * np.where(active, P["th_B"], 1.0) * np.where(arm & ~resp, P["th_nr"], 1.0)
    E_tot = mult * H0(u) + ex("E_rel")                                                 # conditional on relapse-free at entry
    Lp = np.exp(P["lL"] + 0.4 * ex("z_L")); b = u + Lp
    th_r = np.exp(P["lth_r"])
    Hb = H0(b)
    t_plain = H0inv(E_tot / mult)
    t_resp = np.where(E_tot / mult < Hb, t_plain, H0inv(Hb + (E_tot / mult - Hb) / th_r))
    t_cr2 = np.where(resp, t_resp, t_plain)
    t_cr2 = np.where(ex("u_cure") < P["c_b"], np.inf, t_cr2)
    R_ = t_cr2 - u                                                                     # relapse time from randomisation
    h_rem = H65 * np.exp(0.09 * (age - 65)) * P["m_bg"]
    D_rem = ex("E_rem") / h_rem
    P_post = ex("E_pr") * (P["m_p"] / LN2) / np.exp(P["b_prage"] * (age - 66) / 10)
    died_rem = D_rem < R_
    T_nt = np.where(died_rem, D_rem, R_ + P_post)
    sct = ex("u_sct") < P["p_sct"]; t_sct = ex("E_tsct") * 8.0
    sct_done = sct & (t_sct < T_nt)
    T = np.where(sct_done, t_sct + ex("E_post") / SCT_RATE, T_nt)
    relapsed_by_death = (~died_rem) & (~sct_done)
    on_visits = (died_rem & ~sct_done) | False          # death while still in remission and not transplanted
    a = np.clip(enroll_base(X[:, IDX["gamma"]])[:, None, :] + ex("jit"), 0, r.D["apr24"])
    d = a + T
    # reporting: biological death date + Gaussian lag (days), separate mean/sd for BAT and GPS
    lag_days = np.where(arm, LAG["mu_gps"] + LAG["sd_gps"] * ex("pipe"), LAG["mu_bat"] + LAG["sd_bat"] * ex("pipe"))
    arrive = d + np.maximum(lag_days, 0.0) / 30.4375
    interval = np.where(T < 36, 3.0, 12.0)
    return dict(a=a, arm=arm, T=T, d=d, arrive=arrive, relapsed=relapsed_by_death, resp=resp, active=active, sct=sct_done, age=np.broadcast_to(age, T.shape), long=np.broadcast_to(long_, T.shape), interval=interval, u=u, R=R_)


def summarize_block(X, Z):
    """expected counts, P_stall, fu median, P_cont (soft interim), valid"""
    s = sim_block(X, Z); nb = len(X); R = Z["R"]
    dl = np.zeros(nb)
    dates = np.stack([np.full(nb, r.D["e60"]), r.D["e72"] - dl, np.full(nb, r.D["e78"])], 1)
    mu = np.stack([(s["arrive"] <= dates[:, i][:, None, None]).sum(-1).mean(-1) for i in range(3)], 1)
    pst = ((s["arrive"] <= r.D["q2"]).sum(-1) <= 79).mean(-1)
    fu = np.median(np.minimum(s["T"], r.D["e60"] - s["a"]), axis=-1).mean(-1)
    # interim (swept): events by e60, log-rank z and Cox HR
    dead = s["d"] <= r.D["e60"]
    tm = np.clip(np.where(dead, s["T"], r.D["e60"] - s["a"]), 0.01, None).reshape(nb * R, N)
    z, beta = r.logrank(tm, dead.reshape(nb * R, N).astype(np.int8), s["arm"].reshape(nb * R, N).astype(np.int8))
    z_eff = -z.reshape(nb, R); hr = np.exp(beta.reshape(nb, R))
    cont = (z_eff < X[:, IDX["z_b"]][:, None]) & (hr <= X[:, IDX["h_f"]][:, None])
    p_eff_stop = (z_eff >= X[:, IDX["z_b"]][:, None]).mean(1); p_fut_stop = (hr > X[:, IDX["h_f"]][:, None]).mean(1)
    return mu, pst, fu, cont.mean(1), p_eff_stop, p_fut_stop


# ---- parallel evaluation
_Z = None
def _init(R, seed):
    global _Z; _Z = make_Z(R, seed)

def _work(Xc):
    out = [summarize_block(Xc[i:i + 40], _Z) for i in range(0, len(Xc), 40)]
    return [np.concatenate([o[k] for o in out]) for k in range(6)]

class Evaluator:
    def __init__(self, R=120, seed=2024, workers=8):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(R, seed)); self.workers = workers
    def __call__(self, X):
        chunks = np.array_split(X, self.workers * 3)
        res = list(self.pool.map(_work, [c for c in chunks if len(c)]))
        return [np.concatenate([q[k] for q in res]) for k in range(6)]
    def close(self): self.pool.shutdown()
