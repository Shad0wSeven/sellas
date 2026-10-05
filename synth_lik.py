"""Synthetic (multivariate-normal) likelihood for the public counts (N60, N72, N78).
For each parameter set theta: simulate R patient-level trials, estimate mean mu(theta) and covariance S(theta) of the
*reported* counts at the three dates, and score the observed (60,72,78) under N(mu,S).  Also P(<=79 reported at 11 Aug)."""
import sys, pickle, time, numpy as np
from scipy.stats import norm
import regal_model as r, regal_model_v2 as v

LN2 = np.log(2)
OBS = np.array([60.0, 72.0, 78.0])
DATES = np.array([r.D["e60"], r.D["e72"], r.D["e78"]])
PRIOR = dict(M=(6, 20), S36=(0.03, 0.45), c=(0.0, 0.65), theta=(0.3, 1.3), L=(1, 8), leak=(0.002, 0.009), gamma=(1.4, 2.8))

def sim_from_params(p, rng):
    n = len(p["M"]); k = p["k"]
    M, kk = p["M"][:, None], k[:, None]
    a, arm = r.enroll(n, p["gamma"], rng)
    Hl = LN2 * (p["L"][:, None] / M) ** kk
    E = rng.exponential(1.0, (n, r.N))
    t_bat = M * (E / LN2) ** (1 / kk)
    resp = rng.random((n, r.N)) < p["c"][:, None]
    H_after = Hl + (E - Hl) / p["theta"][:, None]
    t_unc = np.where(E < Hl, t_bat, M * (H_after / LN2) ** (1 / kk))
    t_resp = np.where(E < Hl, t_bat, p["L"][:, None] + rng.exponential(1.0 / p["leak"][:, None], (n, r.N)))
    T = np.where(arm == 1, np.where(resp, t_resp, t_unc), t_bat)
    return a, arm, T, resp & (arm == 1)

def run(scen, K=60000, R=120, block=200, seed=11):
    rng = np.random.default_rng(seed)
    th = {kx: rng.uniform(*PRIOR[kx], K) for kx in PRIOR}
    kk = r.weib_k(th["M"], th["S36"]); valid = np.isfinite(kk) & (kk > 0.3) & (kk < 2.0)
    th["k"] = np.where(valid, kk, 1.0)
    mu = np.zeros((K, 3)); cov = np.zeros((K, 3, 3)); pst = np.zeros(K); fu = np.zeros(K); sdfu = np.zeros(K)
    with np.errstate(all="ignore"):
        for s in range(0, K, block):
            sl = slice(s, min(s + block, K)); nb = sl.stop - sl.start
            p = {kx: np.repeat(vv[sl], R) for kx, vv in th.items()}
            a, arm, T, resp = sim_from_params(p, rng)
            d = a + T
            arrive, gap, W = v.add_reporting(nb * R, rng, scen, a, arm, T, d, resp)
            C = np.stack([(arrive <= t).sum(1) for t in DATES], 1).reshape(nb, R, 3).astype(float)
            st = ((arrive <= r.D["q2"]).sum(1) <= 79).reshape(nb, R)
            obs_i = np.minimum(np.minimum(T, W), r.D["e60"] - a)
            f = np.median(obs_i, axis=1).reshape(nb, R)
            mu[sl] = C.mean(1)
            Cc = C - mu[sl][:, None, :]
            cov[sl] = np.einsum("bri,brj->bij", Cc, Cc) / (R - 1)
            pst[sl] = st.mean(1); fu[sl] = f.mean(1); sdfu[sl] = f.std(1)
    return th, valid, mu, cov, pst, fu, sdfu

def loglik(mu, cov, pst, fu=None, sdfu=None, ridge=0.5, use_stall=True, use_fu=False):
    K = len(mu); ll = np.full(K, -np.inf)
    cr = cov + ridge * np.eye(3)[None]
    diff = (OBS[None, :] - mu)
    sol = np.linalg.solve(cr, diff[:, :, None])[:, :, 0]
    _, logdet = np.linalg.slogdet(cr)
    ll = -0.5 * (diff * sol).sum(1) - 0.5 * logdet - 1.5 * np.log(2 * np.pi)
    if use_stall: ll = ll + np.log(np.clip(pst, 1e-3, 1))
    if use_fu: ll = ll + norm.logpdf(13.5, fu, np.sqrt(sdfu ** 2 + 1.0))
    return ll

if __name__ == "__main__":
    out = {}
    for nm, sc in [("S0 no lag", dict(lag=False)),
                   ("V3 calls to wk156 + real-world 50%", v.SCENARIOS_V2["V3 calls to wk156 + real-world 50%"])]:
        t = time.time(); out[nm] = run(sc); print(nm, "done", round(time.time() - t), "s", flush=True)
    pickle.dump(out, open("results_v2/synth.pkl", "wb"))
