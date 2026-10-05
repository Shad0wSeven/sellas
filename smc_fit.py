"""
Calibrated REGAL model: sequential Monte Carlo (tempered) over the model parameters.

Target:  pi(theta)  ∝  prior_literature(theta)  x  exp( -0.5 * sum_i ((OBS_i - mu_i(theta)) / tau)^2 )  x  P_stall(theta)
  mu_i(theta) = EXPECTED reported death count at the 3 public dates, from R common-random-number patient-level trials
  tau         = tolerance (events) on how closely the model's expected path must hit 60 / 72 / 78
  prior       = lognormal BAT 3-yr OS (median .18), BAT median ~N(12.5,2.5), BAT shape ~N(.85,.25) (VDM bridge / literature),
                optional GPS 3-yr OS ~ N(.50,.15) (Phase 2), data-pull/holiday lag on the 72 count ~ Exp(1.0 mo), max 2.5
SMC walks the particle cloud from the prior to the posterior through a tempering ladder, so you can see the parameters move.
Level 2 then simulates many realised trials per posterior particle and computes HR (Cox), log-rank success, interim behaviour.
"""
import sys, time, pickle, datetime as dt, json
import numpy as np
from scipy.stats import norm
import regal_model as r, regal_model_v2 as v, synth_lik as S

np.seterr(all='ignore')
LN2 = np.log(2)
NAMES = ["S36", "M", "c", "theta", "L", "leak", "gamma", "delta"]
LO = np.array([0.03, 6.0, 0.0, 0.3, 1.0, 0.002, 1.4, 0.0])
HI = np.array([0.45, 20.0, 0.65, 1.3, 8.0, 0.009, 2.8, 2.5])
OBS = S.OBS
V3 = v.SCENARIOS_V2["V3 calls to wk156 + real-world 50%"]
SCEN = {"S0": dict(lag=False), "V3": V3}


# --------------------------------------------------------------------- common random numbers
def make_Z(R, seed=2024):
    rng = np.random.default_rng(seed); sh = (R, r.N)
    return dict(E=rng.exponential(1.0, sh), E2=rng.exponential(1.0, sh), u_resp=rng.random(sh), jit=rng.normal(0, 0.15, sh),
                perm=np.argsort(np.argsort(rng.random(sh), axis=1), axis=1), norm=rng.standard_normal(sh), u_disc=rng.random(sh),
                u_imm=rng.random(sh), R=R)


def enroll_base(gamma):
    n = len(gamma); j = np.arange(1, r.N + 1)[None, :]
    a = np.empty((n, r.N)); u = (j[:, :105] - 1) / 104.0
    a[:, :105] = r.D["nov23"] * u ** (1.0 / gamma[:, None])
    a[:, 105:123] = r.D["nov23"] + (j[:, 105:123] - 105 - 0.5) / 18 * (r.D["mar24"] - r.D["nov23"])
    a[:, 123:] = r.D["mar24"] + (j[:, 123:] - 123 - 0.5) / 4 * (r.D["apr24"] - r.D["mar24"])
    return a


def expected_counts(X, Z, scen, block=60):
    """X: (n,8). returns mu (n,3), p_stall (n,), fu_med (n,), valid (n,)"""
    n = len(X); mu = np.zeros((n, 3)); pst = np.zeros(n); fu = np.zeros(n); valid = np.ones(n, bool)
    arm = (Z["perm"] >= r.N_BAT)[None]                                     # (1,R,N) common
    with np.errstate(all="ignore"):
        for s in range(0, n, block):
            x = X[s:s + block]; nb = len(x)
            S36, M, c, th, L, leak, g, dl = [x[:, i][:, None, None] for i in range(8)]
            k = np.log(-np.log(S36) / LN2) / np.log(36.0 / M)
            valid[s:s + nb] = np.isfinite(k[:, 0, 0]) & (k[:, 0, 0] > 0.3) & (k[:, 0, 0] < 2.0)
            k = np.where(np.isfinite(k) & (k > 0.3) & (k < 2.0), k, 1.0)
            a = np.clip(enroll_base(x[:, 6])[:, None, :] + Z["jit"][None], 0, r.D["apr24"])
            Hl = LN2 * (L / M) ** k
            E = Z["E"][None]
            t_bat = M * (E / LN2) ** (1 / k)
            resp = Z["u_resp"][None] < c
            t_unc = np.where(E < Hl, t_bat, M * ((Hl + (E - Hl) / th) / LN2) ** (1 / k))
            t_resp = np.where(E < Hl, t_bat, L + Z["E2"][None] / leak)
            T = np.where(arm, np.where(resp, t_resp, t_unc), t_bat)
            d = a + T
            if scen["lag"]:
                P = np.exp(np.log(scen["pipe_med"]) + 0.6 * Z["norm"][None])
                interval = np.where(T < scen.get("q_until", 36), 3.0, 12.0)
                disc = Z["u_disc"][None] * interval
                is_resp = resp & arm
                gps_imm = np.where(T < 12, 1.0, np.where(is_resp, scen["gps_on_imm"], scen["q_gps_off"]))
                q = np.where(arm, gps_imm, scen["q_bat"])
                disc = np.where(Z["u_imm"][None] < q, 0.0, disc)
                arrive = d + P + disc
            else:
                arrive = d
            dates = np.stack([np.full(nb, r.D["e60"]), r.D["e72"] - x[:, 7], np.full(nb, r.D["e78"])], 1)   # (nb,3)
            for i in range(3):
                mu[s:s + nb, i] = (arrive <= dates[:, i][:, None, None]).sum(-1).mean(-1)
            pst[s:s + nb] = ((arrive <= r.D["q2"]).sum(-1) <= 79).mean(-1)
            fu[s:s + nb] = np.median(np.minimum(T, r.D["e60"] - a), axis=-1).mean(-1)
    return mu, pst, fu, valid


# --------------------------------------------------------------------- prior / likelihood
def gps_S36(X):
    S36, M, c, th, L, leak = [X[:, i] for i in range(6)]
    k = np.log(-np.log(S36) / LN2) / np.log(36.0 / M)
    H = lambda t: LN2 * (t / M) ** k; Hl = H(L)
    unc = np.exp(-np.where(36 > L, Hl + th * (H(36.0) - Hl), H(36.0)))
    rs = np.exp(-Hl - leak * np.clip(36 - L, 0, None))
    return c * rs + (1 - c) * unc


def log_prior(X, gps_informed):
    S36, M = X[:, 0], X[:, 1]
    k = np.log(-np.log(S36) / LN2) / np.log(36.0 / M)
    inbox = np.all((X >= LO) & (X <= HI), axis=1) & np.isfinite(k) & (k > 0.3) & (k < 2.0)
    lp = (-0.5 * ((np.log(S36) - np.log(0.18)) / 0.38) ** 2 - np.log(S36) + norm.logpdf(M, 12.5, 2.5) + norm.logpdf(k, 0.85, 0.25)
          - X[:, 7] / 1.0)
    if gps_informed: lp = lp + norm.logpdf(gps_S36(X), 0.50, 0.15)
    return np.where(inbox, lp, -np.inf)


def log_like(mu, pst, fu, valid, tau, use_fu=False):
    ll = -0.5 * (((OBS[None] - mu) / tau) ** 2).sum(1) + np.log(np.clip(pst, 0.05, 1.0))
    if use_fu: ll = ll - 0.5 * ((13.5 - fu) / 1.5) ** 2
    return np.where(valid, ll, -np.inf)


# --------------------------------------------------------------------- SMC
def ess_of(w): return w.sum() ** 2 / (w ** 2).sum()

def systematic(w, rng):
    n = len(w); u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), n - 1)

def smc(scen_key, tau=1.0, gps_informed=False, N=1500, R=200, seed=1, use_fu=False, verbose=True):
    rng = np.random.default_rng(seed); Z = make_Z(R); scen = SCEN[scen_key]
    # initial population ~ prior (SIR from a large uniform box sample)
    X0 = LO + (HI - LO) * rng.random((60000, 8)); lp0 = log_prior(X0, gps_informed)
    w0 = np.exp(lp0 - lp0.max()); idx = systematic(w0, rng)[:: max(1, 60000 // N)][:N]
    X = X0[idx]; lp = log_prior(X, gps_informed)
    mu, pst, fu, val = expected_counts(X, Z, scen); ll = log_like(mu, pst, fu, val, tau, use_fu)
    beta = 0.0; scale = 0.5; hist = []
    def snap(tag):
        hist.append(dict(stage=tag, beta=beta, S36=np.percentile(X[:, 0], [5, 50, 95]), M=np.percentile(X[:, 1], [5, 50, 95]),
                         gps36=np.percentile(gps_S36(X), [5, 50, 95]), mu_mean=mu.mean(0), mu_sd_across_particles=mu.std(0),
                         rmse=float(np.sqrt((((OBS[None] - mu) ** 2).mean())))))
    snap("prior")
    stage = 0
    while beta < 1.0:
        stage += 1
        lo_, hi_ = 0.0, 1.0 - beta
        f = lambda d: ess_of(np.exp(d * (ll - ll[np.isfinite(ll)].max())) * np.isfinite(ll)) - 0.6 * N
        d = hi_ if f(hi_) >= 0 else (np.array([lo_, hi_]))[0]
        if f(hi_) < 0:
            a_, b_ = 1e-9, hi_
            for _ in range(60):
                mid = 0.5 * (a_ + b_); (a_, b_) = (mid, b_) if f(mid) >= 0 else (a_, mid)
            d = a_
        beta = min(1.0, beta + d)
        wt = np.exp(d * (ll - ll[np.isfinite(ll)].max())) * np.isfinite(ll)
        ii = systematic(wt, rng); X, lp, mu, pst, fu, val, ll = X[ii], lp[ii], mu[ii], pst[ii], fu[ii], val[ii], ll[ii]
        # MCMC moves at current beta
        acc_tot = 0.0
        for step in range(4):
            C = np.cov(X.T) * (2.38 ** 2 / 8) * scale
            prop = X + rng.multivariate_normal(np.zeros(8), C + 1e-12 * np.eye(8), size=N)
            prop = np.where(prop < LO, 2 * LO - prop, prop); prop = np.where(prop > HI, 2 * HI - prop, prop)
            lpp = log_prior(prop, gps_informed)
            ok = np.isfinite(lpp)
            mup, pstp, fup, valp = (np.zeros((N, 3)), np.zeros(N), np.zeros(N), np.zeros(N, bool))
            if ok.any():
                m_, p_, f_, v_ = expected_counts(prop[ok], Z, scen); mup[ok], pstp[ok], fup[ok], valp[ok] = m_, p_, f_, v_
            llp = np.where(ok, log_like(mup, pstp, fup, valp, tau, use_fu), -np.inf)
            la = (lpp + beta * llp) - (lp + beta * ll)
            acc = np.log(rng.random(N)) < np.where(np.isfinite(la), la, -np.inf)
            X[acc], lp[acc], mu[acc], pst[acc], fu[acc], val[acc], ll[acc] = prop[acc], lpp[acc], mup[acc], pstp[acc], fup[acc], valp[acc], llp[acc]
            acc_tot += acc.mean()
        ar = acc_tot / 4; scale *= 1.3 if ar > 0.35 else (0.7 if ar < 0.15 else 1.0)
        snap(stage)
        if verbose:
            h = hist[-1]; print(f"  stage {stage:2d} beta {beta:.4f} acc {ar:.2f}  BAT3y {h['S36'][1]:.3f}  GPS3y {h['gps36'][1]:.3f}  mean mu {np.round(h['mu_mean'],1)}  rmse-to-obs {h['rmse']:.2f}", flush=True)
    for _ in range(4):                                                    # extra diversification at beta=1
        C = np.cov(X.T) * (2.38 ** 2 / 8) * scale
        prop = X + rng.multivariate_normal(np.zeros(8), C + 1e-12 * np.eye(8), size=N)
        prop = np.where(prop < LO, 2 * LO - prop, prop); prop = np.where(prop > HI, 2 * HI - prop, prop)
        lpp = log_prior(prop, gps_informed); ok = np.isfinite(lpp)
        mup, pstp, fup, valp = (np.zeros((N, 3)), np.zeros(N), np.zeros(N), np.zeros(N, bool))
        if ok.any():
            m_, p_, f_, v_ = expected_counts(prop[ok], Z, scen); mup[ok], pstp[ok], fup[ok], valp[ok] = m_, p_, f_, v_
        llp = np.where(ok, log_like(mup, pstp, fup, valp, tau, use_fu), -np.inf)
        la = (lpp + llp) - (lp + ll); acc = np.log(rng.random(N)) < np.where(np.isfinite(la), la, -np.inf)
        X[acc], lp[acc], mu[acc], pst[acc], fu[acc], val[acc], ll[acc] = prop[acc], lpp[acc], mup[acc], pstp[acc], fup[acc], valp[acc], llp[acc]
    return dict(X=X, mu=mu, pst=pst, fu=fu, hist=hist, tau=tau, gps_informed=gps_informed, scen=scen_key)


# --------------------------------------------------------------------- Level 2: realised trials, HR
def realised(X, scen_key, R2=250, nsel=1000, seed=99):
    rng = np.random.default_rng(seed); scen = SCEN[scen_key]
    sel = X[rng.choice(len(X), nsel, replace=True)]
    recs = []
    with np.errstate(all="ignore"):
        for s in range(0, nsel, 40):
            xb = sel[s:s + 40]; nb = len(xb)
            kk = np.log(-np.log(xb[:, 0]) / LN2) / np.log(36.0 / xb[:, 1])
            p = dict(M=xb[:, 1], S36=xb[:, 0], k=kk, c=xb[:, 2], theta=xb[:, 3], L=xb[:, 4], leak=xb[:, 5], gamma=xb[:, 6])
            p = {k_: np.repeat(vv, R2) for k_, vv in p.items()}
            a, arm, T, resp = S.sim_from_params(p, rng); d = a + T
            arrive, gap, W = v.add_reporting(nb * R2, rng, scen, a, arm, T, d, resp)
            dl = np.repeat(xb[:, 7], R2)
            C = np.stack([(arrive <= r.D["e60"]).sum(1), (arrive <= (r.D["e72"] - dl)[:, None]).sum(1), (arrive <= r.D["e78"]).sum(1)], 1)
            T80 = np.partition(arrive, 79, axis=1)[:, 79]
            cut = np.minimum(T80, r.m(dt.date(2030, 1, 1)))
            (zc, lc), nev = v.analysis2(cut, a, arm, T, d, arrive, gap, W, True)
            (zi, li), _ = v.analysis2(np.full(len(cut), r.D["e60"]), a, arm, T, d, arrive, gap, W, True)
            stall = (arrive <= r.D["q2"]).sum(1) <= 79
            ia_dead = (d <= r.D["e60"]); ia_bat = (ia_dead & (arm == 0)).sum(1); ia_n = ia_dead.sum(1)
            # population-level (stacked) HR at a fixed cutoff for each particle (removes sampling noise)
            pop = np.zeros(nb)
            cutp = r.D["q2"]
            for i in range(nb):
                sl = slice(i * R2, (i + 1) * R2)
                tm = np.minimum(T[sl], cutp - a[sl]).reshape(1, -1); ev = (d[sl] <= cutp).reshape(1, -1).astype(np.int8)
                pop[i] = np.exp(r.logrank(tm, ev, arm[sl].reshape(1, -1))[1][0])
            recs.append(dict(C=C, hr=np.exp(lc), z=zc, nev=nev, z_ia=zi, hr_ia=np.exp(li), stall=stall, ia_bat=ia_bat, ia_n=ia_n,
                             pop=np.repeat(pop, R2), part=np.repeat(np.arange(s, s + nb), R2)))
    out = {k_: np.concatenate([q[k_] for q in recs]) for k_ in recs[0]}
    out["theta"] = sel
    return out


def summarize(fit, rel):
    X = fit["X"]; q = lambda x: np.percentile(x, [5, 50, 95])
    C = rel["C"]; sd = C.std(0)
    kern = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1))                 # condition realised trials on the counts (sigma=2)
    ia_cont = rel["z_ia"] > -r.Z1
    succ = (rel["z"] <= -r.ZF)
    def wq(x, w): return r.wq(x, w, [.05, .5, .95])
    ess_k = kern.sum() ** 2 / (kern ** 2).sum()
    out = dict(
        BAT_S36=q(X[:, 0]), BAT_median=q(X[:, 1]), GPS_S36=q(gps_S36(X)), c=q(X[:, 2]), theta=q(X[:, 3]), L=q(X[:, 4]), leak=q(X[:, 5]), gamma=q(X[:, 6]), pull_lag=q(X[:, 7]),
        expected_counts_mean=fit["mu"].mean(0), expected_counts_sd_across_particles=fit["mu"].std(0),
        realised_counts_mean=C.mean(0), realised_counts_sd=sd,
        HR_unconditional=q(rel["hr"]), P_HR_lt_0636=float((rel["hr"] < 0.636).mean()), P_success_uncond=float(succ.mean()), P_IA_continues_uncond=float(ia_cont.mean()),
        P_success_given_IA=float(succ[ia_cont].mean()),
        pop_HR=q(rel["pop"]),
        ESS_counts_conditioned=float(ess_k),
        HR_given_counts=wq(rel["hr"], kern), P_success_given_counts=float((kern * succ).sum() / kern.sum()),
        HR_given_counts_and_IA=wq(rel["hr"], kern * ia_cont), P_success_given_counts_and_IA=float((kern * ia_cont * succ).sum() / (kern * ia_cont).sum()),
        P_stall=float(rel["stall"].mean()),
    )
    return out


if __name__ == "__main__":
    scen_key, tau, gi = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "gps"
    tag = f"{scen_key}_tau{tau}_{'gps' if gi else 'agn'}"
    t = time.time(); print(f"== {tag}", flush=True)
    fit = smc(scen_key, tau=tau, gps_informed=gi, N=1500, R=200, seed=7)
    print("SMC done", round(time.time() - t), "s", flush=True)
    rel = realised(fit["X"], scen_key)
    summ = summarize(fit, rel)
    pickle.dump(dict(fit=fit, summ=summ, rel={k: v_ for k, v_ in rel.items()}), open(f"results_v2/smc_{tag}.pkl", "wb"))
    for k_, v_ in summ.items(): print(f"  {k_}: {np.round(v_, 3)}")
    print("total", round(time.time() - t), "s")
