"""
Likelihood-calibrated REGAL model with a FLEXIBLE GPS survival curve and a LITERATURE-GROUNDED BAT arm.

GPS curve parameters (broad priors; the data move these)
  pi     : cure fraction = share of GPS patients with a durable, cure-like effect                  U(0, 1)
  th_nc  : relapse-hazard multiplier after onset for the NON-cured GPS patients (partial benefit)   U(0.4, 1.2)
  th_d   : residual relapse-hazard multiplier of cured patients (leak; 0 = true cure)               logU(0.01, 0.3)
  L      : median onset delay of the effect, months                                                lognormal, median 3, sd(log) 0.7
BAT parameters (informative; literature)
  bat_eff: relapse HR of active BAT vs observation                                                 N(0.75, 0.15)
  lam_b  : scale of the observation-arm relapse hazard (calibrated to QUAZAR RFS / 40-17-2 pattern) logN(0, 0.2)
  m_p    : post-relapse median survival, months                                                     N(5.5, 0.7)
  + BAT outcome DATA: population BAT 12/24/36-month OS vs the van der Maas bridge (0.52 / 0.30 / 0.18)
Likelihood: expected reported counts at 60/72/78 (tolerance tau), P(80th not reported by 11 Aug), soft interim term, BAT literature data.
"""
import sys, json, time, pickle, datetime as dt
import numpy as np
import regal_model as r, regal_bio as b

OBS = np.array([60., 72., 78.]); FUT = (0.78, 0.83, 0.89)
LN = np.log
# name, lo, hi, kind (n normal / u uniform / lu log-uniform on the parameter / ln normal on log), a, b
PARAMS = [("pi", 0.0, 1.0, "u", 0, 0), ("th_nc", 0.4, 1.2, "u", 0, 0), ("lth_d", LN(0.01), LN(0.3), "u", 0, 0), ("lL", 0.0, LN(12.0), "n", LN(3.0), 0.7),
          ("bat_eff", 0.4, 1.1, "n", 0.75, 0.15), ("llam_b", -0.7, 0.7, "n", 0.0, 0.2), ("m_p", 4.0, 7.5, "n", 5.5, 0.7)]
NAMES = [p[0] for p in PARAMS]; IDX = {n: i for i, n in enumerate(NAMES)}; D = len(PARAMS)
LO = np.array([p[1] for p in PARAMS]); HI = np.array([p[2] for p in PARAMS]); KIND = [p[3] for p in PARAMS]
PA = np.array([p[4] for p in PARAMS], float); PB = np.array([p[5] for p in PARAMS], float)
GPS_PH2 = {"pi": (0.5, 0.2)}                                   # optional Phase 2-informed prior on the cure fraction
PRIOR_MODE = "broad"


def log_prior(X):
    inbox = np.all((X >= LO) & (X <= HI), axis=1); lp = np.zeros(len(X))
    for j, k in enumerate(KIND):
        if k == "n": lp += -0.5 * ((X[:, j] - PA[j]) / PB[j]) ** 2
    if PRIOR_MODE == "ph2": lp += -0.5 * ((X[:, IDX["pi"]] - 0.5) / 0.2) ** 2
    return np.where(inbox, lp, -np.inf)


def sample_prior(n, rng):
    X = np.empty((n, D))
    for j, k in enumerate(KIND):
        if k == "n":
            x = rng.normal(PA[j], PB[j], n * 4); x = x[(x >= LO[j]) & (x <= HI[j])][:n]
        else: x = rng.uniform(LO[j], HI[j], n)
        X[:, j] = x
    if PRIOR_MODE == "ph2":
        keep = rng.random(n) < np.exp(-0.5 * ((X[:, IDX["pi"]] - 0.5) / 0.2) ** 2); X[~keep, IDX["pi"]] = np.clip(rng.normal(0.5, 0.2, (~keep).sum()), 0, 1)
    return X


def extras(X):
    return dict(th_nc=X[:, IDX["th_nc"]], th_d=np.exp(X[:, IDX["lth_d"]]), onset_med=np.exp(X[:, IDX["lL"]]), lam_b=np.exp(X[:, IDX["llam_b"]]), m_p=X[:, IDX["m_p"]])


def c_bio():
    c = json.load(open("calib_bio.json")); return dict(b.BIO, Mr=c["Mr"], kr=c["kr"])


# ---------------------------------------------------------------- evaluation of particles (parallel)
_G = {}
def _init(R, seed):
    bio = c_bio(); Z = b.make_Z(R, seed); _G.update(bio=bio, Z=Z, coh=b.cohort(Z, bio), a=b.enrollment_dates(bio["gamma"]), R=R)

def _block(X):
    G = _G; S = b.simulate(X[:, IDX["bat_eff"]], X[:, IDX["pi"]], G["Z"], G["bio"], G["coh"], G["a"], extras(X)); nb = len(X)
    C = np.stack([(S["arrive"] <= t).sum(-1) for t in (r.D["e60"], r.D["e72"], r.D["e78"])], -1); mu = C.mean(1)
    pst = ((S["arrive"] <= r.D["q2"]).sum(-1) <= 79).mean(1)
    zi, hri, _ = b.analyse(S, r.D["e60"]); ze = -zi
    pc = np.mean([((ze < r.Z1) & (hri <= h)).mean(1) for h in FUT], axis=0); pe = (ze >= r.Z1).mean(1); pf = np.mean([(hri > h).mean(1) for h in FUT], axis=0)
    arm = S["arm"]; T = S["T"]; nbat = (~arm).sum((1, 2)); ngps = arm.sum((1, 2))
    sv = lambda t, m, n: ((T > t) & m).sum((1, 2)) / n
    return [mu, pst, pc, pe, pf, sv(12, ~arm, nbat), sv(24, ~arm, nbat), sv(36, ~arm, nbat), sv(12, arm, ngps), sv(36, arm, ngps)]

def _work(Xc):
    out = [_block(Xc[i:i + 20]) for i in range(0, len(Xc), 20)]
    return [np.concatenate([o[k] for o in out]) for k in range(10)]

class Evaluator:
    def __init__(self, R=120, seed=7, workers=10):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(R, seed)); self.w = workers
    def __call__(self, X):
        res = list(self.pool.map(_work, [c for c in np.array_split(X, self.w * 3) if len(c)]))
        return [np.concatenate([q[k] for q in res]) for k in range(10)]
    def close(self): self.pool.shutdown()


# ---------------------------------------------------------------- likelihood
VDM = dict(b12=(0.52, 0.06), b24=(0.30, 0.06), b36=(0.18, 0.05))
def loglike(ev, tau, w_ia, bat_sd_mult):
    mu, pst, pc, pe, pf, b12, b24, b36, g12, g36 = ev
    ll = -0.5 * (((OBS[None] - mu) / tau) ** 2).sum(1) + np.log(np.clip(pst, .05, 1)) + w_ia * np.log(np.clip(pc, .02, 1))
    if bat_sd_mult > 0:
        for v, (m_, s_) in zip((b12, b24, b36), (VDM["b12"], VDM["b24"], VDM["b36"])): ll = ll - 0.5 * ((v - m_) / (s_ * bat_sd_mult)) ** 2
    return ll

def ess_of(w): return w.sum() ** 2 / (w ** 2).sum()
def systematic(w, rng):
    n = len(w); u = (rng.random() + np.arange(n)) / n; return np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), n - 1)
def reflect(x):
    x = np.where(x < LO, 2 * LO - x, x); return np.where(x > HI, 2 * HI - x, x)


def smc(tau=1.0, w_ia=1.0, bat_sd_mult=1.0, N=2000, R=120, seed=1, workers=10, verbose=True):
    rng = np.random.default_rng(seed); E = Evaluator(R=R, workers=workers)
    X = sample_prior(N, rng); lp = log_prior(X); ev = E(X); ll = loglike(ev, tau, w_ia, bat_sd_mult); beta = 0.0; scale = 0.4; hist = []
    snap = lambda tag: hist.append(dict(stage=tag, beta=beta, mu=ev[0].mean(0), pc=float(ev[2].mean()), rmse=float(np.sqrt(((OBS[None] - ev[0]) ** 2).mean())), bat36=float(ev[7].mean()), gps36=float(ev[9].mean()), X_med=np.median(X, 0)))
    snap("prior"); stage = 0
    while beta < 1.0:
        stage += 1; fin = np.isfinite(ll); mx = ll[fin].max()
        f = lambda d: ess_of(np.where(fin, np.exp(d * (ll - mx)), 0.0)) - 0.6 * N; hi_ = 1.0 - beta
        if f(hi_) >= 0: d = hi_
        else:
            a_, b_ = 1e-10, hi_
            for _ in range(60):
                mid = 0.5 * (a_ + b_); (a_, b_) = (mid, b_) if f(mid) >= 0 else (a_, mid)
            d = a_
        beta = min(1.0, beta + d); wt = np.where(fin, np.exp(d * (ll - mx)), 0.0); ii = systematic(wt, rng)
        X, lp, ll = X[ii], lp[ii], ll[ii]; ev = [e[ii] for e in ev]; acc_t = 0
        for _ in range(3):
            Cv = np.cov(X.T) * (2.38 ** 2 / D) * scale
            prop = reflect(X + rng.multivariate_normal(np.zeros(D), Cv + 1e-10 * np.eye(D), size=N)); lpp = log_prior(prop); ok = np.isfinite(lpp)
            evp = [np.zeros((N, 3))] + [np.zeros(N) for _ in range(9)]
            if ok.any():
                e_ = E(prop[ok])
                for k in range(10): evp[k][ok] = e_[k]
            llp = np.where(ok, loglike(evp, tau, w_ia, bat_sd_mult), -np.inf); la = (lpp + beta * llp) - (lp + beta * ll)
            acc = np.log(rng.random(N)) < np.where(np.isfinite(la), la, -np.inf)
            X[acc], lp[acc], ll[acc] = prop[acc], lpp[acc], llp[acc]
            for k in range(10): ev[k][acc] = evp[k][acc]
            acc_t += acc.mean()
        ar = acc_t / 3; scale *= 1.3 if ar > 0.30 else (0.7 if ar < 0.12 else 1.0); snap(stage)
        if verbose:
            h = hist[-1]; print(f"  stage {stage:2d} beta {beta:.4f} acc {ar:.2f} counts {np.round(h['mu'],1)} rmse {h['rmse']:.2f}  BAT3y {h['bat36']:.3f} GPS3y {h['gps36']:.3f}  P(IA cont) {h['pc']:.2f}", flush=True)
    E.close(); return dict(X=X, ev=ev, ll=ll, hist=hist, tau=tau, w_ia=w_ia, bat_sd_mult=bat_sd_mult)


# ---------------------------------------------------------------- realised trials, HR, curves
TGRID = np.arange(0, 73, 6.0)
def realised(X, R2=100, nsel=800, seed=99):
    rng = np.random.default_rng(seed); sel = X[rng.choice(len(X), nsel, replace=True)]; bio = c_bio(); a_dates = b.enrollment_dates(bio["gamma"]); recs = []
    for s in range(0, nsel, 20):
        xb = sel[s:s + 20]; nb = len(xb); Z = b.make_Z(R2, 1000 + s); coh = b.cohort(Z, bio)
        S = b.simulate(xb[:, IDX["bat_eff"]], xb[:, IDX["pi"]], Z, bio, coh, a_dates, extras(xb)); f = lambda a_: a_.reshape(nb * R2, b.N)
        C = np.stack([(S["arrive"] <= t).sum(-1) for t in (r.D["e60"], r.D["e72"], r.D["e78"])], -1).reshape(nb * R2, 3)
        T80 = np.partition(S["arrive"], 79, axis=-1)[..., 79]; cut = np.minimum(T80, r.m(dt.date(2030, 1, 1)))
        zf, hrf, _ = b.analyse(S, cut); zu, _, _ = b.analyse(dict(S, code=np.zeros_like(S["code"])), cut)
        zi, hri, _ = b.analyse(S, r.D["e60"]); ia = np.mean([((-zi < r.Z1) & (hri <= h)).astype(float) for h in FUT], axis=0)
        stall = ((S["arrive"] <= r.D["q2"]).sum(-1) <= 79)
        arm = S["arm"]; T = S["T"]
        sb = np.stack([[((T[i] > t) & ~arm[i]).sum() / (~arm[i]).sum() for t in TGRID] for i in range(nb)]); sg = np.stack([[((T[i] > t) & arm[i]).sum() / arm[i].sum() for t in TGRID] for i in range(nb)])
        pop = np.zeros(nb)
        for i in range(nb):
            tm = np.clip(np.minimum(T[i], r.D["q2"] - S["a"][i]), 0.01, None).reshape(1, -1); ev_ = (S["d"][i] <= r.D["q2"]).reshape(1, -1).astype(np.int8)
            pop[i] = np.exp(b.strat_stats(tm, ev_, arm[i].reshape(1, -1).astype(np.int8), S["code"][i].reshape(1, -1))[1][0])
        recs.append(dict(C=C, z=zf.reshape(-1), hr=hrf.reshape(-1), zu=zu.reshape(-1), ia=ia.reshape(-1), stall=stall.reshape(-1), part=np.repeat(np.arange(s, s + nb), R2),
                         sb=sb, sg=sg, pop=pop, popidx=np.arange(s, s + nb)))
    out = {k: np.concatenate([q[k] for q in recs]) for k in recs[0]}; out["theta"] = sel; return out


def summarize(fit, rel):
    C = rel["C"]; kern = np.exp(-0.5 * (((C - OBS[None]) / 2.0) ** 2).sum(1)); w = kern * rel["ia"]; succ = rel["z"] <= -r.ZF; succu = rel["zu"] <= -r.ZF
    pw = np.bincount(rel["part"], weights=w, minlength=len(rel["theta"])) + 1e-12
    q = lambda x: r.wq(x, w + 1e-300, [.05, .5, .95]); qp = lambda x: r.wq(x, pw, [.05, .5, .95])
    return dict(P_succ=float((w * succ).sum() / w.sum()), P_succ_unstrat=float((w * succu).sum() / w.sum()), HR=q(rel["hr"]), popHR=qp(rel["pop"]), HR_all=np.percentile(rel["hr"], [5, 50, 95]),
                BAT=[qp(rel["sb"][:, i]) for i in range(len(TGRID))], GPS=[qp(rel["sg"][:, i]) for i in range(len(TGRID))], ESS=float(w.sum() ** 2 / (w ** 2).sum()), pw=pw)


if __name__ == "__main__":
    tag = sys.argv[1]; tau = float(sys.argv[2]); bat_mult = float(sys.argv[3]); PRIOR_MODE = sys.argv[4] if len(sys.argv) > 4 else "broad"
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    t = time.time(); print(f"== {tag} tau {tau} BAT-data sd x{bat_mult} GPS prior {PRIOR_MODE}", flush=True)
    fit = smc(tau=tau, bat_sd_mult=bat_mult, N=N); print("SMC", round(time.time() - t), "s", flush=True)
    rel = realised(fit["X"]); pickle.dump(dict(fit=fit, rel=rel, prior_mode=PRIOR_MODE), open(f"results_v2/sb_{tag}.pkl", "wb")); print("total", round(time.time() - t), "s", flush=True)
