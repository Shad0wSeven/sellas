"""Tempered SMC for the patient-level model (patient_model.py) with a SOFT interim likelihood.

log-likelihood = -0.5*sum(((OBS-mu)/tau)^2)            expected reported counts at 3 dates must sit near 60/72/78
               + log P(80th not reported by 11 Aug)    stall
               + w_ia * log P(IDMC continued | theta)  = P(no efficacy stop AND passed futility) -- soft, no cutoff
Prior = scientific priors in patient_model.PARAMS (normal around literature values, truncated to a wide plausible box).
"""
import sys, time, pickle, datetime as dt
import numpy as np
import regal_model as r, regal_model_v2 as v
import patient_model as pm

OBS = np.array([60.0, 72.0, 78.0])
IDX = pm.IDX


def loglike(ev, tau, w_ia, w_stall=1.0):
    mu, pst, fu, pc, pe, pf = ev
    return (-0.5 * (((OBS[None] - mu) / tau) ** 2).sum(1) + w_stall * np.log(np.clip(pst, 0.05, 1.0)) + w_ia * np.log(np.clip(pc, 0.02, 1.0)))


def ess_of(w): return w.sum() ** 2 / (w ** 2).sum()
def systematic(w, rng):
    n = len(w); u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), n - 1)


def reflect(x):
    x = np.where(x < pm.LO, 2 * pm.LO - x, x); return np.where(x > pm.HI, 2 * pm.HI - x, x)


def smc(tau=1.0, w_ia=1.0, N=2000, R=120, seed=1, workers=8, verbose=True):
    rng = np.random.default_rng(seed); E = pm.Evaluator(R=R, workers=workers)
    X = pm.sample_prior(N, rng); lp = pm.log_prior(X)
    ev = E(X); ll = loglike(ev, tau, w_ia)
    beta = 0.0; scale = 0.4; hist = []
    def snap(tag):
        hist.append(dict(stage=tag, beta=beta, mu_mean=ev[0].mean(0), pc=float(ev[3].mean()), pe=float(ev[4].mean()), pf=float(ev[5].mean()),
                         rmse=float(np.sqrt(((OBS[None] - ev[0]) ** 2).mean())), X_med=np.median(X, 0)))
    snap("prior")
    if verbose: print(f"  prior   mean mu {np.round(ev[0].mean(0),1)}  P(IA continue) {ev[3].mean():.2f} (eff-stop {ev[4].mean():.2f}, fut-stop {ev[5].mean():.2f})", flush=True)
    stage = 0
    while beta < 1.0:
        stage += 1
        fin = np.isfinite(ll); mx = ll[fin].max()
        f = lambda d: ess_of(np.where(fin, np.exp(d * (ll - mx)), 0.0)) - 0.6 * N
        hi_ = 1.0 - beta
        if f(hi_) >= 0: d = hi_
        else:
            a_, b_ = 1e-10, hi_
            for _ in range(60):
                mid = 0.5 * (a_ + b_); (a_, b_) = (mid, b_) if f(mid) >= 0 else (a_, mid)
            d = a_
        beta = min(1.0, beta + d)
        wt = np.where(fin, np.exp(d * (ll - mx)), 0.0); ii = systematic(wt, rng)
        X, lp, ll = X[ii], lp[ii], ll[ii]; ev = [e[ii] for e in ev]
        acc_tot = 0
        for step in range(3):
            C = np.cov(X.T) * (2.38 ** 2 / pm.D) * scale
            prop = reflect(X + rng.multivariate_normal(np.zeros(pm.D), C + 1e-10 * np.eye(pm.D), size=N))
            lpp = pm.log_prior(prop); ok = np.isfinite(lpp)
            evp = [np.zeros((N, 3)), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
            if ok.any():
                e_ = E(prop[ok])
                for k in range(6): evp[k][ok] = e_[k]
            llp = np.where(ok, loglike(evp, tau, w_ia), -np.inf)
            la = (lpp + beta * llp) - (lp + beta * ll)
            acc = np.log(rng.random(N)) < np.where(np.isfinite(la), la, -np.inf)
            X[acc], lp[acc], ll[acc] = prop[acc], lpp[acc], llp[acc]
            for k in range(6): ev[k][acc] = evp[k][acc]
            acc_tot += acc.mean()
        ar = acc_tot / 3; scale *= 1.3 if ar > 0.30 else (0.7 if ar < 0.12 else 1.0)
        snap(stage)
        if verbose:
            h = hist[-1]; print(f"  stage {stage:2d} beta {beta:.4f} acc {ar:.2f} mean mu {np.round(h['mu_mean'],1)} rmse {h['rmse']:.2f}  P(IA continue) {h['pc']:.2f} (eff {h['pe']:.2f}, fut {h['pf']:.2f})", flush=True)
    E.close()
    return dict(X=X, ev=ev, ll=ll, hist=hist, tau=tau, w_ia=w_ia)


def realised(X, R2=100, nsel=800, seed=99):
    rng = np.random.default_rng(seed); sel = X[rng.choice(len(X), nsel, replace=True)]
    recs = []
    for s in range(0, nsel, 16):
        xb = sel[s:s + 16]; nb = len(xb); Z = pm.make_Z(R2, seed=1000 + s)
        S = pm.sim_block(xb, Z)
        flat = lambda a: a.reshape(nb * R2, pm.N)
        a, arm, T, d, arrive = flat(S["a"]), flat(S["arm"]).astype(np.int8), flat(S["T"]), flat(S["d"]), flat(S["arrive"])
        dl = np.repeat(xb[:, IDX["delta"]], R2)
        C = np.stack([(arrive <= r.D["e60"]).sum(1), (arrive <= (r.D["e72"] - dl)[:, None]).sum(1), (arrive <= r.D["e78"]).sum(1)], 1)
        T80 = np.partition(arrive, 79, axis=1)[:, 79]; cut = np.minimum(T80, r.m(dt.date(2030, 1, 1)))
        gap = np.random.default_rng(s).random(a.shape) * flat(S["interval"]); W = np.full(a.shape, np.inf)
        (zc, lc), nev = v.analysis2(cut, a, arm, T, d, arrive, gap, W, True)
        (zi, li), _ = v.analysis2(np.full(len(cut), r.D["e60"]), a, arm, T, d, arrive, gap, W, True)
        stall = (arrive <= r.D["q2"]).sum(1) <= 79
        zb = np.repeat(xb[:, IDX["z_b"]], R2); hf = np.repeat(xb[:, IDX["h_f"]], R2)
        ia_cont = (-zi < zb) & (np.exp(li) <= hf)
        # population arm survival from uncensored simulated times (per particle)
        Tr = S["T"]; armr = S["arm"]
        pop = {}
        for nm, msk in (("bat", ~armr), ("gps", armr)):
            for t in (12, 36, 60):
                pop[f"{nm}_S{t}"] = np.array([np.mean(Tr[i][msk[i]] > t) for i in range(nb)])
            pop[f"{nm}_med"] = np.array([np.median(Tr[i][msk[i]]) for i in range(nb)])
            rel = np.where(S["relapsed"] | True, 1, 0)
        recs.append(dict(C=C, hr=np.exp(lc), z=zc, nev=nev, z_ia=zi, hr_ia=np.exp(li), ia_cont=ia_cont, stall=stall,
                         part=np.repeat(np.arange(s, s + nb), R2), pop_idx=np.arange(s, s + nb), **{f"pop_{k}": v_ for k, v_ in pop.items()},
                         sct=flat(S["sct"]).mean(1), relapsed=flat(S["relapsed"]).mean(1), age=flat(S["age"]).mean(1), longf=flat(S["long"]).mean(1)))
    out = {}
    for k in recs[0]:
        out[k] = np.concatenate([q[k] for q in recs])
    out["theta"] = sel
    return out


if __name__ == "__main__":
    tau = float(sys.argv[1]); w_ia = float(sys.argv[2]); tag = sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    t = time.time(); print(f"== {tag} tau={tau} w_ia={w_ia} N={N}", flush=True)
    fit = smc(tau=tau, w_ia=w_ia, N=N, R=120, seed=5, workers=int(sys.argv[5]) if len(sys.argv) > 5 else 8)
    print("SMC done", round(time.time() - t), "s", flush=True)
    rel = realised(fit["X"], R2=100, nsel=800)
    pickle.dump(dict(fit=fit, rel=rel), open(f"results_v2/smc3_{tag}.pkl", "wb"))
    print("total", round(time.time() - t), "s", flush=True)
