import json, numpy as np, datetime as dt
import regal_model as r, regal_bio as b
# reference biology: relapse anchored to QUAZAR placebo 6/12-mo RFS plus the 40/17/2 yearly relapse pattern (3-yr RFS 22%)
b.RFS_TARGET.clear(); b.RFS_TARGET.update({6: .45, 12: .27, 36: .22})
bio = dict(b.BIO); bio["Mr"], bio["kr"], fit = b.calibrate_relapse(bio); print("reference biology: Mr %.2f kr %.2f  RFS(6,12,36) %s" % (bio["Mr"], bio["kr"], np.round(fit, 3)))
json.dump(dict(Mr=bio["Mr"], kr=bio["kr"], rfs_targets={"6": .45, "12": .27, "36": .22}), open("calib_bio.json", "w"))
R = 1500; Z = b.make_Z(R, 21); a_dates = b.enrollment_dates(bio["gamma"])
base = b.cohort(Z, bio)
def variant(kind):
    c = dict(base)
    if kind.startswith("block"):
        c = b.cohort(Z, dict(bio, block=int(kind[5:])))
    elif kind == "simple":                           # simple 1:1 random permutation, ignores strata
        rng = np.random.default_rng(5); ranks = np.argsort(np.argsort(rng.random((R, b.N)), axis=1), axis=1)
        c["arm"] = ranks < (b.N // 2)
    elif kind == "alternate_sorted":                 # earlier approximation: alternate after sorting on stratum, independent of enrolment order
        rng = np.random.default_rng(6); key = rng.random((R, b.N)); order = np.argsort(base["code"] + 0.9 * key, axis=1)
        pos = np.empty_like(order); np.put_along_axis(pos, order, np.arange(b.N)[None], axis=1); c["arm"] = ((pos + (rng.random((R, 1)) > .5)) % 2).astype(bool)
    return c
print("\nP(success) and HR spread at fixed knobs (bat_eff 0.55, gps_eff 0.95), unconditional realised trials (R=1500)")
print(f"{'randomisation':34s} {'analysis':14s} {'GPS n':>6s} {'sd(logHR)':>9s} {'P(success)':>10s} {'median HR':>9s}")
for kind in ["block4", "block2", "block6", "simple", "alternate_sorted"]:
    coh = variant(kind)
    S = b.simulate(np.array([0.55]), np.array([0.95]), Z, bio, coh, a_dates)
    T80 = np.partition(S["arrive"], 79, axis=-1)[..., 79]
    for an in ["stratified", "unstratified"]:
        S2 = dict(S)
        if an == "unstratified": S2["code"] = np.zeros_like(S["code"])
        zf, hrf, _ = b.analyse(S2, np.minimum(T80, r.m(dt.date(2030, 1, 1))))
        print(f"{kind:34s} {an:14s} {coh['arm'].sum(1).mean():6.1f} {np.log(hrf[0]).std():9.3f} {(zf[0] <= -r.ZF).mean():10.3f} {np.median(hrf[0]):9.3f}")
