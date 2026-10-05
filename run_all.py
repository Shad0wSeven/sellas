import sys, json, datetime as dt, pickle, time
import numpy as np
import regal_model as r

OUT = "results"
import os; os.makedirs(OUT, exist_ok=True)
NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000

def run(name, scen, seed, n=NSIM):
    t = time.time()
    res = r.run_scenario(name, scen, n_sim=n, seed=seed)
    print(f"{name}: kept {len(res['w'])}, ESS(flat) {r.ess(res['w']):.0f}, ev {res['evidence']:.5f}, {time.time()-t:.0f}s", flush=True)
    return res

if __name__ == "__main__":
    allres = {}
    for i, (nm, sc) in enumerate(r.SCENARIOS.items()):
        allres[nm] = run(nm, sc, seed=100 + i)
    pickle.dump(allres, open(f"{OUT}/scenarios.pkl", "wb"))
    # sensitivity: BAT real-world discovery probability sweep (others as S3)
    sweep = {}
    for j, q in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        sc = dict(r.SCENARIOS["S3 protocol + real-world (50%)"]); sc["q_bat"] = q
        sweep[q] = run(f"qbat={q}", sc, seed=200 + j, n=NSIM // 2)
    pickle.dump(sweep, open(f"{OUT}/sweep.pkl", "wb"))
