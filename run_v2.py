import sys, pickle, time
import regal_model as r, regal_model_v2 as v
NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
allres = {}
i = 0
for group in (v.SCENARIOS_V2, v.SCENARIOS_2S):
    for nm, sc in group.items():
        t = time.time(); i += 1
        res = v.run_v2(nm, sc, n_sim=NSIM, seed=300 + i)
        allres[nm] = res
        print(f"{nm}: kept {len(res['w'])}, ESS {r.ess(res['w']):.0f}, p_stall|78 {res['p_stall_given_78']:.3f}, {time.time()-t:.0f}s", flush=True)
        pickle.dump(allres, open("results_v2/scen_v2.pkl", "wb"))
