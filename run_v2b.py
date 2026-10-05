import pickle, time, regal_model as r, regal_model_v2 as v
allres = pickle.load(open("results_v2/scen_v2.pkl", "rb"))
for i, (nm, sc) in enumerate(v.SCENARIOS_2S_LIT.items()):
    t = time.time(); res = v.run_v2(nm, sc, n_sim=5_000_000, seed=500 + i); allres[nm] = res
    print(f"{nm}: kept {len(res['w'])}, ESS {r.ess(res['w']):.0f}, p_stall|78 {res['p_stall_given_78']:.3f}, {time.time()-t:.0f}s", flush=True)
    pickle.dump(allres, open("results_v2/scen_v2.pkl", "wb"))
