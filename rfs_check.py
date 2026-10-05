import numpy as np, patient_model as pm
rng=np.random.default_rng(2); n=240
X=pm.sample_prior(n,rng); Z=pm.make_Z(60,seed=4)
rows=[]
for i in range(0,n,30):
    S=pm.sim_block(X[i:i+30],Z)
    for j in range(30):
        b=~S['arm'][j]; T=S['T'][j]; R=S['R'][j]; u=S['u'][j]; rel=S['relapsed'][j]
        rfs_rand=np.minimum(R, np.where(rel, T-0, T))     # relapse-or-death time from randomisation (approx: relapse time or death)
        rfs=np.where(rel, R, T)
        rows.append([np.median(rfs[b]), (rfs[b]>12).mean(), np.median((rfs+u)[b]), (rfs[b]>36).mean(), rel[b].mean()])
rows=np.array(rows); q=lambda x:np.percentile(x,[5,50,95]).round(2)
print("BAT RFS median from randomisation (mo)",q(rows[:,0]),"| from CR2",q(rows[:,2]))
print("BAT 1-yr RFS from randomisation",q(rows[:,1])," 3-yr RFS",q(rows[:,3])," P(relapse before death)",q(rows[:,4]))
print("literature: QUAZAR placebo CR1 RFS median 4.8, 1-yr RFS 27%; CR2 duration 4-10 mo (Remarkable-Big/Leopold); annual relapse 40/17/2% yrs1-3")
