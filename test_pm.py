import time, numpy as np
import patient_model as pm, regal_model as r
rng=np.random.default_rng(0)
X=pm.sample_prior(40,rng); Z=pm.make_Z(60)
t=time.time(); ev=pm.summarize_block(X,Z); print("block time",round(time.time()-t,2),"s for 40 particles x 60 reps")
mu,pst,fu,pc,pe,pf=ev
print("prior-predictive mean counts",mu.mean(0).round(1),"sd across particles",mu.std(0).round(1))
print("P(stall)",pst.mean().round(2),"FU med",fu.mean().round(1),"P(IA continue)",pc.mean().round(2),"eff stop",pe.mean().round(2),"fut stop",pf.mean().round(2))
# sanity of cohort at median-prior parameters
xm=np.median(X,0)[None]; xm[0,pm.IDX['delta']]=0.5
S=pm.sim_block(xm,pm.make_Z(400,seed=5))
arm=S['arm'][0]; T=S['T'][0]; rel=S['relapsed'][0]; age=S['age'][0]; u=S['u'][0]
print("arms per trial", arm.sum(1).mean().round(1), "GPS /127; mean age",age.mean().round(1),"P(age<65)",(age<65).mean().round(2),"mean u (mo after CR2)",u.mean().round(2))
for nm,m in (("BAT",~arm),("GPS",arm)):
    Tm=np.where(m,T,np.nan)
    print(nm,"OS S12",np.nanmean(np.where(m,(T>12),np.nan)).round(2),"S24",np.nanmean(np.where(m,(T>24),np.nan)).round(2),"S36",np.nanmean(np.where(m,(T>36),np.nan)).round(2),"median",np.nanmedian(Tm).round(1),"| relapsed-before-death",np.nanmean(np.where(m,rel,np.nan)).round(2),"SCT",np.nanmean(np.where(m,S['sct'][0],np.nan)).round(2))
