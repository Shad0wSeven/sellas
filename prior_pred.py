import numpy as np, patient_model as pm
rng=np.random.default_rng(1); n=300
X=pm.sample_prior(n,rng); Z=pm.make_Z(60,seed=3)
med=[];s12=[];s36=[];g36=[];gmed=[];rfs12=[]
for i in range(0,n,30):
    S=pm.sim_block(X[i:i+30],Z)
    for j in range(30):
        arm=S['arm'][j]; T=S['T'][j]
        b=~arm
        med.append(np.median(T[b])); s12.append((T[b]>12).mean()); s36.append((T[b]>36).mean())
        g36.append((T[arm]>36).mean()); gmed.append(np.median(T[arm]))
q=lambda x:np.percentile(x,[5,25,50,75,95]).round(2)
print("PRIOR-PREDICTIVE BAT OS median",q(med)," S12",q(s12)," S36",q(s36))
print("PRIOR-PREDICTIVE GPS OS median",q(gmed)," S36",q(g36))
print("literature targets: BAT median ~12.5 (10-16), S12 ~.50-.55, S36 ~.18 (.11-.30) ; VDM bridge S12 .47-.55")
