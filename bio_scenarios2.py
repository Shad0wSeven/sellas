import json, pickle, numpy as np
from bio_lib import run
import regal_bio as b
b.BIO["Mr"], b.BIO["kr"] = json.load(open("calib_bio.json"))["Mr"], json.load(open("calib_bio.json"))["kr"]
T = {6: .45, 12: .27}
res = []
for nm, ov, ex in [("RFS 36mo 22% (40/17/2 pattern)", {}, {**T, 36: .22}), ("RFS 36mo 17% (VDM/Kurosawa-consistent)", {}, {**T, 36: .17}), ("RFS 36mo 14%", {}, {**T, 36: .14}), ("RFS 36mo 10%", {}, {**T, 36: .10}),
                   ("RFS36 17% + enrollment gamma 1.5", dict(gamma=1.5), {**T, 36: .17}), ("RFS36 17% + enrollment gamma 2.4", dict(gamma=2.4), {**T, 36: .17}), ("RFS36 17% + enrollment gamma 2.8", dict(gamma=2.8), {**T, 36: .17}),
                   ("RFS36 17% + post-relapse 7 mo", dict(m_p=7.0), {**T, 36: .17}), ("RFS36 17% + 50% observation BAT", dict(p_act=0.5), {**T, 36: .17}),
                   ("RFS36 17% + frailty 0.3", dict(sigma_f=0.3), {**T, 36: .17})]:
    res.append(run(nm, bio_over=ov, rfs_extra=ex))
pickle.dump(res, open("results_v2/bio_scenarios2.pkl", "wb"))
