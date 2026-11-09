"""Standalone SAC checkpoints vs LQR in the arm-lag simulator (held-out seeds 9000+)."""
import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune','runs/obs_model']; os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
import numpy as np
import predict_lag as P
if __name__=='__main__':
    cks = sys.argv[1:]
    CK = [('LQR','lqr','configs/lqr_my_rig_coupling_delay.json',P.SWING)] + [(c.replace('runs/sac_finetune/',''),'sac','configs/sac_my_rig.json',c) for c in cks]
    variants = [(l, 0.2, True) for l in (0.014, 0.018, 0.022)]
    jobs = [(*c,*v,range(9000+8*i,9008+8*i)) for c in CK for v in variants for i in range(2)]
    with Pool(12) as pool: out = pool.map(P.job, jobs)
    agg = {}
    for n,v,ps,ns,h in out:
        a = agg.setdefault(n,{}).setdefault(v[0],[0,0]); a[0]+=ps; a[1]+=ns
    for n, d in agg.items():
        print(f'{n:45s} ' + '  '.join(f'lag {l*1e3:.0f}ms: {ps:2d}/{ns}' for l,(ps,ns) in sorted(d.items())), flush=True)
