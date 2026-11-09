import sys, numpy as np
sys.path.insert(0,'runs/sac_real_starts')
from multiprocessing import Pool
from drift_test import run
CKS=sys.argv[1:]
def job(args):
    ck,d=args
    import torch; torch.set_num_threads(1)
    o=run(ck,0.03,d,eps=6)
    return ck,d,sum(r['success'] for r,_ in o),np.nanmean([s for _,s in o]),np.mean([r['longest_balance_s'] for r,_ in o])
if __name__=='__main__':
    jobs=[(c,d) for c in CKS for d in (0.030,0.036,0.042)]
    with Pool(14) as p:
        for ck,d,ok,sat,lb in p.map(job,jobs):
            print(f'{ck[-45:]:45s} delay {d*1e3:.0f} ms: pass {ok}/6  full-strength upright {sat:.0%}  longest {lb:.1f}s')
