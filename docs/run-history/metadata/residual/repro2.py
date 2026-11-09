import sys, os; sys.path.insert(0,'runs/residual'); os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
import repro
if __name__=='__main__':
    CK=[('SAC current','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json'),
        ('SAC A','sac','runs/sac_real_starts/A/policy.zip','configs/sac_my_rig.json'),
        ('SAC F','sac','runs/sac_real_starts/F/policy.zip','configs/sac_my_rig.json'),
        ('SAC J','sac','runs/sac_residual/J/policy.zip','configs/sac_my_rig.json'),
        ('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json')]
    with Pool(15) as pool:
        for n,g,ok,f,sl in pool.map(repro.job,[(*c,g) for c in CK for g in (0.7,1.0,1.3)]): print(f'{n:12s} gated correction {g}: pass {ok}/8 | median first arrival {f:.1f} | slow caught {sl}')
    print('RIG: SAC current 0/13, A 0/3, F 0/3, J 0/1 (never reached top) | LQR 4/17, slow caught 10/21')
