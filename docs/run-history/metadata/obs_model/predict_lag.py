"""Does a physical arm lag behind the step count make the simulator predict the rig?
Rig: LQR captures held >=1 s 43% (passes ~22%); SAC-balance captures >=1 s ~4%; standalone SAC/V1 0 passes."""
import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
SWING='docs/results/sac_focused_policy.zip'
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    name, alg, cfg, ck, lag, kick, jitter, seeds = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = CommandNoise(base, (1, 1)) if jitter else base
    passes = 0; holds = []
    for sd in seeds:
        o, _ = env.reset(seed=sd); p._arm_lag_s = lag; p._accel_transport.zero_cross_kick_rad_s = kick
        p.seed_measurement_ring(); o = p._obs() if not jitter else o
        c.reset(); th=[]; w=[]; start=None; run=0; best=0
        for k in range(750):
            a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            b = getattr(c, 'balancing', None)
            if b is None:  # standalone policy: count upright stretches as holds
                b = abs(t) < math.radians(15)
            if b and start is None: start = k
            if not b and start is not None: holds.append((k-start)/50); start = None
            if te: break
        if start is not None: holds.append((len(th)-start)/50)
        th=np.array(th); w=np.array(w)
        passes += len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
    return name, (lag, kick, jitter), passes, len(seeds), holds
CK=[('LQR','lqr','configs/lqr_my_rig_coupling_delay.json',SWING),
    ('SAC-balance (orig)','custom','configs/custom_sacbal_sac_raw.json',SWING),
    ('V1-balance','custom','configs/custom_sacbal_v1_raw.json',SWING),
    ('SAC standalone','sac','configs/sac_my_rig.json','docs/results/rig_rl_disturbance/sac_long/policy.zip'),
    ('V1 standalone','sac','configs/sac_my_rig.json','runs/sac_finetune/V1/policy.zip')]
if __name__=='__main__':
    lags = [float(x) for x in sys.argv[1:]] or [0.0, 0.006, 0.010, 0.014]
    variants=[(l, 0.2, True) for l in lags]
    jobs=[(*c,*v,range(8100+8*i,8108+8*i)) for c in CK for v in variants for i in range(2)]
    with Pool(16) as pool: out=pool.map(job,jobs)
    agg={}
    for n,v,ps,ns,h in out:
        a=agg.setdefault((n,v),[0,0,[]]); a[0]+=ps; a[1]+=ns; a[2]+=h
    for (n,v),(ps,ns,h) in sorted(agg.items(), key=lambda kv:(kv[0][1][0], kv[0][0])):
        h=np.array(h) if h else np.zeros(1)
        print(f'arm lag {v[0]*1e3:4.0f} ms | {n:19s}: pass {ps:2d}/{ns}  captures {len(h):3d}  held >=1s {np.mean(h>=1):4.0%}  >=2s {np.mean(h>=2):4.0%}', flush=True)
