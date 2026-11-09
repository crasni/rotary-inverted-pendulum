"""Diagnostic in simulation: LQR capture, then LQR or SAC balance. Per-capture hold times."""
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
    name, alg, cfg, kick, jitter, seeds = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(SWING))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    p._accel_transport.zero_cross_kick_rad_s = kick
    env = CommandNoise(base, (1.0, 1.0)) if jitter else base
    passes = 0; holds = []
    for sd in seeds:
        o, _ = env.reset(seed=sd); c.reset(); th=[]; w=[]; start=None
        for k in range(750):
            a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            b = bool(c.balancing)
            if b and start is None: start = k
            if not b and start is not None: holds.append((k-start)/50); start = None
            if te: break
        if start is not None: holds.append((len(th)-start)/50)
        th=np.array(th); w=np.array(w)
        passes += len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
    return name, (kick, jitter), passes, len(seeds), holds
CK=[('LQR','lqr','configs/lqr_my_rig_coupling_delay.json'),('LQR dither 0.10','custom','configs/custom_lqr_dither_10.json'),('LQR dither 0.15','custom','configs/custom_lqr_dither_15.json')] + 0*[
    (f'{n}/{m}','custom',f'configs/custom_sacbal_{n}_{m}.json') for n in ('sac','v1') for m in ('raw','observer','matched')]
if __name__=='__main__':
    variants=[(0.0,False),(0.2,True)]
    jobs=[(*c,*v,range(7100+8*i,7108+8*i)) for c in CK for v in variants for i in range(2)]
    with Pool(16) as pool: out=pool.map(job,jobs)
    agg={}
    for n,v,ps,ns,h in out:
        a=agg.setdefault((n,v),[0,0,[]]); a[0]+=ps; a[1]+=ns; a[2]+=h
    for (n,v),(ps,ns,h) in agg.items():
        h=np.array(h)
        print(f'{n:14s} kick {v[0]} jitter {int(v[1])}: pass {ps:2d}/{ns}  captures {len(h):3d}  held >=2s {np.mean(h>=2):.0%}  median hold {np.median(h):5.2f}s', flush=True)
