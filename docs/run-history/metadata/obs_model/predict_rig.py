"""Held-out prediction: with the fitted command-dependent jitter (fit excludes V1),
what does the simulator predict for each controller? Compare with the rig."""
import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
W2=11.30**2; TOP=2*W2
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    name, alg, ck, cfg, scale, seeds = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = CommandNoise(base, (scale, scale)) if scale else base
    res = []
    for sd in seeds:
        o, _ = env.reset(seed=sd); c.reset(); th=[]; w=[]; run=0; best=0
        for k in range(750):
            a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            run = run + 1 if abs(t) < math.radians(15) else 0; best = max(best, run)
            if te: break
        th=np.array(th); w=np.array(w)
        ok = len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        E = np.median((0.5*w[150:]**2 + W2*(1-np.cos(th[150:]+math.pi)))/TOP)
        res.append((ok, best/50, E))
    return name, scale, res
CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),
    ('SAC orig','sac','docs/results/rig_rl_disturbance/sac_long/policy.zip','configs/sac_my_rig.json'),
    ('R50','sac','runs/sac_finetune/R50/policy.zip','configs/sac_my_rig.json'),
    ('V1','sac','runs/sac_finetune/V1/policy.zip','configs/sac_my_rig.json')]
if __name__=='__main__':
    extra = sys.argv[1:]
    CK += [(x, 'sac', x, 'configs/sac_my_rig.json') for x in extra]
    scales = [0.0, 1.0] if not extra else [0.0, 1.0, 1.5]
    jobs = [(*c, sc, range(7000+8*i, 7008+8*i)) for c in CK for sc in scales for i in range(2)]
    with Pool(16) as pool:
        out = pool.map(job, jobs)
    agg = {}
    for n, sc, r in out: agg.setdefault((n, sc), []).extend(r)
    for (n, sc), r in agg.items():
        r = np.array(r)
        print(f'{n[-34:]:34s} jitter x{sc}: pass {int(r[:,0].sum()):2d}/{len(r)} longest hold median {np.median(r[:,1]):5.2f}s  median energy {np.median(r[:,2]):.2f}x', flush=True)
