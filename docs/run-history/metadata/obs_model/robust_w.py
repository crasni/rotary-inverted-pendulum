import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune','runs/obs_model']; os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    name, alg, cfg, ck, case = arg
    lag, kick, jit, delay, inertia = case[1:]
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': delay}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    if inertia != 1.0:
        p.model.body_inertia[p.model.body('pendulum').id] *= inertia
    env = CommandNoise(base, (jit, jit)) if jit else base
    ok = 0
    for sd in range(9100, 9112):
        o, _ = env.reset(seed=sd); p._arm_lag_s = lag; p._accel_transport.zero_cross_kick_rad_s = kick; p.seed_measurement_ring()
        c.reset(); th=[]; w=[]
        for k in range(750):
            a = c.act(o); o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            th.append(math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi)); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if te: break
        th=np.array(th); w=np.array(w)
        ok += len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
    return name, case[0], ok
CASES = [('original sim', 0.0, 0.0, 0, 0.018, 1.0), ('lag 18 no kick/jitter', 0.018, 0.0, 0, 0.018, 1.0),
         ('lag 26', 0.026, 0.2, 1.0, 0.018, 1.0), ('lag 30', 0.030, 0.2, 1.0, 0.018, 1.0),
         ('lag 18 + delay 30', 0.018, 0.2, 1.0, 0.030, 1.0), ('lag 18 jitter x2', 0.018, 0.2, 2.0, 0.018, 1.0),
         ('lag 18 kick 0.6', 0.018, 0.6, 1.0, 0.018, 1.0), ('lag 18 inertia +15%', 0.018, 0.2, 1.0, 0.018, 1.15)]
if __name__ == '__main__':
    CK = [('LQR','lqr','configs/lqr_my_rig_coupling_delay.json','docs/results/sac_focused_policy.zip')] + [(c.split('/')[2],'sac','configs/sac_my_rig.json',c) for c in sys.argv[1:]]
    with Pool(16) as pool: out = pool.map(job, [(*c, cs) for c in CK for cs in CASES])
    for c in CK:
        print(f'{c[0]:6s} ' + ' | '.join(f'{cs[0]} {ok}/12' for n, cn, ok in out if n == c[0] for cs in CASES if cs[0] == cn), flush=True)
