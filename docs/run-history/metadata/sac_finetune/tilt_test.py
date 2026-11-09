"""Base tilt (gravity leaning) at rig-like noise 0.8 rad/s, 18 ms: does it reproduce arm walk-off?"""
import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS','1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.robustness import stress_env, ObservationNoise
    from pendulum_lab.controllers import load_controller
    from train_rig_rl import RIG
    from finetune_real import RandomTilt
    name, alg, ck, cfg, tilt = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': 0.018}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = ObservationNoise(RandomTilt(base, tilt) if tilt else base, 0.003, 0.8)
    ok = 0; holds = []; clamps = []; maxarm = []
    for e in range(12):
        if tilt: env.env.max_tilt = math.radians(tilt)
        o, _ = env.reset(seed=6000 + e)
        if tilt:  # fixed magnitude, random heading
            g = 9.81 * 0 + float(np.linalg.norm(p.model.opt.gravity)); h = 2*math.pi*e/12; t = math.radians(tilt)
            p.model.opt.gravity[:] = (g*math.sin(t)*math.cos(h), g*math.sin(t)*math.sin(h), -g*math.cos(t))
        c.reset(); th=[]; w=[]; cl=[]; ma=0; run=0; best=0
        for k in range(750):
            a = c.act(o); req = getattr(c, 'requested_action', a)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t_ = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi); th.append(t_); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if abs(t_) < math.radians(15):
                run += 1; best = max(best, run); cl.append(abs(req - a) > 0.2); ma = max(ma, abs(float(p.data.qpos[p._motor_qpos_addr])))
            else: run = 0
            if te: break
        th=np.array(th); w=np.array(w)
        ok += len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        holds.append(best/50); clamps.append(np.mean(cl) if cl else 0); maxarm.append(math.degrees(ma))
    return name, tilt, ok, np.mean(holds), np.mean(clamps), np.mean(maxarm)
CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),
    ('V1','sac','runs/sac_finetune/V1/policy.zip','configs/sac_my_rig.json')]
if __name__ == '__main__':
    extra = sys.argv[1:]
    CK += [(os.path.basename(os.path.dirname(x)) + ':' + os.path.basename(x)[:-4], 'sac', x, 'configs/sac_my_rig.json') for x in extra]
    tilts = [0.0, 0.6, 1.0, 1.5]
    with Pool(16) as pool:
        for n,tl,ok,h,cl,ma in pool.map(job, [(*c, t) for c in CK for t in tilts]):
            print(f'{n:28s} tilt {tl:.1f}° pass {ok:2d}/12 longest hold {h:5.2f}s limiter overrides {cl:5.1%} max|arm| upright {ma:4.0f}°', flush=True)
