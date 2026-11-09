"""Does a small pendulum zero offset make the arm walk into the limiter? (rig-like 0.8 noise, 18 ms)"""
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
    class Offset(ObservationNoise):
        off = 0.0
        def noisy_frame(self, obs):
            f = super().noisy_frame(obs); th = math.atan2(f[1], f[2]) + self.off
            f[1:3] = [math.sin(th), math.cos(th)]; return f
    name, alg, ck, cfg, off = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': 0.018}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = Offset(base, 0.003, 0.8); env.off = math.radians(off)
    ok = 0; holds = []; clamps = []; maxarm = []
    for e in range(10):
        o, _ = env.reset(seed=5000 + e); c.reset(); th=[]; w=[]; cl=[]; ma=0; run=0; best=0
        for k in range(750):
            a = c.act(o); req = getattr(c, 'requested_action', a)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi); th.append(t); w.append(float(p.data.qvel[p._pen_qvel_addr]))
            if abs(t) < math.radians(15):
                run += 1; best = max(best, run); cl.append(abs(req - a) > 0.2); ma = max(ma, abs(float(p.data.qpos[p._motor_qpos_addr])))
            else: run = 0
            if te: break
        th=np.array(th); w=np.array(w)
        ok += len(th)==750 and bool(np.all(np.abs(th[-500:])<math.radians(15)) and np.all(np.abs(w[-500:])<4))
        holds.append(best/50); clamps.append(np.mean(cl) if cl else 0); maxarm.append(math.degrees(ma))
    return name, off, ok, np.mean(holds), np.mean(clamps), np.mean(maxarm)
OFFS = [float(x) for x in sys.argv[1:]] or [0.0, 1.0, -1.0, 2.0]
if __name__ == "__main__":
    CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),
        ('V1','sac','runs/sac_finetune/V1/policy.zip','configs/sac_my_rig.json')]
    jobs=[(*c, off) for c in CK for off in OFFS]
    with Pool(8) as pool:
        for n,off,ok,h,cl,ma in pool.map(job, jobs):
            print(f'{n:4s} offset {off:+.0f}° pass {ok}/10 longest hold {h:5.2f}s limiter overrides {cl:5.1%} max|arm| upright {ma:4.0f}°')
