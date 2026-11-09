"""Arm position while balancing in sim, and how often the arm limiter overrides the policy."""
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
    name, alg, ck, cfg, vn, delay = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': delay}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = ObservationNoise(base, 0.003 if vn else 0.0, vn)
    out = []
    for e in range(10):
        o, _ = env.reset(seed=5000 + e); c.reset(); arms=[]; ths=[]; clamp=[]; ok=True
        for k in range(750):
            a = c.act(o); req = getattr(c, 'requested_action', a)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            t = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2*math.pi)
            arm = float(p.data.qpos[p._motor_qpos_addr])
            if abs(t) < math.radians(10): arms.append(arm); ths.append(t); clamp.append(abs(req - a) > 0.2)
            if te: ok=False; break
        arms=np.degrees(np.array(arms)); out.append((np.abs(arms).max() if len(arms) else np.nan, np.mean(np.abs(arms)) if len(arms) else np.nan, np.degrees(np.mean(ths)) if ths else np.nan, np.mean(clamp) if clamp else np.nan))
    o = np.array(out)
    return name, vn, delay, np.nanmean(o[:,0]), np.nanmax(o[:,0]), np.nanmean(o[:,1]), np.nanmean(o[:,2]), np.nanmean(o[:,3])
if __name__ == '__main__':
    CK=[('LQR','lqr','docs/results/sac_focused_policy.zip','configs/lqr_my_rig_coupling_delay.json'),
        ('V1','sac','runs/sac_finetune/V1/policy.zip','configs/sac_my_rig.json')]
    jobs=[(*c, vn, d) for c in CK for vn in (0.0, 0.8) for d in (0.010, 0.018)]
    with Pool(8) as pool:
        for n,vn,d,mx,mxx,mean,th,cl in pool.map(job, jobs):
            print(f'{n:5s} noise {vn} delay {d*1e3:.0f}ms | upright: max|arm| avg {mx:5.1f}° worst {mxx:5.1f}° mean|arm| {mean:5.1f}° mean th {th:+.2f}° limiter overrides {cl:.1%}')
