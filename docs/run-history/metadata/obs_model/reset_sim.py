"""Reset controller in simulation from end-of-episode states (V1 and W2 episodes, cut at
random times), with the arm-lag/kick/jitter rig model. Reports time to settle and envelope."""
import sys, os, math
sys.path[:0]=['src','scripts','runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from pathlib import Path
from multiprocessing import Pool
def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from pendulum_lab.rig_reset import ResetController
    from pendulum_lab.contracts import state_from_obs
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    ck, lag, seed = arg
    c = load_controller('sac', Path('configs/my_rig.json'), Path('configs/sac_my_rig.json'), Path(ck))
    base = stress_env(Path('configs/my_rig.json'), RIG); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    p.terminate_on_spin_loop = False
    env = CommandNoise(base, (1, 1))
    rc = ResetController(arm_limits=(0.018, 4.4, 1.35))
    rng = np.random.default_rng(seed)
    o, _ = env.reset(seed=seed); p._arm_lag_s = lag; p._accel_transport.zero_cross_kick_rad_s = 0.25; p.seed_measurement_ring(); c.reset()
    cut = int(rng.integers(100, 500)); prev = 0.0
    for k in range(cut):
        prev = c.act(o); o, _, te, tr, _ = env.step(np.array([prev], dtype=np.float32))
        if te: break
    rc.reset(); max_arm = 0; max_arm_speed = 0; ticks = None
    for k in range(1500):  # 30 s budget
        x = state_from_obs(o)
        if rc.update_settled(x): ticks = k; break
        prev = rc.act(x, prev); o, _, te, tr, _ = env.step(np.array([prev], dtype=np.float32))
        max_arm = max(max_arm, abs(float(p.data.qpos[p._motor_qpos_addr]))); max_arm_speed = max(max_arm_speed, abs(float(p.data.qvel[p._motor_qvel_addr])))
        if te: return ck, lag, None, 'terminated', max_arm, max_arm_speed
    return ck, lag, ticks, 'ok' if ticks is not None else 'timeout', max_arm, max_arm_speed
if __name__ == '__main__':
    jobs = [(ck, lag, s) for ck in ('runs/sac_finetune/V1/policy.zip', 'runs/sac_finetune/W_rig/W2_100k.zip') for lag in (0.0, 0.018) for s in range(12)]
    with Pool(16) as pool: out = pool.map(job, jobs)
    t = np.array([o[2] / 50 for o in out if o[2] is not None])
    print(f'settled {len(t)}/{len(out)}; time to settle median {np.median(t):.1f}s p90 {np.percentile(t, 90):.1f}s max {t.max():.1f}s')
    print('failures:', [(o[0].split("/")[-2], o[1], o[3]) for o in out if o[2] is None])
    print(f'max |arm| {math.degrees(max(o[4] for o in out)):.0f} deg, max arm speed {max(o[5] for o in out):.2f} rad/s (guards: 100 deg, 6 rad/s)')
