"""Start the simulator from the first tick of each real capture (LQR rig runs and SAC-balance rig runs),
with the in-flight command preloaded, and let a balance controller run 2 s. Does SAC hold from these
rough states in simulation? (LQR controller and SAC policy, with/without kick+jitter.)"""
import sys, glob, json, math, os
sys.path[:0] = ['src', 'scripts', 'runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import real_state_starts as R

def capture_starts():
    runs = [r for r in sorted(glob.glob('docs/results/hardware/lqr/2026*')) if json.load(open(r + '/run.json'))['arguments'].get('config', '').endswith('lqr_my_rig_coupling_delay.json')]
    runs += sorted(glob.glob('runs/custom_*raw_*')) + sorted(glob.glob('runs/custom_v1matched_*'))
    out = []
    for r in runs:
        d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']; mode = d['hardware_timing'][:, 6]
        for k in range(4, len(s)):
            if mode[k] == 1 and mode[k - 1] != 1:
                out.append((r, k))
    return out

def job(arg):
    import torch; torch.set_num_threads(1)
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    name, alg, cfg, ck, kick, jitter = arg
    c = load_controller(alg, Path('configs/my_rig.json'), Path(cfg) if cfg else None, Path(ck))
    base = stress_env(Path('configs/my_rig.json'), RIG); base.reset(seed=0); p = base.unwrapped
    p._accel_transport.zero_cross_kick_rad_s = kick
    env = CommandNoise(base, (1, 1)) if jitter else base
    if jitter: env.reset(seed=1)
    held = []; by = []
    for r, k in capture_starts():
        s, a = R.load_run(r)
        o = R.set_state(base, s, a, k)
        lim = p._motor_max_accel_rad_s2
        acc = float(np.clip(a[k - 1] * p.max_accel_rad_s2, -lim, lim))
        p._accel_transport.reset(position_rad=float(s[k, 0]), velocity_rad_s=float(s[k, 2]) + acc * 0.004, active_accel_rad_s2=acc)
        if jitter:
            env.history.clear(); env.history.extend([o[i*6:(i+1)*6].copy() for i in range(4)])
        c.reset()
        if hasattr(c, 'lqr') or alg == 'lqr':
            inner = getattr(c, 'lqr', c)
            inner.balancing = True; inner.arm_reference_rad = float(np.clip(s[k, 0], -0.64, 0.64))
        ok = True
        for j in range(100):
            u = c.act(o); o, _, te, _, _ = env.step(np.array([u], dtype=np.float32))
            th = math.remainder(float(p.data.qpos[p._pen_qpos_addr]) - math.pi, 2 * math.pi)
            if te or abs(th) > 0.55: ok = False; break
        held.append(ok); by.append((abs(s[k, 1]), abs(s[k, 3]), abs(s[k, 2]), 'lqr/' in r))
    return name, (kick, jitter), np.array(held), np.array(by)

if __name__ == '__main__':
    CK = [('LQR', 'lqr', 'configs/lqr_my_rig_coupling_delay.json', 'docs/results/sac_focused_policy.zip'),
          ('SAC orig balance', 'custom', 'configs/custom_sacbal_sac_raw.json', 'docs/results/sac_focused_policy.zip'),
          ('V1 balance', 'custom', 'configs/custom_sacbal_v1_raw.json', 'docs/results/sac_focused_policy.zip')]
    jobs = [(*c, kick, jit) for c in CK for kick, jit in ((0.0, False), (0.2, True))]
    with Pool(6) as pool:
        for name, v, held, by in pool.map(job, jobs):
            lqr_states = by[:, 3] == 1
            rough = (by[:, 2] > 2.77)
            print(f'{name:17s} kick {v[0]} jitter {int(v[1])}: held 2 s from real capture states {held.mean():.0%} (n={len(held)}; LQR-run states {held[lqr_states].mean():.0%}, SAC-run states {held[~lqr_states].mean():.0%}; arm speed >2.77: {held[rough].mean():.0%}, below: {held[~rough].mean():.0%})', flush=True)
