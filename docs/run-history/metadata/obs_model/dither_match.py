"""Indirect inference: run the identical dithered-LQR experiment in simulator variants and
compare the IV estimates (unstable rate, command sum, lag shape) with the rig's."""
import sys, os, math, itertools, json
sys.path[:0] = ['src', 'scripts', 'runs/obs_model', 'runs/sac_finetune']; os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
import dither_iv as D

def summary(X, Z, y, boot=200):
    X, Z, y = map(np.asarray, (X, Z, y))
    def est(idx):
        Xh = Z[idx] @ np.linalg.lstsq(Z[idx], X[idx], rcond=None)[0]
        c = np.linalg.lstsq(Xh, y[idx], rcond=None)[0]
        pole = np.max(np.abs(np.roots([1, -c[0], -c[1]])))
        u = c[2:]; s = u.sum()
        return np.array([math.log(pole) / 0.02, s, (np.arange(len(u)) @ u) / s if s else np.nan])
    full = est(np.arange(len(y)))
    rng = np.random.default_rng(0); n = len(y); blk = 50
    B = np.array([est(np.concatenate([np.arange(i, min(i + blk, n)) for i in rng.integers(0, n, n // blk)])) for _ in range(boot)])
    return full, np.percentile(B, [5, 95], axis=0)

def job(args):
    import torch; torch.set_num_threads(1)
    delay, kick, gain = args
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    from pendulum_lab.algorithms.custom import dither_sequence
    from pathlib import Path
    c = load_controller('custom', Path('configs/my_rig.json'), Path('configs/custom_lqr_dither_10.json'), Path('docs/results/sac_focused_policy.zip'))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': delay}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    env = CommandNoise(base, (1, 1))
    dither = dither_sequence(11, 0.1); X, Z, y = [], [], []
    for sd in range(7900, 7912):
        o, _ = env.reset(seed=sd); p._accel_transport.zero_cross_kick_rad_s = kick; p.max_accel_rad_s2 = 50.0 * gain
        c.reset(); S = []; M = []
        for k in range(750):
            a = c.act(o); f = o[-6:]
            S.append([f[0], math.atan2(f[1], f[2]), f[3], f[4], a]); M.append(1 if c.balancing else 0)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            if te: break
        x, z, yy = D.rows(np.arange(len(S)) * 0.02, np.array(S), np.array(M), dither); X += x; Z += z; y += yy
    if len(y) < 200: return args, None, None
    full, ci = summary(X, Z, y, boot=50)
    return args, full, len(y)

if __name__ == '__main__':
    full, ci = summary(*D.rig('runs/custom_lqrdither_*', 0.1))
    print(f'RIG: unstable rate {full[0]:.1f}/s [{ci[0,0]:.1f}, {ci[1,0]:.1f}]  command sum {full[1]:.4f} [{ci[0,1]:.4f}, {ci[1,1]:.4f}]  lag centroid {full[2]:.2f} ticks [{ci[0,2]:.2f}, {ci[1,2]:.2f}]', flush=True)
    grid = list(itertools.product((0.018, 0.026, 0.034), (0.0, 0.2, 0.5), (1.0, 1.3, 1.6)))
    with Pool(16) as pool:
        res = pool.map(job, grid)
    def dist(f):
        return ((f[0] - full[0]) / max(ci[1, 0] - ci[0, 0], 1e-6)) ** 2 + ((f[1] - full[1]) / (ci[1, 1] - ci[0, 1])) ** 2 + ((f[2] - full[2]) / (ci[1, 2] - ci[0, 2])) ** 2
    res = [r for r in res if r[1] is not None]
    res.sort(key=lambda r: dist(r[1]))
    for (d, k, g), f, n in res:
        inside = all(ci[0, i] <= f[i] <= ci[1, i] for i in range(3))
        print(f'SIM delay {d*1e3:.0f}ms kick {k} gain {g}: rate {f[0]:5.1f}/s  sum {f[1]:.4f}  lag {f[2]:.2f}  {"<- inside rig 90% CI on all three" if inside else ""} (n={n})', flush=True)
