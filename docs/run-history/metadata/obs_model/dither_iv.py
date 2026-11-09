"""Instrumental-variable (2SLS) identification of near-upright dynamics from dithered LQR holds.
theta_ddot(k) = c1 sin(theta) + c2 pen speed + c3 arm speed + sum_L g_L u[k-1-L];
instruments: the known dither at lags 1..10 plus the regressors' own past (k-6) values."""
import sys, glob, json, math
sys.path[:0] = ['src', 'scripts', 'runs/obs_model', 'runs/sac_finetune']
import numpy as np
from pathlib import Path
import velocity_check as V
from pendulum_lab.algorithms.custom import dither_sequence
LAGS = 4; ZL = 10
names = ['theta[k]', 'theta[k-1]'] + [f'u[k-{L}]' for L in range(LAGS)]

def rows(t, S, mode, dither):
    """S columns: arm, theta, arm speed, pen speed, sent action."""
    n_bal = np.cumsum(mode == 1) - (mode == 1)            # balance ticks before k
    dz = np.where(mode == 1, dither[np.minimum(n_bal, len(dither) - 1)], 0.0)
    th = S[:, 1]
    X, Z, y = [], [], []
    for k in range(ZL + 6, len(S) - 1):
        if abs(th[k]) > math.radians(12) or not np.all(mode[k - ZL - 1:k + 2] == 1): continue
        if np.any(np.abs(np.diff(th[k - 6:k + 2])) > 1.0): continue   # no wrap inside the window
        X.append([th[k], th[k - 1], *[S[k - L, 4] for L in range(LAGS)]])
        Z.append([*[dz[k - L] for L in range(ZL)], th[k - 3], th[k - 4], th[k - 5]])
        y.append(th[k + 1])
    return X, Z, y

def tsls(X, Z, y):
    X, Z, y = map(np.asarray, (X, Z, y))
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    c = np.linalg.lstsq(Xh, y, rcond=None)[0]
    rng = np.random.default_rng(0); B = []; n = len(y); blk = 50
    for _ in range(200):
        idx = np.concatenate([np.arange(i, min(i + blk, n)) for i in rng.integers(0, n, max(1, n // blk))])
        Xb = Z[idx] @ np.linalg.lstsq(Z[idx], X[idx], rcond=None)[0]
        B.append(np.linalg.lstsq(Xb, y[idx], rcond=None)[0])
    return c, np.std(B, 0), n

def show(label, X, Z, y):
    c, sd, n = tsls(X, Z, y)
    poles = np.roots([1, -c[0], -c[1]])
    print(f'{label} (n={n}): poles {np.round(np.sort(np.abs(poles))[::-1], 3)} -> unstable rate {np.log(np.max(np.abs(poles)))/0.02:.1f}/s; command sum {c[2:].sum():+.4f}')
    print('   ' + '  '.join(f'{nm} {ci:+.4f}±{s:.4f}' for nm, ci, s in zip(names, c, sd)), flush=True)

def sim(cfg, kick, jitter, seeds, amplitude, delay=0.018):
    from pendulum_lab.controllers import load_controller
    from pendulum_lab.robustness import stress_env
    from train_rig_rl import RIG
    from finetune_real import CommandNoise
    c = load_controller('custom', Path('configs/my_rig.json'), Path(cfg), Path('docs/results/sac_focused_policy.zip'))
    base = stress_env(Path('configs/my_rig.json'), {**RIG, 'delay_s': delay}); p = base.unwrapped; p.exact_hanging_reset_frac = 0.0
    p._accel_transport.zero_cross_kick_rad_s = kick
    env = CommandNoise(base, (1, 1)) if jitter else base
    dither = dither_sequence(11, amplitude)
    X, Z, y = [], [], []
    for sd in seeds:
        o, _ = env.reset(seed=sd); c.reset(); S = []; M = []
        for k in range(750):
            a = c.act(o); f = o[-6:]
            S.append([f[0], math.atan2(f[1], f[2]), f[3], f[4], a]); M.append(1 if c.balancing else 0)
            o, _, te, _, _ = env.step(np.array([a], dtype=np.float32))
            if te: break
        x, z, yy = rows(np.arange(len(S)) * 0.02, np.array(S), np.array(M), dither); X += x; Z += z; y += yy
    return X, Z, y

def rig(pattern, amplitude):
    dither = dither_sequence(11, amplitude); X, Z, y = [], [], []
    for r in sorted(glob.glob(pattern)):
        d = np.load(r + '/episode_000.npz', allow_pickle=True); s = d['samples']; T = d['hardware_timing']
        if len(s) < 60 or len(T) != len(s): continue
        t = (np.unwrap(T[:, 1].astype(float), period=2**32) - T[0, 1]) * 1e-6
        x, z, yy = rows(t, np.c_[s[:, 1:5], s[:, 5]], T[:, 6].astype(int), dither); X += x; Z += z; y += yy
    return X, Z, y

if __name__ == '__main__':
    if sys.argv[1] == 'sim':
        amp = float(sys.argv[2]); cfg = f'configs/custom_lqr_dither_{int(amp*100):02d}.json'
        print('truth (linear model): poles 1.278 and 0.782 -> unstable rate 12.3/s')
        show(f'SIM dither {amp} clean', *sim(cfg, 0.0, False, range(7800, 7812), amp))
        show(f'SIM dither {amp} kick+jitter', *sim(cfg, 0.2, True, range(7800, 7812), amp))
        for dl in (0.008, 0.030):
            show(f'SIM dither {amp} clean, delay {dl*1e3:.0f} ms', *sim(cfg, 0.0, False, range(7800, 7812), amp, delay=dl))
    else:
        show(f'RIG {sys.argv[2]}', *rig(sys.argv[2], float(sys.argv[3])))
